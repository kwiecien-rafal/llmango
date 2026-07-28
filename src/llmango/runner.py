"""Run orchestration: plan a run from disk, then execute it through a backend.

A run turns one question into validated responses across every arm it declares,
writes them to Parquet, and records a manifest. Prompts are rendered per sample
from templates and the question's prompt inputs, which the experiment builds, so
what varies per sample is the experiment's to decide.

The work is split in two. plan reads everything a run needs, builds every request
and its manifest and resolves the price, all without touching the network, so a
dry run is the same code path stopped before execution. run then picks one of the
backend's two transports, generating inline or submitting a batch to fetch later.
Which transport is a property of the call, not of the run: the manifest records
the provider that answered, and both transports record it the same way.

Running the same configuration twice is deliberate, not an accident to guard
against: every run of a question lands in its own Parquet file and normalize
pools them all, so a repeat is simply more samples of the same arms.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llmango.backends import backend_for
from llmango.backends.base import (
    Backend,
    GenRequest,
    GenResult,
    Usage,
)
from llmango.config import sha256_text
from llmango.experiments import spec_for
from llmango.inputs import InputSource, load_input_sources, render, resolve
from llmango.manifest import (
    ArmRecord,
    Manifest,
    UsageTotals,
    build_run_id,
    manifest_path,
    read_manifest,
    write_manifest,
)
from llmango.pricing import (
    PricingEntry,
    PricingTable,
    compute_cost,
    load_pricing,
    round_usd,
)
from llmango.questions import (
    Arm,
    PromptTemplate,
    Question,
    load_question,
)
from llmango.spec import FREE_TEXT, ExperimentSpec, answer_field, schema_name
from llmango.storage import COST_COLUMNS, USAGE_COLUMNS, results_path, write_results


@dataclass(frozen=True)
class RunPlan:
    """One run, fully built and validated, with nothing sent yet.

    Holding the requests rather than the material to build them is what makes a
    dry run honest: a malformed input declaration or an unrenderable template
    fails while the run is still being priced, not partway through paying for it.

    pricing is None when the model has no entry in the pricing file, which a dry
    run reports and a real run refuses.
    """

    spec: ExperimentSpec
    manifest: Manifest
    requests: list[GenRequest]
    pricing: PricingEntry | None

    @property
    def question_id(self) -> str:
        """The question this run covers."""
        return self.manifest.question_id

    @property
    def provider(self) -> str:
        """The provider this run's question names, which serves every arm."""
        return self.manifest.provider


@dataclass(frozen=True)
class RunOutcome:
    """The result of a run: what was written, and where."""

    run_id: str
    manifest: Manifest
    parquet_path: Path
    manifest_path: Path
    rows_written: int
    batch_id: str | None = None


_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
)

_COST_FIELDS = ("input_cost_usd", "output_cost_usd", "total_cost_usd")


def plan(
    question_id: str,
    *,
    samples: int = 1,
    model: str | None = None,
    languages: list[str] | None = None,
    pricing_table: PricingTable | None = None,
) -> RunPlan:
    """Build one run of a question: its manifest, every request and its price."""
    spec = spec_for(question_id)
    question = load_question(question_id)
    model = model or question.model
    _check_languages(question, languages)
    arms = _selected_arms(question, languages)
    sources = load_input_sources(spec.folder, question_id, list(question.inputs))
    pricing = _price(model, pricing_table)

    manifest = Manifest(
        question_id=question_id,
        provider=question.provider,
        model=model,
        temperature=question.temperature,
        samples=samples,
        arms=[_arm_record(arm, question.templates[arm.lang]) for arm in arms],
        inputs=question.inputs,
        input_sha256={name: source.sha256 for name, source in sources.items()},
    )
    return RunPlan(
        spec=spec,
        manifest=manifest,
        requests=_build_requests(manifest, spec, arms, question.templates, sources),
        pricing=pricing,
    )


def run(
    plan: RunPlan, backend: Backend | None = None, *, batch: bool = False
) -> RunOutcome:
    """Execute a planned run through one of the backend's two transports."""
    manifest = plan.manifest
    if plan.pricing is None:
        raise ValueError(
            f"No pricing for model '{manifest.model}' in the pricing file. Add it "
            f"to data/pricing.json, prices per 1M tokens, before generating."
        )
    backend = backend or backend_for(manifest.provider)

    manifest.created_at = datetime.now(UTC)
    manifest.run_id = build_run_id(manifest)
    if manifest_path(manifest.run_id).exists():
        raise ValueError(
            f"Run {manifest.run_id} already exists, and a run never overwrites "
            f"another one's files."
        )

    manifest.pricing = plan.pricing
    manifest.model_snapshot = backend.resolve_model_snapshot(manifest.model)

    if batch:
        manifest.batch_id = backend.submit(plan.requests)
        _write_submitted_manifest(manifest)
        return _outcome(manifest, rows_written=0)

    results = backend.generate_many(plan.requests)
    rows = _rows(results, manifest, plan.spec, batched=False)
    manifest.usage = _totals(rows)
    write_results(rows, manifest.run_id, manifest.model, plan.spec.extra_raw_dtypes)
    write_manifest(manifest)
    return _outcome(manifest, rows_written=len(rows))


def fetch_batch(run_id: str, backend: Backend | None = None) -> RunOutcome:
    """Fetch a submitted batch's results and persist them to Parquet.

    The batch is collected from the provider its own manifest records, which is
    the only one that holds its id.

    The manifest is rewritten with the usage and cost the batch turned out to
    consume, which submit time could not know; its configuration is untouched.
    """
    manifest = read_manifest(run_id)
    if manifest.batch_id is None:
        raise ValueError(f"Run {run_id} has no batch to fetch.")
    backend = backend or backend_for(manifest.provider)

    submitted = _plan_from_manifest(manifest)
    results = backend.fetch(manifest.batch_id, submitted.requests)
    rows = _rows(results, manifest, submitted.spec, batched=True)
    manifest.usage = _totals(rows)
    write_results(
        rows, manifest.run_id, manifest.model, submitted.spec.extra_raw_dtypes
    )
    write_manifest(manifest)
    return _outcome(manifest, rows_written=len(rows))


def _plan_from_manifest(manifest: Manifest) -> RunPlan:
    """Rebuild a submitted run's plan, checking its inputs still match the manifest.

    A batch is fetched long after it was submitted, so the requests are rebuilt
    from the manifest and the templates, prompt inputs and response schemas behind
    them are verified against what was recorded at submit time. An edited file
    means the rows would no longer describe what was sent.
    """
    spec = spec_for(manifest.question_id)
    question = load_question(manifest.question_id)
    sources = load_input_sources(
        spec.folder, manifest.question_id, list(manifest.inputs)
    )
    for name, source in sources.items():
        if source.sha256 != manifest.input_sha256.get(name):
            raise ValueError(
                f"Prompt input {name} for {manifest.question_id} changed since "
                f"submit; its hash no longer matches the manifest."
            )

    arms = [_arm_for(record, question) for record in manifest.arms]
    return RunPlan(
        spec=spec,
        manifest=manifest,
        requests=_build_requests(manifest, spec, arms, question.templates, sources),
        pricing=manifest.pricing,
    )


def _check_languages(question: Question, requested: list[str] | None) -> None:
    """Reject a language the question is not asked in, naming the ones it is."""
    unknown = [lang for lang in requested or [] if lang not in question.templates]
    if unknown:
        raise ValueError(
            f"Question {question.question_id} has no prompt template for "
            f"{', '.join(unknown)}. It declares {', '.join(question.languages)}."
        )


def _selected_arms(question: Question, requested: list[str] | None) -> list[Arm]:
    """Narrow a question to the arms asked in the requested languages."""
    if requested is None:
        return question.arms
    return [arm for arm in question.arms if arm.lang in requested]


def _arm_record(arm: Arm, template: PromptTemplate) -> ArmRecord:
    """Record one arm as the manifest pins it: its language, schema and prompt."""
    return ArmRecord(
        lang=arm.lang,
        schema_name=schema_name(arm.schema),
        response_schema=_schema_json(arm.schema),
        template_sha256=template.sha256,
    )


def _arm_for(record: ArmRecord, question: Question) -> Arm:
    """Match one recorded arm back to the question, refusing anything edited."""
    arm = next(
        (
            arm
            for arm in question.arms
            if (arm.lang, schema_name(arm.schema)) == (record.lang, record.schema_name)
        ),
        None,
    )
    if arm is None:
        raise ValueError(
            f"Question {question.question_id} no longer asks {record.lang} under "
            f"{record.schema_name or FREE_TEXT}, so its submitted batch cannot be "
            f"rebuilt."
        )
    if question.templates[arm.lang].sha256 != record.template_sha256:
        raise ValueError(
            f"Template {question.question_id}/{arm.lang}.md changed since submit; "
            f"it was edited, so it no longer matches the manifest."
        )
    if _schema_json(arm.schema) != record.response_schema:
        raise ValueError(
            f"The response schema {record.schema_name} changed since submit; it "
            f"no longer matches the one the manifest records."
        )
    return arm


def _schema_json(schema: type[BaseModel] | None) -> dict[str, Any] | None:
    """Render a response schema as the JSON stored with the run it was sent in.

    The free-text arm has no schema, and records none. Keys are left in
    declaration order rather than sorted, because that order is one of the things
    the model sees.
    """
    return schema.model_json_schema() if schema is not None else None


def _answer(parsed: BaseModel | None, raw_json: str | None) -> str:
    """Read the answer off a parsed response, or off free text when there is none.

    An answer schema declares exactly one field, so a parsed response carries its
    answer in the only field it has. Anything unparsed, whether free text by
    design or a refusal, answers with whatever text came back.
    """
    if parsed is None:
        return raw_json or ""
    return str(getattr(parsed, answer_field(type(parsed))))


def _build_requests(
    manifest: Manifest,
    spec: ExperimentSpec,
    arms: list[Arm],
    templates: dict[str, PromptTemplate],
    sources: dict[str, InputSource],
) -> list[GenRequest]:
    """Render one request per arm and sample from the run's templates."""
    requests: list[GenRequest] = []
    for arm in arms:
        template = templates[arm.lang]
        for sample_idx in range(manifest.samples):
            resolved = resolve(
                spec.build_input,
                sources,
                manifest.inputs,
                arm.lang,
                sample_idx,
                manifest.question_id,
            )
            prompt = render(template.text, resolved)
            recorded = {
                name: value.value
                for name, value in resolved.items()
                if value.value is not None
            }
            requests.append(
                GenRequest(
                    question_id=manifest.question_id,
                    lang=arm.lang,
                    model=manifest.model,
                    prompt=prompt,
                    prompt_sha256=sha256_text(prompt),
                    sample_idx=sample_idx,
                    response_schema=arm.schema,
                    temperature=manifest.temperature,
                    prompt_inputs=json.dumps(recorded, ensure_ascii=False),
                )
            )
    return requests


def _price(model: str, pricing_table: PricingTable | None) -> PricingEntry | None:
    """Look up a model's price, tolerating an absent file so a plan can report it.

    A plan is built before a run is authorized, so a missing price is information
    here rather than a failure; run refuses to generate without one. The lookup is
    the configured model id exactly: a price guessed from a similar id is worse
    than a refusal to run.
    """
    try:
        table = pricing_table if pricing_table is not None else load_pricing()
    except FileNotFoundError:
        return None
    return table.models.get(model)


def _write_submitted_manifest(manifest: Manifest) -> None:
    """Save a submitted batch's manifest, surfacing its id if the write fails.

    A batch that the provider accepted but whose manifest never landed can only be
    fetched if its id reaches the operator, so the failure carries it.
    """
    try:
        write_manifest(manifest)
    except OSError as error:
        raise RuntimeError(
            f"Batch {manifest.batch_id} was submitted but its manifest could not "
            f"be saved ({error}). Record this batch id to fetch it later."
        ) from error


def _outcome(manifest: Manifest, rows_written: int) -> RunOutcome:
    """Describe a finished run, whose files are named by its run id and model."""
    return RunOutcome(
        run_id=manifest.run_id,
        manifest=manifest,
        parquet_path=results_path(manifest.run_id, manifest.model),
        manifest_path=manifest_path(manifest.run_id),
        rows_written=rows_written,
        batch_id=manifest.batch_id,
    )


def _rows(
    results: list[GenResult],
    manifest: Manifest,
    spec: ExperimentSpec,
    batched: bool,
) -> list[dict[str, object]]:
    """Turn every generation into the row the raw parquet stores it as.

    Each arm's schema is serialized once for the whole run rather than per row,
    off the manifest itself, so a row and the manifest describing it cannot
    disagree about what was asked.
    """
    serialized = {
        record.schema_name: json.dumps(record.response_schema, ensure_ascii=False)
        if record.response_schema is not None
        else None
        for record in manifest.arms
    }
    return [
        _result_to_row(result, manifest, spec, serialized, batched)
        for result in results
    ]


def _result_to_row(
    result: GenResult,
    manifest: Manifest,
    spec: ExperimentSpec,
    response_schemas: dict[str | None, str | None],
    batched: bool,
) -> dict[str, object]:
    """Combine the common columns, the experiment's extras, provenance and cost."""
    request = result.request
    answer = _answer(result.parsed, result.raw_json)
    schema = schema_name(request.response_schema)
    extra = (
        spec.extra_raw_columns(result.parsed, answer) if spec.extra_raw_columns else {}
    )
    return {
        "question_id": request.question_id,
        "lang": request.lang,
        "schema_name": schema,
        "model": request.model,
        "provider": manifest.provider,
        "run_id": manifest.run_id,
        "sample_idx": request.sample_idx,
        "temperature": request.temperature,
        "prompt_sha256": request.prompt_sha256,
        "prompt": request.prompt,
        "prompt_inputs": request.prompt_inputs,
        "raw_json": result.raw_json,
        "answer": answer,
        **extra,
        "model_snapshot": result.model_snapshot,
        "finish_reason": result.finish_reason,
        "refusal": result.refusal,
        "error": result.error,
        "response_id": result.response_id,
        "system_fingerprint": result.system_fingerprint,
        "service_tier": result.service_tier,
        "provider_created_at": result.provider_created_at,
        "response_schema": response_schemas[schema],
        "request_envelope": result.request_envelope,
        "response_envelope": result.response_envelope,
        **_usage_columns(result.usage),
        **_cost_columns(result.usage, manifest.pricing, batched),
        "created_at": result.created_at,
    }


def _usage_columns(usage: Usage | None) -> dict[str, object]:
    """Map token usage to its columns, all null when usage is missing."""
    if usage is None:
        return {column: None for column in USAGE_COLUMNS}
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cached_tokens": usage.cached_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _cost_columns(
    usage: Usage | None, pricing_entry: PricingEntry | None, batched: bool
) -> dict[str, object]:
    """Compute cost columns from usage and price, null when either is absent."""
    if usage is None or pricing_entry is None:
        return {column: None for column in COST_COLUMNS}
    cost = compute_cost(pricing_entry, usage, batched=batched)
    return {
        "input_cost_usd": cost.input_cost_usd,
        "output_cost_usd": cost.output_cost_usd,
        "total_cost_usd": cost.total_cost_usd,
        "pricing_version": pricing_entry.last_updated,
    }


def _totals(rows: list[dict[str, object]]) -> UsageTotals:
    """Sum a run's rows into its token and cost totals in a single pass.

    Costs are summed from the values written to the parquet, already rounded, so
    the manifest total and the file it describes cannot disagree. A row that
    errored or was refused carries null tokens and null cost, which count as zero.
    """
    tokens = dict.fromkeys(_TOKEN_FIELDS, 0)
    costs = dict.fromkeys(_COST_FIELDS, 0.0)
    errors = 0
    provider_refusals = 0
    for row in rows:
        if row.get("error") is not None:
            errors += 1
        if row.get("refusal") is not None:
            provider_refusals += 1
        for token_field in _TOKEN_FIELDS:
            count = row.get(token_field)
            tokens[token_field] += count if isinstance(count, int) else 0
        for cost_field in _COST_FIELDS:
            cost = row.get(cost_field)
            costs[cost_field] += cost if isinstance(cost, float) else 0.0
    return UsageTotals(
        rows=len(rows),
        errors=errors,
        provider_refusals=provider_refusals,
        prompt_tokens=tokens["prompt_tokens"],
        completion_tokens=tokens["completion_tokens"],
        total_tokens=tokens["total_tokens"],
        cached_tokens=tokens["cached_tokens"],
        reasoning_tokens=tokens["reasoning_tokens"],
        input_cost_usd=round_usd(costs["input_cost_usd"]),
        output_cost_usd=round_usd(costs["output_cost_usd"]),
        total_cost_usd=round_usd(costs["total_cost_usd"]),
    )
