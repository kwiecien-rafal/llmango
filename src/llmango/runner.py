"""Run orchestration: plan a run from disk, then execute it through a backend.

A run turns one question into validated responses across languages and samples,
writes them to Parquet, and records a manifest. Prompts are rendered per sample
from templates and the question's prompt inputs, which the experiment builds, so
what varies per sample is the experiment's to decide.

The work is split in two. plan reads everything a run needs, builds every request
and its manifest and resolves the price, all without touching the network, so a
dry run is the same code path stopped before execution. run then picks one of the
backend's two transports, generating inline or submitting a batch to fetch later.

Running the same configuration twice is deliberate, not an accident to guard
against: every run of a question lands in its own Parquet file and normalize
pools them all, so a repeat is simply more samples of the same arm.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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
    RunManifest,
    RunUsage,
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
    QuestionConfig,
    load_question,
)
from llmango.spec import ExperimentSpec, answer_field
from llmango.storage import COST_COLUMNS, USAGE_COLUMNS, results_path, write_results


@dataclass(frozen=True)
class RunOptions:
    """What a caller chooses about one run, before anything is read from disk.

    backend_id is the provider and transport the run is recorded under, and batch
    picks which of that provider's two transports executes it. The caller supplies
    both because choosing a provider is not this module's decision, and because a
    plan built for a dry run must need no client and no API key.

    Anything left unset falls back to the question's own configuration: its model,
    its languages and its seed. languages narrows each arm to the languages asked
    for rather than replacing them, since an arm is only ever run in the languages
    the question declared under its schema.
    """

    backend_id: str
    model: str | None = None
    samples: int = 1
    languages: list[str] | None = None
    seed: int | None = None
    batch: bool = False


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
    manifest: RunManifest
    requests: list[GenRequest]
    pricing: PricingEntry | None
    batch: bool

    @property
    def question_id(self) -> str:
        """The question this run covers."""
        return self.manifest.question_id


@dataclass(frozen=True)
class RunOutcome:
    """The result of a run: what was written, and where."""

    run_id: str
    manifest: RunManifest
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
    options: RunOptions,
    pricing_table: PricingTable | None = None,
) -> list[RunPlan]:
    """Build one plan per arm of a question: its manifest, requests and price.

    Everything a run depends on is read here once, in one place: the question's
    config and templates, the data behind each prompt input, every rendered
    request and the price of the model. Nothing reaches the network, so this is
    what --dry-run reports and what a real run then executes.

    A question is planned whole because its arms differ only in the schema they
    ask under, so one read of disk covers them all. An arm left with no languages
    once --lang has narrowed it is not planned at all.
    """
    spec = spec_for(question_id)
    config = load_question(question_id)
    model = options.model or config.model
    if not model:
        raise ValueError(
            f"No model given and none set in experiment.yaml for {spec.folder}"
        )
    _check_languages(config, options.languages)
    sources = load_input_sources(spec.folder, question_id, list(config.inputs))
    pricing = _price(model, pricing_table)

    plans: list[RunPlan] = []
    for arm in config.arms:
        languages = _languages_for(arm, options.languages)
        if not languages:
            continue
        templates = {lang: config.templates[lang] for lang in languages}
        manifest = RunManifest(
            question_id=question_id,
            backend=options.backend_id,
            model=model,
            schema_name=_schema_name(arm.schema),
            response_schema=_schema_json(arm.schema),
            languages=languages,
            sampling=config.sampling,
            seed=options.seed if options.seed is not None else config.sampling.seed,
            samples_per_language=options.samples,
            inputs=config.inputs,
            template_sha256={
                lang: template.sha256 for lang, template in templates.items()
            },
            input_sha256={name: source.sha256 for name, source in sources.items()},
        )
        plans.append(
            RunPlan(
                spec=spec,
                manifest=manifest,
                requests=_build_requests(
                    manifest, spec, arm.schema, templates, sources
                ),
                pricing=pricing,
                batch=options.batch,
            )
        )
    return plans


def run(plan: RunPlan, backend: Backend) -> RunOutcome:
    """Execute a planned run through one of the backend's two transports.

    The run is stamped with its id here rather than at plan time, so the id names
    the moment the run actually started. The price is pinned into the manifest
    before anything is generated, so a run never proceeds without a known cost,
    and the model snapshot is resolved so the manifest records exactly which
    revision answered.

    Batched, the requests are submitted and only the manifest is written; its
    results are collected later by fetch_batch. Synchronously, they are generated
    inline and written to Parquet alongside the manifest and its measured usage.

    Transient failures are the provider SDK's to retry; the client is configured
    for that rather than wrapped in a second retry layer here.
    """
    manifest = plan.manifest
    if plan.pricing is None:
        raise ValueError(
            f"No pricing for model '{manifest.model}' in the pricing file. Add it "
            f"to data/pricing.json, prices per 1M tokens, before generating."
        )
    if backend.backend_id != manifest.backend:
        raise ValueError(
            f"This plan was built for backend '{manifest.backend}' but was handed "
            f"'{backend.backend_id}'. Plan and run the same one."
        )

    manifest.created_at = datetime.now(UTC)
    manifest.run_id = build_run_id(manifest)
    if manifest_path(manifest.run_id).exists():
        raise ValueError(
            f"Run {manifest.run_id} already exists, and a run never overwrites "
            f"another one's files."
        )

    manifest.pricing = plan.pricing
    manifest.model_snapshot = backend.resolve_model_snapshot(manifest.model)

    if plan.batch:
        manifest.batch_id = backend.submit(plan.requests)
        _write_submitted_manifest(manifest)
        return _outcome(manifest, rows_written=0)

    results = backend.generate_many(plan.requests)
    rows = _rows(results, manifest, plan.spec, batched=False)
    manifest.usage = _run_usage(rows, manifest.languages)
    write_results(rows, manifest.run_id, manifest.model, plan.spec.extra_raw_dtypes)
    write_manifest(manifest)
    return _outcome(manifest, rows_written=len(rows))


def fetch_batch(run_id: str, backend: Backend) -> RunOutcome:
    """Fetch a submitted batch's results and persist them to Parquet.

    The manifest is rewritten with the usage and cost the batch turned out to
    consume, which submit time could not know; its configuration is untouched.
    """
    manifest = read_manifest(run_id)
    if manifest.batch_id is None:
        raise ValueError(f"Run {run_id} has no batch to fetch.")

    submitted = _plan_from_manifest(manifest)
    results = backend.fetch(manifest.batch_id, submitted.requests)
    rows = _rows(results, manifest, submitted.spec, batched=True)
    manifest.usage = _run_usage(rows, manifest.languages)
    write_results(
        rows, manifest.run_id, manifest.model, submitted.spec.extra_raw_dtypes
    )
    write_manifest(manifest)
    return _outcome(manifest, rows_written=len(rows))


def _plan_from_manifest(manifest: RunManifest) -> RunPlan:
    """Rebuild a submitted run's plan, checking its inputs still match the manifest.

    A batch is fetched long after it was submitted, so the requests are rebuilt
    from the manifest and the templates, prompt inputs and response schema behind
    them are verified against what was recorded at submit time. An edited file
    means the rows would no longer describe what was sent.
    """
    spec = spec_for(manifest.question_id)
    config = load_question(manifest.question_id)
    templates: dict[str, PromptTemplate] = {}
    for lang in manifest.languages:
        template = config.templates.get(lang)
        if template is None or template.sha256 != manifest.template_sha256[lang]:
            raise ValueError(
                f"Template {manifest.question_id}/{lang}.md changed since submit; "
                f"it was edited or removed, so it no longer matches the manifest."
            )
        templates[lang] = template

    sources = load_input_sources(
        spec.folder, manifest.question_id, list(manifest.inputs)
    )
    for name, source in sources.items():
        if source.sha256 != manifest.input_sha256.get(name):
            raise ValueError(
                f"Prompt input {name} for {manifest.question_id} changed since "
                f"submit; its hash no longer matches the manifest."
            )

    arm = _arm_for(config, manifest.schema_name)
    if _schema_json(arm.schema) != manifest.response_schema:
        raise ValueError(
            f"The response schema {manifest.schema_name} changed since submit; it "
            f"no longer matches the one the manifest records."
        )

    return RunPlan(
        spec=spec,
        manifest=manifest,
        requests=_build_requests(manifest, spec, arm.schema, templates, sources),
        pricing=manifest.pricing,
        batch=True,
    )


def _check_languages(config: QuestionConfig, requested: list[str] | None) -> None:
    """Reject a language the question is not asked in, naming the ones it is."""
    unknown = [lang for lang in requested or [] if lang not in config.templates]
    if unknown:
        raise ValueError(
            f"Question {config.question_id} has no prompt template for "
            f"{', '.join(unknown)}. It declares {', '.join(config.languages)}."
        )


def _languages_for(arm: Arm, requested: list[str] | None) -> list[str]:
    """Narrow one arm to the languages asked for, keeping its declared order."""
    if requested is None:
        return arm.languages
    return [lang for lang in arm.languages if lang in requested]


def _arm_for(config: QuestionConfig, schema_name: str | None) -> Arm:
    """Find the arm a submitted run was planned from, by the schema it recorded."""
    for arm in config.arms:
        if _schema_name(arm.schema) == schema_name:
            return arm
    raise ValueError(
        f"Question {config.question_id} no longer asks anything under "
        f"{schema_name or 'no schema'}, so its submitted batch cannot be rebuilt."
    )


def _schema_name(schema: type[BaseModel] | None) -> str | None:
    """The class name a response schema is recorded under, None for free text."""
    return schema.__name__ if schema is not None else None


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
    manifest: RunManifest,
    spec: ExperimentSpec,
    schema: type[BaseModel] | None,
    templates: dict[str, PromptTemplate],
    sources: dict[str, InputSource],
) -> list[GenRequest]:
    """Render one request per language and sample from the run's templates."""
    requests: list[GenRequest] = []
    for lang in manifest.languages:
        template = templates[lang]
        for sample_idx in range(manifest.samples_per_language):
            resolved = resolve(
                spec.build_input,
                sources,
                manifest.inputs,
                lang,
                sample_idx,
                manifest.seed,
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
                    lang=lang,
                    model=manifest.model,
                    prompt=prompt,
                    prompt_sha256=sha256_text(prompt),
                    sample_idx=sample_idx,
                    seed=manifest.seed,
                    sampling=manifest.sampling,
                    response_schema=schema,
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


def _write_submitted_manifest(manifest: RunManifest) -> None:
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


def _outcome(manifest: RunManifest, rows_written: int) -> RunOutcome:
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
    manifest: RunManifest,
    spec: ExperimentSpec,
    batched: bool,
) -> list[dict[str, object]]:
    """Turn every generation into the row the raw parquet stores it as.

    The schema is serialized once for the whole run rather than per row, since
    every row of a run was asked under the same one.
    """
    schema = manifest.response_schema
    serialized = json.dumps(schema, ensure_ascii=False) if schema is not None else None
    return [
        _result_to_row(result, manifest, spec, serialized, batched)
        for result in results
    ]


def _result_to_row(
    result: GenResult,
    manifest: RunManifest,
    spec: ExperimentSpec,
    response_schema: str | None,
    batched: bool,
) -> dict[str, object]:
    """Combine the common columns, the experiment's extras, provenance and cost."""
    request = result.request
    answer = _answer(result.parsed, result.raw_json)
    extra = (
        spec.extra_raw_columns(result.parsed, answer) if spec.extra_raw_columns else {}
    )
    return {
        "question_id": request.question_id,
        "lang": request.lang,
        "schema_name": manifest.schema_name,
        "model": request.model,
        "backend": manifest.backend,
        "run_id": manifest.run_id,
        "sample_idx": request.sample_idx,
        "seed": request.seed,
        "temperature": request.sampling.temperature,
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
        "response_schema": response_schema,
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


def _run_usage(rows: list[dict[str, object]], languages: list[str]) -> RunUsage:
    """Measure what a run consumed, in total and per language."""
    grouped: dict[str, list[dict[str, object]]] = {lang: [] for lang in languages}
    for row in rows:
        group = grouped.get(str(row.get("lang")))
        if group is not None:
            group.append(row)
    return RunUsage(
        measured_at=datetime.now(UTC),
        total=_totals(rows),
        by_language={lang: _totals(group) for lang, group in grouped.items()},
    )


def _totals(rows: list[dict[str, object]]) -> UsageTotals:
    """Sum one group of rows into its token and cost totals in a single pass.

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
