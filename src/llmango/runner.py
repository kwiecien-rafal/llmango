"""Run orchestration: plan a run from disk, then execute it through a backend."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llmango.backends import backend_for
from llmango.backends.base import Backend, GenRequest, GenResult, Usage
from llmango.config import sha256_text
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
    Cost,
    PricingEntry,
    PricingTable,
    compute_cost,
    load_pricing,
    round_usd,
)
from llmango.questions import Arm, PromptTemplate, Question, load_question
from llmango.spec import ExperimentSpec, answer_field, schema_name
from llmango.storage import results_path, write_results

type Generation = tuple[GenResult, Cost | None]


@dataclass(frozen=True)
class RunPlan:
    """One run, fully built and priced, with nothing sent yet."""

    spec: ExperimentSpec
    manifest: Manifest
    requests: list[GenRequest]


@dataclass(frozen=True)
class RunOutcome:
    """What a run wrote, and where, read off the manifest it wrote it under."""

    manifest: Manifest
    rows_written: int

    @property
    def run_id(self) -> str:
        """The id naming this run's files."""
        return self.manifest.run_id

    @property
    def batch_id(self) -> str | None:
        """The batch this run submitted, None when it generated inline."""
        return self.manifest.batch_id

    @property
    def parquet_path(self) -> Path:
        """Where this run's raw results landed."""
        return results_path(self.manifest.run_id, self.manifest.model)

    @property
    def manifest_path(self) -> Path:
        """Where this run's manifest landed."""
        return manifest_path(self.manifest.run_id)


def plan(
    question_id: str,
    *,
    samples_per_arm: int = 1,
    model: str | None = None,
    languages: list[str] | None = None,
    pricing_table: PricingTable | None = None,
) -> RunPlan:
    """Build one plan for running a question: its manifest, requests and price."""
    question = load_question(question_id)
    spec = question.spec
    model = model or question.model
    arms = _arms_for_languages(question, languages)
    sources = load_input_sources(spec.folder, question_id, list(question.inputs))

    manifest = Manifest(
        question_id=question_id,
        provider=question.provider,
        model=model,
        temperature=question.temperature,
        samples_per_arm=samples_per_arm,
        arms=[_arm_record(arm, question.templates[arm.lang]) for arm in arms],
        inputs=question.inputs,
        input_sha256={name: source.sha256 for name, source in sources.items()},
        pricing=_price(model, pricing_table),
    )
    return RunPlan(
        spec=spec,
        manifest=manifest,
        requests=_build_requests(manifest, spec, arms, question.templates, sources),
    )


def run(
    plan: RunPlan, backend: Backend | None = None, *, batch: bool = False
) -> RunOutcome:
    """Execute a planned run, inline or as a batch to fetch later."""
    manifest = plan.manifest
    if manifest.pricing is None:
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

    if batch:
        manifest.batch_id = backend.submit(plan.requests)
        _write_submitted_manifest(manifest)
        return RunOutcome(manifest=manifest, rows_written=0)

    results = backend.generate_many(plan.requests)
    return _persist(manifest, plan.spec, results, batched=False)


def fetch_batch(run_id: str, backend: Backend | None = None) -> RunOutcome:
    """Fetch a submitted batch's results and persist them to Parquet."""
    manifest = read_manifest(run_id)
    if manifest.batch_id is None:
        raise ValueError(f"Run {run_id} has no batch to fetch.")
    backend = backend or backend_for(manifest.provider)

    submitted = _plan_from_manifest(manifest)
    results = backend.fetch(manifest.batch_id, submitted.requests)
    return _persist(manifest, submitted.spec, results, batched=True)


def _persist(
    manifest: Manifest, spec: ExperimentSpec, results: list[GenResult], batched: bool
) -> RunOutcome:
    """Write a run's rows and its manifest, costing each generation once."""
    generations = [
        (result, _cost(result.usage, manifest.pricing, batched)) for result in results
    ]
    rows = _rows(generations, manifest, spec)
    for arm in manifest.arms:
        arm.usage = _totals(_of_arm(generations, arm))
    manifest.usage = _totals(generations)
    write_results(rows, manifest.run_id, manifest.model, spec.extra_raw_dtypes)
    write_manifest(manifest)
    return RunOutcome(manifest=manifest, rows_written=len(rows))


def _plan_from_manifest(manifest: Manifest) -> RunPlan:
    """Rebuild a submitted run's plan, refusing anything edited since submit."""
    question = load_question(manifest.question_id)
    spec = question.spec
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
    )


def _arms_for_languages(question: Question, requested: list[str] | None) -> list[Arm]:
    """Narrow a question to the requested languages, refusing one it is not asked in."""
    if requested is None:
        return question.arms
    unknown = [lang for lang in requested if lang not in question.templates]
    if unknown:
        raise ValueError(
            f"Question {question.question_id} has no prompt template for "
            f"{', '.join(unknown)}. It declares {', '.join(question.languages)}."
        )
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
            f"{record.label}, so its submitted batch cannot be rebuilt."
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
    """Render a response schema as the JSON stored with the run it was sent in."""
    return schema.model_json_schema() if schema is not None else None


def _answer(parsed: BaseModel | None, raw_json: str | None) -> str:
    """Read the answer off a parsed response, or off free text when there is none."""
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
        for sample_idx in range(manifest.samples_per_arm):
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
    """Look up a model's price, tolerating an absent file so a plan can report it."""
    try:
        table = pricing_table if pricing_table is not None else load_pricing()
    except FileNotFoundError:
        return None
    return table.models.get(model)


def _write_submitted_manifest(manifest: Manifest) -> None:
    """Save a submitted batch's manifest, surfacing its id if the write fails."""
    try:
        write_manifest(manifest)
    except OSError as error:
        raise RuntimeError(
            f"Batch {manifest.batch_id} was submitted but its manifest could not "
            f"be saved ({error}). Record this batch id to fetch it later."
        ) from error


def _rows(
    generations: list[Generation], manifest: Manifest, spec: ExperimentSpec
) -> list[dict[str, object]]:
    """Turn every generation into the row the raw parquet stores it as."""
    response_schemas = {
        record.schema_name: json.dumps(record.response_schema, ensure_ascii=False)
        if record.response_schema is not None
        else None
        for record in manifest.arms
    }
    return [
        _result_to_row(result, cost, manifest, spec, response_schemas)
        for result, cost in generations
    ]


def _result_to_row(
    result: GenResult,
    cost: Cost | None,
    manifest: Manifest,
    spec: ExperimentSpec,
    response_schemas: dict[str | None, str | None],
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
        "service_tier": result.service_tier,
        "provider_created_at": result.provider_created_at,
        "response_schema": response_schemas[schema_name(request.response_schema)],
        "request_envelope": result.request_envelope,
        "response_envelope": result.response_envelope,
        **_usage_columns(result.usage),
        **_cost_columns(cost, manifest.pricing),
        "created_at": result.created_at,
    }


def _cost(
    usage: Usage | None, pricing: PricingEntry | None, batched: bool
) -> Cost | None:
    """Cost one generation, None when its usage or its model's price is missing."""
    if usage is None or pricing is None:
        return None
    return compute_cost(pricing, usage, batched=batched)


def _usage_columns(usage: Usage | None) -> dict[str, object]:
    """Map token usage to its columns, none of them when the provider reported none."""
    if usage is None:
        return {}
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cached_tokens": usage.cached_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _cost_columns(cost: Cost | None, pricing: PricingEntry | None) -> dict[str, object]:
    """Map one generation's cost to its columns, none of them when it has no cost."""
    if cost is None or pricing is None:
        return {}
    return {
        "input_cost_usd": cost.input_cost_usd,
        "output_cost_usd": cost.output_cost_usd,
        "total_cost_usd": cost.total_cost_usd,
        "pricing_version": pricing.last_updated,
    }


def _of_arm(generations: list[Generation], arm: ArmRecord) -> list[Generation]:
    """Select the generations one arm of a run produced."""
    return [
        (result, cost)
        for result, cost in generations
        if (result.request.lang, schema_name(result.request.response_schema))
        == (arm.lang, arm.schema_name)
    ]


def _totals(generations: list[Generation]) -> UsageTotals:
    """Sum generations and their costs into their token and cost totals."""
    usages = [result.usage for result, _ in generations if result.usage is not None]
    priced = [cost for _, cost in generations if cost is not None]
    return UsageTotals(
        errors=sum(result.error is not None for result, _ in generations),
        provider_refusals=sum(result.refusal is not None for result, _ in generations),
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        cached_tokens=sum(usage.cached_tokens for usage in usages),
        reasoning_tokens=sum(usage.reasoning_tokens for usage in usages),
        input_cost_usd=round_usd(sum(cost.input_cost_usd for cost in priced)),
        output_cost_usd=round_usd(sum(cost.output_cost_usd for cost in priced)),
        total_cost_usd=round_usd(sum(cost.total_cost_usd for cost in priced)),
    )
