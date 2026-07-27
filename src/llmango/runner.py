"""Run orchestration for the sync and batch generation paths.

A run turns one question into validated responses across languages and samples,
writes them to Parquet, and records a manifest. Prompts are rendered per sample
from templates and the question's prompt inputs, which the experiment builds, so
what varies per sample is the experiment's to decide. Reruns with the same
configuration are skipped by matching the manifest content hash, so results are
never duplicated. The batch path splits this into submit and fetch.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llmango.backends.base import (
    BatchBackend,
    GenerationBackend,
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
    find_manifest_by_content_hash,
    manifest_path,
    read_manifest,
    write_manifest,
)
from llmango.pricing import (
    PricingEntry,
    PricingTable,
    compute_cost,
    load_pricing,
    resolve_entry,
    round_usd,
)
from llmango.questions import (
    PromptTemplate,
    QuestionConfig,
    load_question,
    load_template,
)
from llmango.spec import ExperimentSpec, SchemaVariant
from llmango.storage import COST_COLUMNS, USAGE_COLUMNS, results_path, write_results


@dataclass(frozen=True)
class RunOutcome:
    """The result of a run: what was written, or that it was skipped."""

    run_id: str
    manifest: RunManifest
    parquet_path: Path
    manifest_path: Path
    rows_written: int
    skipped: bool
    batch_id: str | None = None


@dataclass(frozen=True)
class RunPlan:
    """A preview of a run: its manifest, any existing duplicate, and the price."""

    manifest: RunManifest
    duplicate: RunManifest | None
    pricing: PricingEntry | None


_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
)

_COST_FIELDS = ("input_cost_usd", "output_cost_usd", "total_cost_usd")


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


def _result_to_row(
    result: GenResult,
    backend_id: str,
    run_id: str,
    spec: ExperimentSpec,
    variant: SchemaVariant,
    pricing_entry: PricingEntry | None,
    batched: bool = False,
) -> dict[str, object]:
    """Combine the common columns, the experiment's extras, provenance and cost."""
    request = result.request
    answer = variant.extract(result.parsed, result.raw_json)
    extra = (
        spec.extra_raw_columns(result.parsed, answer) if spec.extra_raw_columns else {}
    )
    return {
        "question_id": request.question_id,
        "lang": request.lang,
        "schema_variant": request.schema_variant,
        "schema_name": variant.schema_name,
        "model": request.model,
        "backend": backend_id,
        "run_id": run_id,
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
        "request_envelope": result.request_envelope,
        "response_envelope": result.response_envelope,
        **_usage_columns(result.usage),
        **_cost_columns(result.usage, pricing_entry, batched),
        "created_at": result.created_at,
    }


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
        for field in _TOKEN_FIELDS:
            count = row.get(field)
            tokens[field] += count if isinstance(count, int) else 0
        for field in _COST_FIELDS:
            cost = row.get(field)
            costs[field] += cost if isinstance(cost, float) else 0.0
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


@dataclass(frozen=True)
class _PreparedRun:
    """The shared setup both the sync and batch paths build a run from."""

    spec: ExperimentSpec
    manifest: RunManifest
    templates: dict[str, PromptTemplate]
    sources: dict[str, InputSource]


def _prepare(
    question_id: str,
    backend_id: str,
    model: str | None,
    samples: int,
    languages: list[str] | None,
    seed: int | None,
    schema_variant: str | None,
    run_id: str | None,
) -> _PreparedRun:
    """Load the question and build its manifest, ready for the idempotency check.

    An unset schema variant falls back to the experiment's first declared variant,
    so no caller has to name an arm that only some experiments happen to have.

    The run id is derived from the finished manifest, since it embeds the
    configuration's content hash, so it is filled in once the rest is known.
    """
    config = load_question(question_id)
    spec = spec_for(question_id)
    schema_variant = schema_variant or next(iter(spec.schema_variants))
    variant = spec.variant(schema_variant)

    model = model or config.model
    if not model:
        raise ValueError(
            f"No model given and none set in experiment.yaml for {spec.folder}"
        )

    languages = languages or config.languages
    effective_seed = seed if seed is not None else config.sampling.seed
    templates = {
        lang: load_template(spec.folder, config.question_id, lang) for lang in languages
    }
    sources = load_input_sources(spec.folder, config.question_id, list(config.inputs))
    _check_inputs_build(spec, config, sources, languages[0], effective_seed)

    manifest = RunManifest(
        run_id="",
        question_id=config.question_id,
        backend=backend_id,
        model=model,
        schema_variant=schema_variant,
        schema_name=variant.schema_name,
        schema_sha256=variant.schema_sha256,
        languages=languages,
        sampling=config.sampling,
        seed=effective_seed,
        samples_per_language=samples,
        inputs=config.inputs,
        template_sha256={lang: template.sha256 for lang, template in templates.items()},
        input_sha256={name: source.sha256 for name, source in sources.items()},
    )
    manifest.run_id = run_id or build_run_id(manifest)
    return _PreparedRun(
        spec=spec, manifest=manifest, templates=templates, sources=sources
    )


def _check_inputs_build(
    spec: ExperimentSpec,
    config: QuestionConfig,
    sources: dict[str, InputSource],
    lang: str,
    seed: int | None,
) -> None:
    """Build one sample's inputs so a bad declaration fails before anything is spent.

    An experiment validates its own declarations inside build_input, which only
    the request loop reaches. Calling it once here puts that validation back on
    the plan path, where a malformed order or an unknown input name is caught
    while a run is still being priced rather than partway through paying for it.
    """
    resolve(
        spec.build_input,
        sources,
        config.inputs,
        lang,
        0,
        seed,
        config.question_id,
    )


def _skipped_outcome(manifest: RunManifest) -> RunOutcome:
    """Build the outcome returned when an identical run already exists."""
    return RunOutcome(
        run_id=manifest.run_id,
        manifest=manifest,
        parquet_path=results_path(manifest.run_id, manifest.model),
        manifest_path=manifest_path(manifest.run_id),
        rows_written=0,
        skipped=True,
        batch_id=manifest.batch_id,
    )


def _pin_pricing(
    manifest: RunManifest, pricing_table: PricingTable | None
) -> PricingEntry:
    """Resolve the price for the run's model and pin it into the manifest.

    Fails loud before any generation if the model is absent from the pricing
    file, so a paid run never proceeds without a known cost.
    """
    table = pricing_table if pricing_table is not None else load_pricing()
    entry = resolve_entry(table, manifest.model, manifest.model_snapshot)
    manifest.pricing = entry
    return entry


def run(
    question_id: str,
    backend: GenerationBackend,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    schema_variant: str | None = None,
    run_id: str | None = None,
    pricing_table: PricingTable | None = None,
) -> RunOutcome:
    """Generate responses for one question and persist them to Parquet.

    Loads the question config and experiment spec, renders one prompt per
    language and sample, and writes the validated results plus a run manifest.
    If a manifest with the same content hash already exists, the run is skipped
    and nothing is regenerated. The pricing table defaults to the committed
    pricing file and is pinned into the manifest so cost stays reproducible.
    An unset schema variant uses the experiment's first declared variant.

    Transient failures are the OpenAI SDK's to retry; the client is configured
    for that rather than wrapped in a second retry layer here.
    """
    prepared = _prepare(
        question_id,
        backend.backend_id,
        model,
        samples,
        languages,
        seed,
        schema_variant,
        run_id,
    )
    manifest = prepared.manifest

    existing = find_manifest_by_content_hash(manifest.content_hash())
    if existing is not None:
        return _skipped_outcome(existing)

    manifest.model_snapshot = backend.resolve_model_snapshot(manifest.model)
    pricing_entry = _pin_pricing(manifest, pricing_table)
    variant = prepared.spec.variant(manifest.schema_variant)
    requests = _build_requests(
        manifest, prepared.spec, prepared.templates, prepared.sources
    )
    results = [backend.generate(request) for request in requests]
    rows = [
        _result_to_row(
            result,
            manifest.backend,
            manifest.run_id,
            prepared.spec,
            variant,
            pricing_entry,
        )
        for result in results
    ]
    manifest.usage = _run_usage(rows, manifest.languages)

    parquet_path = write_results(
        rows, manifest.run_id, manifest.model, prepared.spec.extra_raw_dtypes
    )
    written_manifest_path = write_manifest(manifest)

    return RunOutcome(
        run_id=manifest.run_id,
        manifest=manifest,
        parquet_path=parquet_path,
        manifest_path=written_manifest_path,
        rows_written=len(rows),
        skipped=False,
    )


def plan_run(
    question_id: str,
    backend_id: str,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    schema_variant: str | None = None,
) -> RunPlan:
    """Build a run's manifest and check for a duplicate without generating anything.

    Takes the backend id rather than a backend so a dry run needs no client and
    no API key. Reuses the same preparation the real run does, so the previewed
    languages and model match what a run would use.
    """
    manifest = _prepare(
        question_id, backend_id, model, samples, languages, seed, schema_variant, None
    ).manifest
    return RunPlan(
        manifest=manifest,
        duplicate=find_manifest_by_content_hash(manifest.content_hash()),
        pricing=_preview_pricing(manifest.model),
    )


def _preview_pricing(model: str) -> PricingEntry | None:
    """Look up a model's price for a dry run, tolerating a missing pricing file."""
    try:
        return resolve_entry(load_pricing(), model, None)
    except (FileNotFoundError, KeyError):
        return None


def submit_batch(
    question_id: str,
    backend: BatchBackend,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    schema_variant: str | None = None,
    run_id: str | None = None,
    pricing_table: PricingTable | None = None,
) -> RunOutcome:
    """Submit one question as an OpenAI batch and record its manifest.

    Nothing is generated inline: the batch is queued and its id stored in the
    manifest so results can be fetched later. Skips submission if an identical
    run already exists, so a batch is never submitted twice. The price is pinned
    into the manifest at submit time so fetch computes cost from the same price.
    """
    prepared = _prepare(
        question_id,
        backend.backend_id,
        model,
        samples,
        languages,
        seed,
        schema_variant,
        run_id,
    )
    manifest = prepared.manifest

    existing = find_manifest_by_content_hash(manifest.content_hash())
    if existing is not None:
        return _skipped_outcome(existing)

    manifest.model_snapshot = backend.resolve_model_snapshot(manifest.model)
    _pin_pricing(manifest, pricing_table)
    manifest.batch_id = backend.submit(
        _build_requests(manifest, prepared.spec, prepared.templates, prepared.sources)
    )
    try:
        written_manifest_path = write_manifest(manifest)
    except OSError as error:
        raise RuntimeError(
            f"Batch {manifest.batch_id} was submitted but its manifest could not "
            f"be saved ({error}). Record this batch id to fetch it later."
        ) from error

    return RunOutcome(
        run_id=manifest.run_id,
        manifest=manifest,
        parquet_path=results_path(manifest.run_id, manifest.model),
        manifest_path=written_manifest_path,
        rows_written=0,
        skipped=False,
        batch_id=manifest.batch_id,
    )


def fetch_batch(run_id: str, backend: BatchBackend) -> RunOutcome:
    """Fetch a submitted batch's results and persist them to Parquet.

    Rebuilds the exact requests from the stored manifest, verifying the templates
    and prompt inputs still hash to the values recorded at submit time before
    writing results. The manifest is rewritten with the usage and cost the batch
    turned out to consume, which submit time could not know; its configuration,
    and so its content hash, is untouched.
    """
    manifest = read_manifest(run_id)
    if manifest.batch_id is None:
        raise ValueError(f"Run {run_id} has no batch to fetch.")

    spec = spec_for(manifest.question_id)
    variant = spec.variant(manifest.schema_variant)
    requests = _requests_from_manifest(manifest, spec)
    results = backend.fetch(manifest.batch_id, requests)
    rows = [
        _result_to_row(
            result,
            manifest.backend,
            manifest.run_id,
            spec,
            variant,
            manifest.pricing,
            batched=True,
        )
        for result in results
    ]
    manifest.usage = _run_usage(rows, manifest.languages)

    parquet_path = write_results(
        rows, manifest.run_id, manifest.model, spec.extra_raw_dtypes
    )
    return RunOutcome(
        run_id=manifest.run_id,
        manifest=manifest,
        parquet_path=parquet_path,
        manifest_path=write_manifest(manifest),
        rows_written=len(rows),
        skipped=False,
        batch_id=manifest.batch_id,
    )


def _requests_from_manifest(
    manifest: RunManifest, spec: ExperimentSpec
) -> list[GenRequest]:
    """Rebuild a run's requests from its manifest, checking inputs still match."""
    templates = {
        lang: load_template(spec.folder, manifest.question_id, lang)
        for lang in manifest.languages
    }
    for lang, template in templates.items():
        if template.sha256 != manifest.template_sha256[lang]:
            raise ValueError(
                f"Template {manifest.question_id}/{lang}.md changed since submit; "
                f"its hash no longer matches the manifest."
            )
    sources = load_input_sources(
        spec.folder, manifest.question_id, list(manifest.inputs)
    )
    for name, source in sources.items():
        if source.sha256 != manifest.input_sha256.get(name):
            raise ValueError(
                f"Prompt input {name} for {manifest.question_id} changed since "
                f"submit; its hash no longer matches the manifest."
            )
    return _build_requests(manifest, spec, templates, sources)


def _build_requests(
    manifest: RunManifest,
    spec: ExperimentSpec,
    templates: dict[str, PromptTemplate],
    sources: dict[str, InputSource],
) -> list[GenRequest]:
    """Render one request per language and sample from the run's templates."""
    variant = spec.variant(manifest.schema_variant)
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
                    response_schema=variant.schema,
                    prompt_inputs=json.dumps(recorded, ensure_ascii=False),
                    schema_variant=manifest.schema_variant,
                )
            )
    return requests
