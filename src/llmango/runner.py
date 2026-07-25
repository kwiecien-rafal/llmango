"""Run orchestration for the sync and batch generation paths.

A run turns one question into validated responses across languages and samples,
writes them to Parquet, and records a manifest. Prompts are rendered per sample
from templates and the shared fruit table, so the option order can be fixed or
shuffled per sample. Reruns with the same configuration are skipped by matching
the manifest content hash, so results are never duplicated. The batch path splits
this into submit and fetch.
"""

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from llmango.backends.base import (
    BatchBackend,
    GenerationBackend,
    GenRequest,
    GenResult,
    Usage,
)
from llmango.manifest import (
    RunManifest,
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
    pricing_version,
    resolve_entry,
)
from llmango.questions import (
    FruitTable,
    PromptTemplate,
    load_fruits,
    load_question,
    load_template,
    prompt_sha256,
    render_prompt,
)
from llmango.registry import ExperimentSpec, SchemaVariant, get_experiment
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


def _new_run_id(question_id: str) -> str:
    return f"{question_id}-{uuid.uuid4().hex[:12]}"


def _generate_with_retry(
    backend: GenerationBackend,
    request: GenRequest,
    max_retries: int,
    retry_backoff: float,
) -> GenResult:
    """Generate one result, retrying with linear backoff while it errors."""
    result = backend.generate(request)
    attempt = 0
    while result.error is not None and attempt < max_retries:
        attempt += 1
        time.sleep(retry_backoff * attempt)
        result = backend.generate(request)
    return result


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
    usage: Usage | None, pricing_entry: PricingEntry | None
) -> dict[str, object]:
    """Compute cost columns from usage and price, null when either is absent."""
    if usage is None or pricing_entry is None:
        return {column: None for column in COST_COLUMNS}
    cost = compute_cost(pricing_entry, usage)
    return {
        "input_cost_usd": cost.input_cost_usd,
        "output_cost_usd": cost.output_cost_usd,
        "total_cost_usd": cost.total_cost_usd,
        "pricing_version": pricing_version(pricing_entry),
    }


def _result_to_row(
    result: GenResult,
    backend_id: str,
    run_id: str,
    spec: ExperimentSpec,
    variant: SchemaVariant,
    pricing_entry: PricingEntry | None,
) -> dict[str, object]:
    """Combine the common columns, parsed fields, provenance, usage and cost."""
    request = result.request
    raw_text = variant.extract(result.parsed, result.raw_json)
    parsed_fields = spec.to_row(result.parsed, raw_text) if spec.to_row else {}
    return {
        "question_id": request.question_id,
        "lang": request.lang,
        "schema_lang": request.schema_lang,
        "model": request.model,
        "backend": backend_id,
        "run_id": run_id,
        "sample_idx": request.sample_idx,
        "seed": request.seed,
        "temperature": request.sampling.temperature,
        "prompt_sha256": request.prompt_sha256,
        "prompt": request.prompt,
        "option_order": json.dumps(list(request.option_order), ensure_ascii=False),
        "raw_json": result.raw_json,
        **parsed_fields,
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
        **_cost_columns(result.usage, pricing_entry),
        "created_at": result.created_at,
    }


@dataclass(frozen=True)
class _PreparedRun:
    """The shared setup both the sync and batch paths build a run from."""

    spec: ExperimentSpec
    manifest: RunManifest
    templates: dict[str, PromptTemplate]
    fruits: FruitTable


def _prepare(
    question_ref: str,
    backend_id: str,
    model: str | None,
    samples: int,
    languages: list[str] | None,
    seed: int | None,
    schema_lang: str,
    run_id: str | None,
) -> _PreparedRun:
    """Load the question and build its manifest, ready for the idempotency check."""
    config = load_question(question_ref)
    spec = get_experiment(config.experiment_id)
    spec.variant(schema_lang)

    model = model or config.model
    if not model:
        raise ValueError(
            f"No model given and none set in experiment.yaml for {config.experiment_id}"
        )

    languages = languages or config.languages
    effective_seed = seed if seed is not None else config.sampling.seed
    templates = {
        lang: load_template(config.experiment_id, config.question_id, lang)
        for lang in languages
    }
    fruits = load_fruits(config.experiment_id)

    manifest = RunManifest(
        run_id=run_id or _new_run_id(config.question_id),
        experiment_id=config.experiment_id,
        question_id=config.question_id,
        backend=backend_id,
        model=model,
        schema_lang=schema_lang,
        languages=languages,
        sampling=config.sampling,
        seed=effective_seed,
        samples=samples,
        order=config.order,
        order_ids=config.order_ids,
        template_sha256={lang: template.sha256 for lang, template in templates.items()},
        fruits_sha256=fruits.sha256,
    )
    return _PreparedRun(
        spec=spec, manifest=manifest, templates=templates, fruits=fruits
    )


def _skipped_outcome(manifest: RunManifest) -> RunOutcome:
    """Build the outcome returned when an identical run already exists."""
    return RunOutcome(
        run_id=manifest.run_id,
        manifest=manifest,
        parquet_path=results_path(
            manifest.question_id, manifest.model, manifest.run_id
        ),
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
    question: str,
    backend: GenerationBackend,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    schema_lang: str = "en",
    run_id: str | None = None,
    max_retries: int = 3,
    retry_backoff: float = 1.0,
    requests_per_minute: float | None = None,
    pricing_table: PricingTable | None = None,
) -> RunOutcome:
    """Generate responses for one question and persist them to Parquet.

    Loads the question config and experiment spec, renders one prompt per
    language and sample, and writes the validated results plus a run manifest.
    If a manifest with the same content hash already exists, the run is skipped
    and nothing is regenerated. The pricing table defaults to the committed
    pricing file and is pinned into the manifest so cost stays reproducible.
    """
    prepared = _prepare(
        question,
        backend.backend_id,
        model,
        samples,
        languages,
        seed,
        schema_lang,
        run_id,
    )
    manifest = prepared.manifest

    existing = find_manifest_by_content_hash(manifest.content_hash())
    if existing is not None:
        return _skipped_outcome(existing)

    manifest.model_snapshot = backend.resolve_model_snapshot(manifest.model)
    pricing_entry = _pin_pricing(manifest, pricing_table)
    variant = prepared.spec.variant(manifest.schema_lang)
    results = _generate_all(
        backend,
        _build_requests(manifest, prepared.spec, prepared.templates, prepared.fruits),
        max_retries,
        retry_backoff,
        requests_per_minute,
    )
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

    parquet_path = write_results(
        rows, manifest.question_id, manifest.model, manifest.run_id
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
    question: str,
    backend_id: str,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    schema_lang: str = "en",
) -> RunPlan:
    """Build a run's manifest and check for a duplicate without generating anything.

    Takes the backend id rather than a backend so a dry run needs no client and
    no API key. Reuses the same preparation the real run does, so the previewed
    languages and model match what a run would use.
    """
    manifest = _prepare(
        question, backend_id, model, samples, languages, seed, schema_lang, None
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
    question: str,
    backend: BatchBackend,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    schema_lang: str = "en",
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
        question,
        backend.backend_id,
        model,
        samples,
        languages,
        seed,
        schema_lang,
        run_id,
    )
    manifest = prepared.manifest

    existing = find_manifest_by_content_hash(manifest.content_hash())
    if existing is not None:
        return _skipped_outcome(existing)

    manifest.model_snapshot = backend.resolve_model_snapshot(manifest.model)
    _pin_pricing(manifest, pricing_table)
    manifest.batch_id = backend.submit(
        _build_requests(manifest, prepared.spec, prepared.templates, prepared.fruits)
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
        parquet_path=results_path(
            manifest.question_id, manifest.model, manifest.run_id
        ),
        manifest_path=written_manifest_path,
        rows_written=0,
        skipped=False,
        batch_id=manifest.batch_id,
    )


def fetch_batch(run_id: str, backend: BatchBackend) -> RunOutcome:
    """Fetch a submitted batch's results and persist them to Parquet.

    Rebuilds the exact requests from the stored manifest, verifying the templates
    and fruit table still hash to the values recorded at submit time before
    writing results.
    """
    manifest = read_manifest(run_id)
    if manifest.batch_id is None:
        raise ValueError(f"Run {run_id} has no batch to fetch.")

    spec = get_experiment(manifest.experiment_id)
    variant = spec.variant(manifest.schema_lang)
    requests = _requests_from_manifest(manifest, spec)
    results = backend.fetch(manifest.batch_id, requests)
    rows = [
        _result_to_row(
            result, manifest.backend, manifest.run_id, spec, variant, manifest.pricing
        )
        for result in results
    ]

    parquet_path = write_results(
        rows, manifest.question_id, manifest.model, manifest.run_id
    )
    return RunOutcome(
        run_id=manifest.run_id,
        manifest=manifest,
        parquet_path=parquet_path,
        manifest_path=manifest_path(manifest.run_id),
        rows_written=len(rows),
        skipped=False,
        batch_id=manifest.batch_id,
    )


def _requests_from_manifest(
    manifest: RunManifest, spec: ExperimentSpec
) -> list[GenRequest]:
    """Rebuild a run's requests from its manifest, checking inputs still match."""
    templates = {
        lang: load_template(manifest.experiment_id, manifest.question_id, lang)
        for lang in manifest.languages
    }
    for lang, template in templates.items():
        if template.sha256 != manifest.template_sha256[lang]:
            raise ValueError(
                f"Template {manifest.question_id}/{lang}.md changed since submit; "
                f"its hash no longer matches the manifest."
            )
    fruits = load_fruits(manifest.experiment_id)
    if fruits.sha256 != manifest.fruits_sha256:
        raise ValueError(
            f"fruits.yaml for {manifest.experiment_id} changed since submit; "
            f"its hash no longer matches the manifest."
        )
    return _build_requests(manifest, spec, templates, fruits)


def _build_requests(
    manifest: RunManifest,
    spec: ExperimentSpec,
    templates: dict[str, PromptTemplate],
    fruits: FruitTable,
) -> list[GenRequest]:
    """Render one request per language and sample from the run's templates."""
    variant = spec.variant(manifest.schema_lang)
    requests: list[GenRequest] = []
    for lang in manifest.languages:
        template = templates[lang]
        for sample_idx in range(manifest.samples):
            prompt, shown = render_prompt(
                template,
                fruits,
                manifest.order,
                manifest.order_ids,
                sample_idx,
                manifest.seed,
            )
            requests.append(
                GenRequest(
                    question_id=manifest.question_id,
                    lang=lang,
                    model=manifest.model,
                    prompt=prompt,
                    prompt_sha256=prompt_sha256(prompt),
                    sample_idx=sample_idx,
                    seed=manifest.seed,
                    sampling=manifest.sampling,
                    response_schema=variant.schema,
                    option_order=tuple(shown),
                    schema_lang=manifest.schema_lang,
                )
            )
    return requests


def _generate_all(
    backend: GenerationBackend,
    requests: list[GenRequest],
    max_retries: int,
    retry_backoff: float,
    requests_per_minute: float | None,
) -> list[GenResult]:
    """Generate every request in order, honoring retries and a rate cap."""
    interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
    results: list[GenResult] = []
    for index, request in enumerate(requests):
        if interval and index > 0:
            time.sleep(interval)
        results.append(
            _generate_with_retry(backend, request, max_retries, retry_backoff)
        )
    return results
