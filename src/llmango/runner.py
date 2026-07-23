"""Run orchestration for the sync and batch generation paths.

A run turns one question into validated responses across languages and samples,
writes them to Parquet, and records a manifest. Reruns with the same
configuration are skipped by matching the manifest content hash, so results are
never duplicated. The batch path splits this into submit and fetch.
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from llmango.backends.base import (
    BatchBackend,
    GenerationBackend,
    GenRequest,
    GenResult,
)
from llmango.manifest import (
    RunManifest,
    find_manifest_by_content_hash,
    manifest_path,
    read_manifest,
    write_manifest,
)
from llmango.questions import PromptFile, SamplingParams, load_prompt, load_question
from llmango.registry import ExperimentSpec, get_experiment
from llmango.storage import results_path, write_results


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
    """A preview of a run: its manifest and any existing duplicate."""

    manifest: RunManifest
    duplicate: RunManifest | None


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


def _result_to_row(
    result: GenResult,
    backend_id: str,
    run_id: str,
    spec: ExperimentSpec,
) -> dict[str, object]:
    """Combine the common columns with the experiment's parsed fields."""
    request = result.request
    parsed_fields = spec.to_row(result.parsed) if spec.to_row else {}
    return {
        "question_id": request.question_id,
        "lang": request.lang,
        "model": request.model,
        "backend": backend_id,
        "run_id": run_id,
        "sample_idx": request.sample_idx,
        "seed": request.seed,
        "temperature": request.sampling.temperature,
        "prompt_sha256": request.prompt_sha256,
        "raw_json": result.raw_json,
        **parsed_fields,
        "created_at": result.created_at,
    }


@dataclass(frozen=True)
class _PreparedRun:
    """The shared setup both the sync and batch paths build a run from."""

    spec: ExperimentSpec
    manifest: RunManifest
    prompts: dict[str, PromptFile]


def _prepare(
    question_id: str,
    backend_id: str,
    model: str | None,
    samples: int,
    languages: list[str] | None,
    seed: int | None,
    run_id: str | None,
) -> _PreparedRun:
    """Load the question and build its manifest, ready for the idempotency check."""
    config = load_question(question_id)
    spec = get_experiment(question_id)

    model = model or config.model
    if not model:
        raise ValueError(f"No model given and none set in meta.yaml for {question_id}")

    languages = languages or config.languages
    effective_seed = seed if seed is not None else config.sampling.seed
    prompts = {lang: load_prompt(question_id, lang) for lang in languages}

    manifest = RunManifest(
        run_id=run_id or _new_run_id(question_id),
        question_id=question_id,
        backend=backend_id,
        model=model,
        languages=languages,
        sampling=config.sampling,
        seed=effective_seed,
        samples=samples,
        prompt_sha256={lang: prompt.sha256 for lang, prompt in prompts.items()},
    )
    return _PreparedRun(spec=spec, manifest=manifest, prompts=prompts)


def _requests_for(prepared: _PreparedRun) -> list[GenRequest]:
    """Build the requests for a prepared run, once past the idempotency check."""
    manifest = prepared.manifest
    return _build_requests(
        manifest.question_id,
        manifest.model,
        manifest.samples,
        prepared.prompts,
        manifest.seed,
        manifest.sampling,
        prepared.spec,
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


def run(
    question_id: str,
    backend: GenerationBackend,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    run_id: str | None = None,
    max_retries: int = 3,
    retry_backoff: float = 1.0,
    requests_per_minute: float | None = None,
) -> RunOutcome:
    """Generate responses for one question and persist them to Parquet.

    Loads the question config and experiment spec, builds one request per
    language and sample, and writes the validated results plus a run manifest.
    If a manifest with the same content hash already exists, the run is skipped
    and nothing is regenerated.
    """
    prepared = _prepare(
        question_id, backend.backend_id, model, samples, languages, seed, run_id
    )
    manifest = prepared.manifest

    existing = find_manifest_by_content_hash(manifest.content_hash())
    if existing is not None:
        return _skipped_outcome(existing)

    manifest.model_snapshot = backend.resolve_model_snapshot(manifest.model)
    results = _generate_all(
        backend,
        _requests_for(prepared),
        max_retries,
        retry_backoff,
        requests_per_minute,
    )
    rows = [
        _result_to_row(result, manifest.backend, manifest.run_id, prepared.spec)
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
    question_id: str,
    backend_id: str,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
) -> RunPlan:
    """Build a run's manifest and check for a duplicate without generating anything.

    Takes the backend id rather than a backend so a dry run needs no client and
    no API key. Reuses the same preparation the real run does, so the previewed
    languages and model match what a run would use.
    """
    manifest = _prepare(
        question_id, backend_id, model, samples, languages, seed, None
    ).manifest
    return RunPlan(
        manifest=manifest,
        duplicate=find_manifest_by_content_hash(manifest.content_hash()),
    )


def submit_batch(
    question_id: str,
    backend: BatchBackend,
    *,
    model: str | None = None,
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    run_id: str | None = None,
) -> RunOutcome:
    """Submit one question as an OpenAI batch and record its manifest.

    Nothing is generated inline: the batch is queued and its id stored in the
    manifest so results can be fetched later. Skips submission if an identical
    run already exists, so a batch is never submitted twice.
    """
    prepared = _prepare(
        question_id, backend.backend_id, model, samples, languages, seed, run_id
    )
    manifest = prepared.manifest

    existing = find_manifest_by_content_hash(manifest.content_hash())
    if existing is not None:
        return _skipped_outcome(existing)

    manifest.model_snapshot = backend.resolve_model_snapshot(manifest.model)
    manifest.batch_id = backend.submit(_requests_for(prepared))
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

    Rebuilds the exact requests from the stored manifest, verifying each prompt
    still hashes to the value recorded at submit time before writing results.
    """
    manifest = read_manifest(run_id)
    if manifest.batch_id is None:
        raise ValueError(f"Run {run_id} has no batch to fetch.")

    spec = get_experiment(manifest.question_id)
    requests = _requests_from_manifest(manifest, spec)
    results = backend.fetch(manifest.batch_id, requests)
    rows = [
        _result_to_row(result, manifest.backend, manifest.run_id, spec)
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
    """Rebuild a run's requests from its manifest, checking prompt hashes match."""
    prompts = {
        lang: load_prompt(manifest.question_id, lang) for lang in manifest.languages
    }
    for lang, prompt in prompts.items():
        if prompt.sha256 != manifest.prompt_sha256[lang]:
            raise ValueError(
                f"Prompt {manifest.question_id}/{lang}.md changed since submit; "
                f"its hash no longer matches the manifest."
            )
    return _build_requests(
        manifest.question_id,
        manifest.model,
        manifest.samples,
        prompts,
        manifest.seed,
        manifest.sampling,
        spec,
    )


def _build_requests(
    question_id: str,
    model: str,
    samples: int,
    prompts: dict[str, PromptFile],
    seed: int | None,
    sampling: SamplingParams,
    spec: ExperimentSpec,
) -> list[GenRequest]:
    """Build one request per language and sample index."""
    requests: list[GenRequest] = []
    for lang, prompt in prompts.items():
        for sample_idx in range(samples):
            requests.append(
                GenRequest(
                    question_id=question_id,
                    lang=lang,
                    model=model,
                    prompt=prompt.text,
                    prompt_sha256=prompt.sha256,
                    sample_idx=sample_idx,
                    seed=seed,
                    sampling=sampling,
                    response_schema=spec.response_schema,
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
