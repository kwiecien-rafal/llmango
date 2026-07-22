"""Run orchestration for the sync generation path.

A run turns one question into validated responses across languages and samples,
writes them to Parquet, and records a manifest. Reruns with the same
configuration are skipped by matching the manifest content hash, so results are
never duplicated.
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from llmango.backends.base import GenerationBackend, GenRequest, GenResult
from llmango.manifest import (
    RunManifest,
    find_manifest_by_content_hash,
    manifest_path,
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
    config = load_question(question_id)
    spec = get_experiment(question_id)

    model = model or config.model
    if not model:
        raise ValueError(f"No model given and none set in meta.yaml for {question_id}")

    languages = languages or config.languages
    effective_seed = seed if seed is not None else config.sampling.seed
    prompts = {lang: load_prompt(question_id, lang) for lang in languages}
    run_id = run_id or _new_run_id(question_id)

    manifest = RunManifest(
        run_id=run_id,
        question_id=question_id,
        backend=backend.backend_id,
        model=model,
        languages=languages,
        sampling=config.sampling,
        seed=effective_seed,
        samples=samples,
        prompt_sha256={lang: prompt.sha256 for lang, prompt in prompts.items()},
    )

    existing = find_manifest_by_content_hash(manifest.content_hash())
    if existing is not None:
        return RunOutcome(
            run_id=existing.run_id,
            manifest=existing,
            parquet_path=results_path(question_id, model, existing.run_id),
            manifest_path=manifest_path(existing.run_id),
            rows_written=0,
            skipped=True,
        )

    manifest.model_snapshot = backend.resolve_model_snapshot(model)

    requests = _build_requests(
        question_id, model, samples, prompts, effective_seed, config.sampling, spec
    )
    results = _generate_all(
        backend, requests, max_retries, retry_backoff, requests_per_minute
    )
    rows = [
        _result_to_row(result, backend.backend_id, run_id, spec) for result in results
    ]

    parquet_path = write_results(rows, question_id, model, run_id)
    written_manifest_path = write_manifest(manifest)

    return RunOutcome(
        run_id=run_id,
        manifest=manifest,
        parquet_path=parquet_path,
        manifest_path=written_manifest_path,
        rows_written=len(rows),
        skipped=False,
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
                    response_model=spec.response_model,
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
