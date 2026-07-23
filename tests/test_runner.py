"""Tests for the runner: persistence, idempotency and refusal handling."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmango import runner as runner_module
from llmango.backends.base import GenerationBackend, GenRequest, GenResult
from llmango.manifest import RunManifest
from llmango.runner import fetch_batch, run, submit_batch
from llmango.storage import read_results


class RefusingBackend(GenerationBackend):
    """Backend that refuses every request with no parsed response."""

    backend_id = "refuse"

    def resolve_model_snapshot(self, model: str) -> str:
        return f"{model}-refuse"

    def generate(self, request: GenRequest) -> GenResult:
        return GenResult(
            request=request,
            raw_json=None,
            parsed=None,
            model_snapshot=self.resolve_model_snapshot(request.model),
            finish_reason="stop",
            refusal="I can't help with that.",
            error=None,
            created_at=datetime.now(UTC),
        )


class FakeBatchBackend:
    """Fake batch backend that records submissions and answers via an inner backend."""

    backend_id = "openai-batch"

    def __init__(self, inner: GenerationBackend) -> None:
        self._inner = inner
        self.submitted: list[list[GenRequest]] = []

    def resolve_model_snapshot(self, model: str) -> str:
        return self._inner.resolve_model_snapshot(model)

    def submit(self, requests: list[GenRequest]) -> str:
        self.submitted.append(requests)
        return "batch-xyz"

    def fetch(self, batch_id: str, requests: list[GenRequest]) -> list[GenResult]:
        return [self._inner.generate(request) for request in requests]


@pytest.fixture(autouse=True)
def _isolate_dirs(data_dirs: Path) -> None:
    """Redirect output directories into tmp_path for every runner test."""


def test_run_writes_rows_and_manifest(fake_backend: GenerationBackend) -> None:
    outcome = run("favorite_fruit", fake_backend, samples=2, languages=["en", "pl"])

    assert not outcome.skipped
    assert outcome.rows_written == 4
    assert outcome.parquet_path.exists()
    assert outcome.manifest_path.exists()

    frame = read_results("*.parquet")
    assert frame.height == 4
    assert set(frame["lang"].to_list()) == {"en", "pl"}
    assert outcome.manifest.model_snapshot == "gpt-5.6-luna-fake"


def test_rerun_with_same_config_adds_no_rows(fake_backend: GenerationBackend) -> None:
    first = run("favorite_fruit", fake_backend, samples=2, languages=["en"])
    second = run("favorite_fruit", fake_backend, samples=2, languages=["en"])

    assert not first.skipped
    assert second.skipped
    assert second.rows_written == 0
    assert second.run_id == first.run_id
    assert read_results("*.parquet").height == 2


def test_refusals_persist_with_empty_fruit_raw() -> None:
    outcome = run("favorite_fruit", RefusingBackend(), samples=1, languages=["en"])

    frame = read_results("*.parquet")
    assert outcome.rows_written == 1
    assert frame["fruit_raw"].to_list() == [""]
    assert frame["raw_json"].to_list() == [None]


def test_submit_batch_records_batch_id_without_writing_rows(
    fake_backend: GenerationBackend,
) -> None:
    backend = FakeBatchBackend(fake_backend)
    outcome = submit_batch("favorite_fruit", backend, samples=2, languages=["en", "pl"])

    assert not outcome.skipped
    assert outcome.rows_written == 0
    assert outcome.batch_id == "batch-xyz"
    assert outcome.manifest_path.exists()
    assert not outcome.parquet_path.exists()
    assert outcome.manifest.batch_id == "batch-xyz"


def test_submit_batch_is_idempotent(fake_backend: GenerationBackend) -> None:
    backend = FakeBatchBackend(fake_backend)
    first = submit_batch("favorite_fruit", backend, samples=1, languages=["en"])
    second = submit_batch("favorite_fruit", backend, samples=1, languages=["en"])

    assert not first.skipped
    assert second.skipped
    assert second.run_id == first.run_id
    assert len(backend.submitted) == 1


def test_fetch_batch_writes_the_submitted_results(
    fake_backend: GenerationBackend,
) -> None:
    backend = FakeBatchBackend(fake_backend)
    submitted = submit_batch(
        "favorite_fruit", backend, samples=2, languages=["en", "pl"]
    )

    fetched = fetch_batch(submitted.run_id, backend)

    assert fetched.rows_written == 4
    assert fetched.run_id == submitted.run_id
    frame = read_results("*.parquet")
    assert frame.height == 4
    assert frame["fruit_raw"].to_list() == ["apple"] * 4


def test_submit_batch_surfaces_batch_id_when_manifest_write_fails(
    fake_backend: GenerationBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBatchBackend(fake_backend)

    def _fail(manifest: RunManifest) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(runner_module, "write_manifest", _fail)

    with pytest.raises(RuntimeError, match="batch-xyz"):
        submit_batch("favorite_fruit", backend, samples=1, languages=["en"])
