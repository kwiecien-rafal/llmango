"""Tests for the runner: persistence, idempotency and refusal handling."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmango import manifest as manifest_module
from llmango import storage as storage_module
from llmango.backends.base import GenerationBackend, GenRequest, GenResult
from llmango.runner import run
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


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(manifest_module, "RUNS_DIR", tmp_path / "runs")


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
