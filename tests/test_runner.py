"""Tests for the runner: persistence, idempotency and refusal handling."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmango import runner as runner_module
from llmango.backends.base import GenerationBackend, GenRequest, GenResult
from llmango.manifest import RunManifest, read_manifest
from llmango.pricing import PricingTable
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


class FreeTextBackend(GenerationBackend):
    """Backend that answers with plain text and no parsed schema (free-text path)."""

    backend_id = "freetext"

    def resolve_model_snapshot(self, model: str) -> str:
        return f"{model}-free"

    def generate(self, request: GenRequest) -> GenResult:
        assert request.response_schema is None
        return GenResult(
            request=request,
            raw_json="jabłko",
            parsed=None,
            model_snapshot=self.resolve_model_snapshot(request.model),
            finish_reason="stop",
            refusal=None,
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


def test_run_writes_rows_and_manifest(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        "001a",
        fake_backend,
        samples=2,
        languages=["en", "pl"],
        pricing_table=pricing_table,
    )

    assert not outcome.skipped
    assert outcome.rows_written == 4
    assert outcome.parquet_path.exists()
    assert outcome.manifest_path.exists()
    assert outcome.parquet_path.stem.startswith(outcome.run_id)
    assert outcome.manifest_path.stem == outcome.run_id

    frame = read_results("*.parquet")
    assert frame.height == 4
    assert set(frame["lang"].to_list()) == {"en", "pl"}
    assert outcome.manifest.model_snapshot == "gpt-5.6-luna-fake"
    assert outcome.manifest.pricing is not None


def test_run_records_provenance_tokens_and_cost(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    run(
        "001a",
        fake_backend,
        samples=1,
        languages=["en"],
        pricing_table=pricing_table,
    )

    frame = read_results("*.parquet")
    assert frame["prompt"].to_list()[0]
    assert frame["model_snapshot"].to_list() == ["gpt-5.6-luna-fake"]
    assert frame["response_id"].to_list() == ["chatcmpl-fake"]
    assert frame["response_envelope"].to_list()[0] is not None
    assert frame["prompt_tokens"].to_list() == [12]
    assert frame["cached_tokens"].to_list() == [4]
    assert frame["reasoning_tokens"].to_list() == [1]
    assert frame["pricing_version"].to_list() == ["2026-07-24"]
    cost = frame["total_cost_usd"].to_list()[0]
    assert cost is not None
    assert cost > 0


def test_rerun_with_same_config_adds_no_rows(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    first = run(
        "001a",
        fake_backend,
        samples=2,
        languages=["en"],
        pricing_table=pricing_table,
    )
    second = run(
        "001a",
        fake_backend,
        samples=2,
        languages=["en"],
        pricing_table=pricing_table,
    )

    assert not first.skipped
    assert second.skipped
    assert second.rows_written == 0
    assert second.run_id == first.run_id
    assert read_results("*.parquet").height == 2


def test_refusals_persist_with_empty_fruit_raw(pricing_table: PricingTable) -> None:
    outcome = run(
        "001a",
        RefusingBackend(),
        samples=1,
        languages=["en"],
        pricing_table=pricing_table,
    )

    frame = read_results("*.parquet")
    assert outcome.rows_written == 1
    assert frame["fruit_raw"].to_list() == [""]
    assert frame["raw_json"].to_list() == [None]
    assert frame["refusal"].to_list() == ["I can't help with that."]
    assert frame["total_cost_usd"].to_list() == [None]
    assert frame["prompt_tokens"].to_list() == [None]


def test_run_tags_the_schema_variant_and_option_order(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        "001a",
        fake_backend,
        samples=1,
        languages=["en"],
        seed=1,
        pricing_table=pricing_table,
    )

    frame = read_results("*.parquet")
    assert frame["schema_variant"].to_list() == ["en"]
    assert frame["schema_name"].to_list() == ["FruitChoice"]
    assert outcome.manifest.schema_sha256
    order = frame["option_order"].to_list()[0]
    assert order.startswith("[") and "apple" in order


def test_free_text_variant_reads_plain_text(pricing_table: PricingTable) -> None:
    outcome = run(
        "001d",
        FreeTextBackend(),
        samples=1,
        languages=["pl"],
        schema_variant="none",
        pricing_table=pricing_table,
    )

    frame = read_results("*.parquet")
    assert outcome.manifest.schema_variant == "none"
    assert outcome.manifest.schema_name is None
    assert outcome.manifest.schema_sha256 is None
    assert frame["schema_variant"].to_list() == ["none"]
    assert frame["schema_name"].to_list() == [None]
    assert frame["fruit_raw"].to_list() == ["jabłko"]
    assert frame["raw_json"].to_list() == ["jabłko"]


def test_submit_batch_records_batch_id_without_writing_rows(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    backend = FakeBatchBackend(fake_backend)
    outcome = submit_batch(
        "001a",
        backend,
        samples=2,
        languages=["en", "pl"],
        pricing_table=pricing_table,
    )

    assert not outcome.skipped
    assert outcome.rows_written == 0
    assert outcome.batch_id == "batch-xyz"
    assert outcome.manifest_path.exists()
    assert not outcome.parquet_path.exists()
    assert outcome.manifest.batch_id == "batch-xyz"
    assert outcome.manifest.pricing is not None


def test_submit_batch_is_idempotent(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    backend = FakeBatchBackend(fake_backend)
    first = submit_batch(
        "001a",
        backend,
        samples=1,
        languages=["en"],
        pricing_table=pricing_table,
    )
    second = submit_batch(
        "001a",
        backend,
        samples=1,
        languages=["en"],
        pricing_table=pricing_table,
    )

    assert not first.skipped
    assert second.skipped
    assert second.run_id == first.run_id
    assert len(backend.submitted) == 1


def test_fetch_batch_writes_the_submitted_results(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    backend = FakeBatchBackend(fake_backend)
    submitted = submit_batch(
        "001a",
        backend,
        samples=2,
        languages=["en", "pl"],
        pricing_table=pricing_table,
    )

    fetched = fetch_batch(submitted.run_id, backend)

    assert fetched.rows_written == 4
    assert fetched.run_id == submitted.run_id
    frame = read_results("*.parquet")
    assert frame.height == 4
    assert frame["fruit_raw"].to_list() == ["apple"] * 4
    assert frame["total_cost_usd"].to_list() == pytest.approx([0.81e-6] * 4)


def test_fetch_batch_records_usage_in_the_manifest(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    backend = FakeBatchBackend(fake_backend)
    submitted = submit_batch(
        "001a",
        backend,
        samples=2,
        languages=["en", "pl"],
        pricing_table=pricing_table,
    )
    assert submitted.manifest.usage is None

    fetch_batch(submitted.run_id, backend)

    manifest = read_manifest(submitted.run_id)
    assert manifest.usage is not None
    assert manifest.usage.total.rows == 4
    assert manifest.usage.total.total_cost_usd == pytest.approx(3.24e-6)
    assert manifest.usage.by_language["pl"].rows == 2


def test_run_records_usage_in_total_and_per_language(
    fake_backend: GenerationBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        "001a",
        fake_backend,
        samples=2,
        languages=["en", "pl"],
        pricing_table=pricing_table,
    )

    usage = outcome.manifest.usage
    assert usage is not None
    assert usage.total.rows == 4
    assert usage.total.prompt_tokens == 48
    assert usage.total.errors == 0
    assert usage.total.provider_refusals == 0
    assert set(usage.by_language) == {"en", "pl"}
    assert usage.by_language["en"].rows == 2
    assert usage.by_language["en"].total_cost_usd == pytest.approx(
        usage.total.total_cost_usd / 2
    )


def test_usage_counts_provider_refusals_and_keeps_their_cost_null(
    pricing_table: PricingTable,
) -> None:
    outcome = run(
        "001a",
        RefusingBackend(),
        samples=2,
        languages=["en"],
        pricing_table=pricing_table,
    )

    usage = outcome.manifest.usage
    assert usage is not None
    assert usage.total.rows == 2
    assert usage.total.provider_refusals == 2
    assert usage.total.total_tokens == 0
    assert usage.total.total_cost_usd == 0.0


def test_submit_batch_surfaces_batch_id_when_manifest_write_fails(
    fake_backend: GenerationBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBatchBackend(fake_backend)

    def _fail(manifest: RunManifest) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(runner_module, "write_manifest", _fail)

    with pytest.raises(RuntimeError, match="batch-xyz"):
        submit_batch(
            "001a",
            backend,
            samples=1,
            languages=["en"],
            pricing_table=pricing_table,
        )
