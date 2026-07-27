"""Tests for the runner: planning, persistence, idempotency and refusal handling."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import FakeBackend
from llmango import runner as runner_module
from llmango.backends.base import Backend, GenRequest, GenResult
from llmango.manifest import RunManifest, read_manifest
from llmango.pricing import PricingTable
from llmango.runner import RunOptions, RunPlan, fetch_batch, plan, run
from llmango.storage import read_results


class RefusingBackend:
    """Sync backend that refuses every request with no parsed response."""

    backend_id = "refuse"

    def resolve_model_snapshot(self, model: str) -> str:
        return f"{model}-refuse"

    def generate_many(self, requests: list[GenRequest]) -> list[GenResult]:
        return [
            GenResult(
                request=request,
                raw_json=None,
                parsed=None,
                model_snapshot=self.resolve_model_snapshot(request.model),
                finish_reason="stop",
                refusal="I can't help with that.",
                error=None,
                created_at=datetime.now(UTC),
            )
            for request in requests
        ]


class FreeTextBackend:
    """Sync backend answering with plain text and no parsed schema (free-text path)."""

    backend_id = "freetext"

    def resolve_model_snapshot(self, model: str) -> str:
        return f"{model}-free"

    def generate_many(self, requests: list[GenRequest]) -> list[GenResult]:
        assert all(request.response_schema is None for request in requests)
        return [
            GenResult(
                request=request,
                raw_json="jabłko",
                parsed=None,
                model_snapshot=self.resolve_model_snapshot(request.model),
                finish_reason="stop",
                refusal=None,
                error=None,
                created_at=datetime.now(UTC),
            )
            for request in requests
        ]


@pytest.fixture(autouse=True)
def _isolate_dirs(data_dirs: Path) -> None:
    """Redirect output directories into tmp_path for every runner test."""


def _plan(
    backend: Backend,
    pricing_table: PricingTable,
    question: str = "001a",
    samples: int = 1,
    languages: list[str] | None = None,
    seed: int | None = None,
    schema_variant: str | None = None,
    batch: bool = False,
) -> RunPlan:
    """Plan a run for the backend under test, priced from an injected table."""
    return plan(
        question,
        RunOptions(
            backend_id=backend.backend_id,
            samples=samples,
            languages=languages,
            seed=seed,
            schema_variant=schema_variant,
            batch=batch,
        ),
        pricing_table=pricing_table,
    )


def test_plan_builds_every_request_and_writes_nothing(
    fake_backend: FakeBackend, pricing_table: PricingTable, data_dirs: Path
) -> None:
    planned = _plan(fake_backend, pricing_table, samples=2, languages=["en", "pl"])

    assert len(planned.requests) == 4
    assert [request.lang for request in planned.requests] == ["en", "en", "pl", "pl"]
    assert all(request.prompt for request in planned.requests)
    assert planned.pricing is not None
    assert planned.duplicate is None
    assert not (data_dirs / "runs").exists()
    assert not (data_dirs / "raw").exists()


def test_plan_rejects_a_language_the_question_has_no_template_for(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    with pytest.raises(ValueError, match="no prompt template for xx"):
        _plan(fake_backend, pricing_table, languages=["xx"])


def test_run_writes_rows_and_manifest(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        _plan(fake_backend, pricing_table, samples=2, languages=["en", "pl"]),
        fake_backend,
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
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    run(_plan(fake_backend, pricing_table, languages=["en"]), fake_backend)

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


def test_run_refuses_a_model_the_pricing_file_does_not_cover(
    fake_backend: FakeBackend,
) -> None:
    unpriced = PricingTable(currency="USD", unit="per_1m_tokens", models={})
    planned = _plan(fake_backend, unpriced, languages=["en"])

    assert planned.pricing is None
    with pytest.raises(ValueError, match="No pricing for model"):
        run(planned, fake_backend)


def test_run_refuses_a_backend_the_plan_was_not_built_for(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    planned = _plan(fake_backend, pricing_table, languages=["en"])

    with pytest.raises(ValueError, match="built for backend 'fake'"):
        run(planned, RefusingBackend())


def test_rerun_with_same_config_adds_no_rows(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    first = run(
        _plan(fake_backend, pricing_table, samples=2, languages=["en"]), fake_backend
    )
    second = run(
        _plan(fake_backend, pricing_table, samples=2, languages=["en"]), fake_backend
    )

    assert not first.skipped
    assert second.skipped
    assert second.rows_written == 0
    assert second.run_id == first.run_id
    assert read_results("*.parquet").height == 2


def test_refusals_persist_with_an_empty_answer(pricing_table: PricingTable) -> None:
    backend = RefusingBackend()
    outcome = run(_plan(backend, pricing_table, languages=["en"]), backend)

    frame = read_results("*.parquet")
    assert outcome.rows_written == 1
    assert frame["answer"].to_list() == [""]
    assert frame["raw_json"].to_list() == [None]
    assert frame["refusal"].to_list() == ["I can't help with that."]
    assert frame["total_cost_usd"].to_list() == [None]
    assert frame["prompt_tokens"].to_list() == [None]


def test_run_tags_the_schema_variant_and_prompt_inputs(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        _plan(fake_backend, pricing_table, languages=["en"], seed=1), fake_backend
    )

    frame = read_results("*.parquet")
    assert frame["schema_variant"].to_list() == ["en"]
    assert frame["schema_name"].to_list() == ["FruitChoice"]
    assert outcome.manifest.schema_sha256
    recorded = json.loads(frame["prompt_inputs"].to_list()[0])
    assert recorded["fruit_list"] == outcome.manifest.inputs["fruit_list"]["order_ids"]
    assert outcome.manifest.input_sha256["fruit_list"]


def test_free_text_variant_reads_plain_text(pricing_table: PricingTable) -> None:
    backend = FreeTextBackend()
    outcome = run(
        _plan(
            backend,
            pricing_table,
            question="001d",
            languages=["pl"],
            schema_variant="none",
        ),
        backend,
    )

    frame = read_results("*.parquet")
    assert outcome.manifest.schema_variant == "none"
    assert outcome.manifest.schema_name is None
    assert outcome.manifest.schema_sha256 is None
    assert frame["schema_variant"].to_list() == ["none"]
    assert frame["schema_name"].to_list() == [None]
    assert frame["answer"].to_list() == ["jabłko"]
    assert frame["raw_json"].to_list() == ["jabłko"]


def test_batch_run_records_batch_id_without_writing_rows(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        _plan(
            fake_backend, pricing_table, samples=2, languages=["en", "pl"], batch=True
        ),
        fake_backend,
    )

    assert not outcome.skipped
    assert outcome.rows_written == 0
    assert outcome.batch_id == "batch-xyz"
    assert outcome.manifest_path.exists()
    assert not outcome.parquet_path.exists()
    assert outcome.manifest.batch_id == "batch-xyz"
    assert outcome.manifest.pricing is not None


def test_batch_run_is_idempotent(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    first = run(
        _plan(fake_backend, pricing_table, languages=["en"], batch=True), fake_backend
    )
    second = run(
        _plan(fake_backend, pricing_table, languages=["en"], batch=True), fake_backend
    )

    assert not first.skipped
    assert second.skipped
    assert second.run_id == first.run_id
    assert len(fake_backend.submitted) == 1


def test_fetch_batch_writes_the_submitted_results(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    submitted = run(
        _plan(
            fake_backend, pricing_table, samples=2, languages=["en", "pl"], batch=True
        ),
        fake_backend,
    )

    fetched = fetch_batch(submitted.run_id, fake_backend)

    assert fetched.rows_written == 4
    assert fetched.run_id == submitted.run_id
    frame = read_results("*.parquet")
    assert frame.height == 4
    assert frame["answer"].to_list() == ["apple"] * 4
    assert frame["total_cost_usd"].to_list() == pytest.approx([0.81e-6] * 4)


def test_fetch_batch_records_usage_in_the_manifest(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    submitted = run(
        _plan(
            fake_backend, pricing_table, samples=2, languages=["en", "pl"], batch=True
        ),
        fake_backend,
    )
    assert submitted.manifest.usage is None

    fetch_batch(submitted.run_id, fake_backend)

    manifest = read_manifest(submitted.run_id)
    assert manifest.usage is not None
    assert manifest.usage.total.rows == 4
    assert manifest.usage.total.total_cost_usd == pytest.approx(3.24e-6)
    assert manifest.usage.by_language["pl"].rows == 2


def test_fetch_batch_refuses_an_edited_template(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = run(
        _plan(fake_backend, pricing_table, languages=["en"], batch=True), fake_backend
    )
    manifest = read_manifest(submitted.run_id)
    manifest.template_sha256["en"] = "not-the-hash-that-was-submitted"
    monkeypatch.setattr(runner_module, "read_manifest", lambda run_id: manifest)

    with pytest.raises(ValueError, match="changed since submit"):
        fetch_batch(submitted.run_id, fake_backend)


def test_run_records_usage_in_total_and_per_language(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        _plan(fake_backend, pricing_table, samples=2, languages=["en", "pl"]),
        fake_backend,
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
    backend = RefusingBackend()
    outcome = run(_plan(backend, pricing_table, samples=2, languages=["en"]), backend)

    usage = outcome.manifest.usage
    assert usage is not None
    assert usage.total.rows == 2
    assert usage.total.provider_refusals == 2
    assert usage.total.total_tokens == 0
    assert usage.total.total_cost_usd == 0.0


def test_batch_run_surfaces_batch_id_when_manifest_write_fails(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(manifest: RunManifest) -> Path:
        raise OSError("disk full")

    planned = _plan(fake_backend, pricing_table, languages=["en"], batch=True)
    monkeypatch.setattr(runner_module, "write_manifest", _fail)

    with pytest.raises(RuntimeError, match="batch-xyz"):
        run(planned, fake_backend)
