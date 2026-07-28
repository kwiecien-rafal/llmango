"""Tests for the runner: planning, persistence, batching and refusal handling."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from conftest import FakeBackend
from llmango import runner as runner_module
from llmango.backends.base import GenRequest, GenResult, Usage
from llmango.experiments.fruit import FruitChoice, WyborOwocu
from llmango.manifest import Manifest, read_manifest
from llmango.pricing import PricingTable
from llmango.runner import RunPlan, fetch_batch, plan, run
from llmango.spec import answer_field
from llmango.storage import read_results


class RefusingBackend:
    """Sync backend that refuses every request with no parsed response."""

    def generate_many(self, requests: list[GenRequest]) -> list[GenResult]:
        return [
            GenResult(
                request=request,
                raw_json=None,
                parsed=None,
                model_snapshot=f"{request.model}-refuse",
                finish_reason="stop",
                refusal="I can't help with that.",
                error=None,
                created_at=datetime.now(UTC),
            )
            for request in requests
        ]


class PolishBackend:
    """Sync backend answering each arm the way its own schema asks, free text last."""

    def generate_many(self, requests: list[GenRequest]) -> list[GenResult]:
        return [self._generate(request) for request in requests]

    def _generate(self, request: GenRequest) -> GenResult:
        schema = request.response_schema
        parsed = schema(**{answer_field(schema): "jabłko"}) if schema else None
        return GenResult(
            request=request,
            raw_json=parsed.model_dump_json() if parsed is not None else "jabłko",
            parsed=parsed,
            model_snapshot=f"{request.model}-polish",
            finish_reason="stop",
            refusal=None,
            error=None,
            created_at=datetime.now(UTC),
            usage=Usage(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
                cached_tokens=0,
                reasoning_tokens=0,
            ),
        )


@pytest.fixture(autouse=True)
def _isolate_dirs(data_dirs: Path) -> None:
    """Redirect output directories into tmp_path for every runner test."""


def _plan(
    pricing_table: PricingTable,
    question: str = "001a",
    samples_per_arm: int = 1,
    languages: list[str] | None = None,
) -> RunPlan:
    """Plan one run of a question, priced from an injected table."""
    return plan(
        question,
        samples_per_arm=samples_per_arm,
        languages=languages,
        pricing_table=pricing_table,
    )


def _record_resolved_providers(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> list[str]:
    """Capture the provider names the runner resolves for itself."""
    asked: list[str] = []

    def _backend_for(provider: str) -> FakeBackend:
        asked.append(provider)
        return backend

    monkeypatch.setattr(runner_module, "backend_for", _backend_for)
    return asked


def test_plan_builds_every_request_and_writes_nothing(
    fake_backend: FakeBackend, pricing_table: PricingTable, data_dirs: Path
) -> None:
    planned = _plan(pricing_table, samples_per_arm=2, languages=["en", "pl"])

    assert len(planned.requests) == 4
    assert [request.lang for request in planned.requests] == ["en", "en", "pl", "pl"]
    assert all(request.prompt for request in planned.requests)
    assert planned.manifest.pricing is not None
    assert not (data_dirs / "runs").exists()
    assert not (data_dirs / "raw").exists()


def test_a_plan_reads_its_provider_model_and_temperature_from_the_question(
    pricing_table: PricingTable,
) -> None:
    """A run's identity is the question's config, not anything the caller passed."""
    planned = _plan(pricing_table)

    assert planned.manifest.provider == "openai"
    assert planned.manifest.model == "gpt-5.6-luna"
    assert planned.manifest.temperature == 1.0
    assert all(request.temperature == 1.0 for request in planned.requests)


def test_plan_rejects_a_language_the_question_has_no_template_for(
    pricing_table: PricingTable,
) -> None:
    with pytest.raises(ValueError, match="no prompt template for xx"):
        _plan(pricing_table, languages=["xx"])


def test_run_writes_rows_and_manifest(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        _plan(pricing_table, samples_per_arm=2, languages=["en", "pl"]), fake_backend
    )

    assert outcome.rows_written == 4
    assert outcome.parquet_path.exists()
    assert outcome.manifest_path.exists()
    assert outcome.parquet_path.stem.startswith(outcome.run_id)
    assert outcome.manifest_path.stem == outcome.run_id

    frame = read_results("*.parquet")
    assert frame.height == 4
    assert set(frame["lang"].to_list()) == {"en", "pl"}
    assert frame["provider"].to_list() == ["openai"] * 4
    assert outcome.manifest.pricing is not None


def test_run_resolves_the_backend_its_question_names(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller passes a backend only to stand in; a run knows its own provider."""
    asked = _record_resolved_providers(monkeypatch, fake_backend)

    outcome = run(_plan(pricing_table, languages=["en"]))

    assert asked == ["openai"]
    assert outcome.rows_written == 1


def test_run_records_provenance_tokens_and_cost(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    run(_plan(pricing_table, languages=["en"]), fake_backend)

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
    planned = _plan(unpriced, languages=["en"])

    assert planned.manifest.pricing is None
    with pytest.raises(ValueError, match="No pricing for model"):
        run(planned, fake_backend)


def test_a_rerun_is_more_samples_rather_than_a_replacement(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    """Two runs of one question are more samples of it, so both files are kept."""
    first = run(_plan(pricing_table, samples_per_arm=2, languages=["en"]), fake_backend)
    second = run(
        _plan(pricing_table, samples_per_arm=2, languages=["en"]), fake_backend
    )

    assert first.run_id != second.run_id
    assert first.rows_written == second.rows_written == 2
    assert read_results("001a__*.parquet").height == 4


def test_a_run_never_overwrites_another_ones_files(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run ids are stamped to the millisecond; a collision refuses rather than eats."""
    monkeypatch.setattr(runner_module, "build_run_id", lambda manifest: "001a__fixed")
    run(_plan(pricing_table, languages=["en"]), fake_backend)

    with pytest.raises(ValueError, match="already exists"):
        run(_plan(pricing_table, languages=["en"]), fake_backend)


def test_a_run_id_names_the_question_and_when_it_started(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    outcome = run(_plan(pricing_table, languages=["en"]), fake_backend)

    assert re.fullmatch(r"001a__\d{8}T\d{9}Z", outcome.run_id)


def test_refusals_persist_with_an_empty_answer(pricing_table: PricingTable) -> None:
    backend = RefusingBackend()
    outcome = run(_plan(pricing_table, languages=["en"]), backend)

    frame = read_results("*.parquet")
    assert outcome.rows_written == 1
    assert frame["answer"].to_list() == [""]
    assert frame["raw_json"].to_list() == [None]
    assert frame["refusal"].to_list() == ["I can't help with that."]
    assert frame["total_cost_usd"].to_list() == [None]
    assert frame["prompt_tokens"].to_list() == [None]


def test_every_row_carries_the_schema_it_was_asked_under(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    """The schema itself is stored, so the raw data explains itself alone."""
    outcome = run(_plan(pricing_table, languages=["en"]), fake_backend)

    frame = read_results("*.parquet")
    assert json.loads(frame["response_schema"].to_list()[0]) == (
        FruitChoice.model_json_schema()
    )
    arm = outcome.manifest.arms[0]
    assert arm.response_schema == FruitChoice.model_json_schema()
    recorded = json.loads(frame["prompt_inputs"].to_list()[0])
    assert recorded["fruit_list"] == outcome.manifest.inputs["fruit_list"]["order_ids"]
    assert outcome.manifest.input_sha256["fruit_list"]


def test_one_run_covers_every_arm_a_question_declares(
    pricing_table: PricingTable,
) -> None:
    """001d asks one language three ways, so one run writes all three arms."""
    planned = _plan(pricing_table, question="001d", samples_per_arm=2)

    assert [arm.schema_name for arm in planned.manifest.arms] == [
        "FruitChoice",
        "WyborOwocu",
        None,
    ]
    assert planned.manifest.arms[1].response_schema == WyborOwocu.model_json_schema()
    assert planned.manifest.samples_total == 6

    outcome = run(planned, PolishBackend())

    frame = read_results("*.parquet")
    assert outcome.rows_written == 6
    assert frame["lang"].to_list() == ["pl"] * 6
    assert frame["answer"].to_list() == ["jabłko"] * 6
    assert frame["response_schema"].str.json_path_match("$.title").to_list() == (
        ["FruitChoice"] * 2 + ["WyborOwocu"] * 2 + [None] * 2
    )

    free_text = frame.filter(pl.col("response_schema").is_null())
    assert free_text.height == 2
    assert free_text["raw_json"].to_list() == ["jabłko"] * 2


def test_batch_run_records_batch_id_without_writing_rows(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        _plan(pricing_table, samples_per_arm=2, languages=["en", "pl"]),
        fake_backend,
        batch=True,
    )

    assert outcome.rows_written == 0
    assert outcome.batch_id == "batch-xyz"
    assert outcome.manifest_path.exists()
    assert not outcome.parquet_path.exists()
    assert outcome.manifest.batch_id == "batch-xyz"
    assert outcome.manifest.pricing is not None


def test_fetch_batch_writes_the_submitted_results(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    submitted = run(
        _plan(pricing_table, samples_per_arm=2, languages=["en", "pl"]),
        fake_backend,
        batch=True,
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
        _plan(pricing_table, samples_per_arm=2, languages=["en", "pl"]),
        fake_backend,
        batch=True,
    )
    assert submitted.manifest.usage is None

    fetch_batch(submitted.run_id, fake_backend)

    manifest = read_manifest(submitted.run_id)
    assert manifest.usage is not None
    assert manifest.usage.total_tokens == 60
    assert manifest.usage.total_cost_usd == pytest.approx(3.24e-6)


def test_fetch_batch_resolves_the_backend_its_manifest_records(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest is the only record of which provider holds the batch."""
    submitted = run(_plan(pricing_table, languages=["en"]), fake_backend, batch=True)
    asked = _record_resolved_providers(monkeypatch, fake_backend)

    fetched = fetch_batch(submitted.run_id)

    assert asked == ["openai"]
    assert fetched.rows_written == 1


def test_fetch_batch_refuses_an_edited_template(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = run(_plan(pricing_table, languages=["en"]), fake_backend, batch=True)
    manifest = read_manifest(submitted.run_id)
    manifest.arms[0].template_sha256 = "not-the-hash-that-was-submitted"
    monkeypatch.setattr(runner_module, "read_manifest", lambda run_id: manifest)

    with pytest.raises(ValueError, match="changed since submit"):
        fetch_batch(submitted.run_id, fake_backend)


def test_fetch_batch_refuses_an_edited_response_schema(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema edited between submit and fetch would misdescribe what was sent."""
    submitted = run(_plan(pricing_table, languages=["en"]), fake_backend, batch=True)
    manifest = read_manifest(submitted.run_id)
    manifest.arms[0].response_schema = {"title": "SomethingElse"}
    monkeypatch.setattr(runner_module, "read_manifest", lambda run_id: manifest)

    with pytest.raises(ValueError, match="changed since submit"):
        fetch_batch(submitted.run_id, fake_backend)


def test_fetch_batch_refuses_an_arm_the_question_no_longer_asks(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = run(_plan(pricing_table, languages=["en"]), fake_backend, batch=True)
    manifest = read_manifest(submitted.run_id)
    manifest.arms[0].schema_name = "WyborOwocu"
    monkeypatch.setattr(runner_module, "read_manifest", lambda run_id: manifest)

    with pytest.raises(ValueError, match="no longer asks en under WyborOwocu"):
        fetch_batch(submitted.run_id, fake_backend)


def test_run_records_usage_for_the_whole_run(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    outcome = run(
        _plan(pricing_table, samples_per_arm=2, languages=["en", "pl"]), fake_backend
    )

    usage = outcome.manifest.usage
    assert usage is not None
    assert usage.prompt_tokens == 48
    assert usage.errors == 0
    assert usage.provider_refusals == 0


def test_usage_is_recorded_for_each_arm_as_well_as_the_run(
    fake_backend: FakeBackend, pricing_table: PricingTable
) -> None:
    """Every arm carries its own share of what the run used, not the whole of it."""
    outcome = run(
        _plan(pricing_table, samples_per_arm=2, languages=["en", "pl"]), fake_backend
    )

    arms = outcome.manifest.arms
    assert [arm.lang for arm in arms] == ["en", "pl"]
    assert all(arm.usage is not None and arm.usage.prompt_tokens == 24 for arm in arms)
    assert outcome.manifest.usage is not None
    assert outcome.manifest.usage.prompt_tokens == 48


def test_arm_usage_separates_arms_that_share_a_language(
    pricing_table: PricingTable,
) -> None:
    """001d asks one language three ways, so an arm is its schema as much as it."""
    outcome = run(
        _plan(pricing_table, question="001d", samples_per_arm=2), PolishBackend()
    )

    arms = outcome.manifest.arms
    assert [arm.schema_name for arm in arms] == ["FruitChoice", "WyborOwocu", None]
    assert all(arm.usage is not None and arm.usage.prompt_tokens == 24 for arm in arms)
    assert outcome.manifest.usage is not None
    assert outcome.manifest.usage.prompt_tokens == 72


def test_usage_counts_provider_refusals_and_keeps_their_cost_null(
    pricing_table: PricingTable,
) -> None:
    backend = RefusingBackend()
    outcome = run(_plan(pricing_table, samples_per_arm=2, languages=["en"]), backend)

    usage = outcome.manifest.usage
    assert usage is not None
    assert usage.provider_refusals == 2
    assert usage.total_tokens == 0
    assert usage.total_cost_usd == 0.0


def test_batch_run_surfaces_batch_id_when_manifest_write_fails(
    fake_backend: FakeBackend,
    pricing_table: PricingTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(manifest: Manifest) -> Path:
        raise OSError("disk full")

    planned = _plan(pricing_table, languages=["en"])
    monkeypatch.setattr(runner_module, "write_manifest", _fail)

    with pytest.raises(RuntimeError, match="batch-xyz"):
        run(planned, fake_backend, batch=True)
