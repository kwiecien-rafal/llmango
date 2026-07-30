"""Tests for the runner: planning, persistence and refusal handling."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from conftest import FakeBackend
from llmango import pricing as pricing_module
from llmango import runner as runner_module
from llmango.backends.base import GenRequest, GenResult, Usage
from llmango.experiments.e001_fruit.experiment import FruitChoice, WyborOwocu
from llmango.manifest import Manifest
from llmango.pricing import COST_GUARD_CALLS, PricingTable
from llmango.rows import column_dtypes
from llmango.runner import RunPlan, plan, run
from llmango.spec import FREE_TEXT, answer_field
from llmango.storage import read_results, results_path


class RefusingBackend:
    """Backend that refuses every request with no parsed response."""

    def generate(self, request: GenRequest) -> GenResult:
        return GenResult(
            request=request,
            raw_json=None,
            parsed=None,
            model_snapshot=f"{request.model}-refuse",
            finish_reason="stop",
            refusal="I can't help with that.",
            error=None,
            created_at=datetime.now(UTC),
        )


class CrashingBackend(FakeBackend):
    """Backend that answers until a given call, which dies the way a network does."""

    def __init__(self, dies_on: int, error: type[BaseException] = RuntimeError) -> None:
        super().__init__()
        self._dies_on = dies_on
        self._error = error

    def generate(self, request: GenRequest) -> GenResult:
        if self.calls + 1 == self._dies_on:
            raise self._error("connection dropped")

        return super().generate(request)


class PolishBackend:
    """Backend answering each arm the way its own schema asks, free text last."""

    def generate(self, request: GenRequest) -> GenResult:
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


def _plan(question: str = "001a", samples_per_arm: int = 1) -> RunPlan:
    """Plan one run of a question. 001a is asked in en, pl and ja: three arms."""
    return plan(question, samples_per_arm=samples_per_arm)


def _raw(pattern: str = "*.jsonl") -> pl.DataFrame:
    """Read back what the runner appended, under the dtypes it declares."""
    return read_results(pattern, column_dtypes({}))


def _only_manifest(data_dirs: Path) -> Manifest:
    """Read back the one run a test started, the record a crash still leaves in git."""
    written = list((data_dirs / "runs").glob("*.json"))
    assert len(written) == 1

    return Manifest.model_validate_json(written[0].read_text(encoding="utf-8"))


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


def test_plan_builds_every_request_and_writes_nothing(data_dirs: Path) -> None:
    planned = _plan(samples_per_arm=2)

    assert len(planned.requests) == 6
    assert [sample.arm.lang for sample in planned.samples] == [
        "en",
        "en",
        "pl",
        "pl",
        "ja",
        "ja",
    ]
    assert all(request.prompt for request in planned.requests)
    assert planned.price is not None
    assert not (data_dirs / "runs").exists()
    assert not (data_dirs / "raw").exists()


def test_a_plan_reads_its_provider_model_and_temperature_from_the_question() -> None:
    """A run's identity is the question's config, and nothing overrides it."""
    planned = _plan()

    assert planned.question.provider == "openai"
    assert planned.question.model == "gpt-5.6-luna"
    assert planned.question.temperature == 1.0
    assert all(request.temperature == 1.0 for request in planned.requests)


def test_run_writes_rows_and_manifest(
    fake_backend: FakeBackend, data_dirs: Path
) -> None:
    outcome = run(_plan(samples_per_arm=2), fake_backend)

    assert outcome.rows_written == 6
    assert outcome.finished
    assert outcome.results_path.exists()
    assert outcome.manifest_path.exists()
    assert outcome.results_path.stem == outcome.run_id
    assert outcome.manifest_path.stem == outcome.run_id

    frame = _raw()
    assert frame.height == 6
    assert set(frame["lang"].to_list()) == {"en", "pl", "ja"}
    assert frame["provider"].to_list() == ["openai"] * 6
    assert outcome.manifest.pricing is not None
    assert _only_manifest(data_dirs).samples_written == 6


def test_a_crash_keeps_every_result_that_already_landed(data_dirs: Path) -> None:
    """A dropped connection on call four must not cost the three already paid for."""
    with pytest.raises(RuntimeError, match="connection dropped"):
        run(_plan(samples_per_arm=2), CrashingBackend(dies_on=4))

    assert _raw().height == 3
    assert _only_manifest(data_dirs).samples_written == 3


def test_ctrl_c_reports_what_was_written_rather_than_a_traceback(
    data_dirs: Path,
) -> None:
    outcome = run(
        _plan(samples_per_arm=2), CrashingBackend(dies_on=4, error=KeyboardInterrupt)
    )

    assert not outcome.finished
    assert outcome.rows_written == 3
    assert _raw().height == 3
    assert _only_manifest(data_dirs).samples_written == 3


def test_a_run_reports_progress_within_each_arm(fake_backend: FakeBackend) -> None:
    """Arms run one after another, so each counts its own samples from one."""
    reported: list[tuple[str, int, int]] = []

    run(
        _plan(samples_per_arm=2),
        fake_backend,
        report_progress=lambda arm, done, total: reported.append(
            (arm.lang, done, total)
        ),
    )

    assert reported == [
        ("en", 1, 2),
        ("en", 2, 2),
        ("pl", 1, 2),
        ("pl", 2, 2),
        ("ja", 1, 2),
        ("ja", 2, 2),
    ]


def test_a_run_is_recorded_before_its_first_call_is_paid_for(data_dirs: Path) -> None:
    """A run that dies on call one still left the record that money was going out."""
    with pytest.raises(RuntimeError, match="connection dropped"):
        run(_plan(), CrashingBackend(dies_on=1))

    manifest = _only_manifest(data_dirs)
    assert manifest.samples_written == 0
    assert manifest.samples_total == 3
    assert [arm.usage for arm in manifest.arms] == [None, None, None]
    assert not results_path(manifest.run_id).exists()


def test_a_partial_run_records_usage_only_for_the_arms_it_reached(
    data_dirs: Path,
) -> None:
    """Arms run sequentially, so a killed run leaves the ones it never asked null."""
    with pytest.raises(RuntimeError, match="connection dropped"):
        run(_plan(samples_per_arm=2), CrashingBackend(dies_on=3))

    arms = _only_manifest(data_dirs).arms
    assert [arm.lang for arm in arms] == ["en", "pl", "ja"]
    assert arms[0].usage is not None
    assert arms[0].usage.prompt_tokens == 24
    assert [arm.usage for arm in arms[1:]] == [None, None]


def test_run_resolves_the_backend_its_question_names(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller passes a backend only to stand in; a run knows its own provider."""
    asked = _record_resolved_providers(monkeypatch, fake_backend)

    outcome = run(_plan())

    assert asked == ["openai"]
    assert outcome.rows_written == 3


def test_run_records_provenance_tokens_and_cost(fake_backend: FakeBackend) -> None:
    run(_plan(), fake_backend)

    frame = _raw()
    assert all(frame["prompt"].to_list())
    assert frame["model_snapshot"].to_list() == ["gpt-5.6-luna-fake"] * 3
    assert frame["response_id"].to_list() == ["chatcmpl-fake"] * 3
    assert frame["response_envelope"].to_list()[0] is not None
    assert frame["prompt_tokens"].to_list() == [12] * 3
    assert frame["cached_tokens"].to_list() == [4] * 3
    assert frame["reasoning_tokens"].to_list() == [1] * 3
    assert frame["pricing_version"].to_list() == ["2026-07-24"] * 3
    cost = frame["total_cost_usd"].to_list()[0]
    assert cost is not None
    assert cost > 0


def test_a_run_guards_its_own_spending(
    fake_backend: FakeBackend, data_dirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard sits where the money is spent, so no caller can skip it."""
    unpriced = data_dirs / "unpriced.json"
    unpriced.write_text(
        PricingTable(currency="USD", unit="per_1m_tokens", models={}).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(pricing_module, "PRICING_FILE", unpriced)
    planned = _plan()

    assert planned.price is None
    with pytest.raises(ValueError, match="No price for model"):
        run(planned, fake_backend)


def test_a_run_larger_than_the_limit_is_refused_without_force(
    fake_backend: FakeBackend,
) -> None:
    planned = _plan(samples_per_arm=COST_GUARD_CALLS)

    with pytest.raises(ValueError, match="without --force"):
        run(planned, fake_backend)

    assert run(planned, fake_backend, force=True).rows_written == 3 * COST_GUARD_CALLS


def test_a_rerun_is_more_samples_rather_than_a_replacement(
    fake_backend: FakeBackend,
) -> None:
    """Two runs of one question are more samples of it, so both files are kept."""
    first = run(_plan(samples_per_arm=2), fake_backend)
    second = run(_plan(samples_per_arm=2), fake_backend)

    assert first.run_id != second.run_id
    assert first.rows_written == second.rows_written == 6
    assert _raw("001a__*.jsonl").height == 12


def test_a_run_id_names_the_question_and_when_it_started(
    fake_backend: FakeBackend,
) -> None:
    outcome = run(_plan(), fake_backend)

    assert re.fullmatch(r"001a__\d{8}T\d{9}Z", outcome.run_id)


def test_refusals_persist_with_an_empty_answer() -> None:
    outcome = run(_plan(), RefusingBackend())

    frame = _raw()
    assert outcome.rows_written == 3
    assert frame["answer"].to_list() == [""] * 3
    assert frame["raw_json"].to_list() == [None] * 3
    assert frame["refusal"].to_list() == ["I can't help with that."] * 3
    assert frame["total_cost_usd"].to_list() == [None] * 3
    assert frame["prompt_tokens"].to_list() == [None] * 3


def test_every_row_carries_the_schema_it_was_asked_under(
    fake_backend: FakeBackend,
) -> None:
    """The schema itself is stored, so the raw data explains itself alone."""
    outcome = run(_plan(), fake_backend)

    frame = _raw()
    assert json.loads(frame["response_schema"].to_list()[0]) == (
        FruitChoice.model_json_schema()
    )
    arm = outcome.manifest.arms[0]
    assert arm.response_schema == FruitChoice.model_json_schema()
    recorded = json.loads(frame["prompt_inputs"].to_list()[0])
    assert recorded["fruit_list"] == outcome.manifest.inputs["fruit_list"]["order_ids"]
    assert outcome.manifest.input_sha256["fruit_list"]


def test_one_run_covers_every_arm_a_question_declares() -> None:
    """001d asks one language three ways, so one run writes all three arms."""
    planned = _plan(question="001d", samples_per_arm=2)

    assert [arm.label for arm in planned.question.arms] == [
        "FruitChoice",
        "WyborOwocu",
        FREE_TEXT,
    ]
    assert planned.samples_total == 6

    outcome = run(planned, PolishBackend())

    assert outcome.manifest.samples_total == 6
    assert [arm.schema_name for arm in outcome.manifest.arms] == [
        "FruitChoice",
        "WyborOwocu",
        None,
    ]
    assert outcome.manifest.arms[1].response_schema == WyborOwocu.model_json_schema()

    frame = _raw()
    assert outcome.rows_written == 6
    assert frame["lang"].to_list() == ["pl"] * 6
    assert frame["answer"].to_list() == ["jabłko"] * 6
    assert frame["response_schema"].str.json_path_match("$.title").to_list() == (
        ["FruitChoice"] * 2 + ["WyborOwocu"] * 2 + [None] * 2
    )

    free_text = frame.filter(pl.col("response_schema").is_null())
    assert free_text.height == 2
    assert free_text["raw_json"].to_list() == ["jabłko"] * 2


def test_run_records_usage_for_the_whole_run(fake_backend: FakeBackend) -> None:
    outcome = run(_plan(samples_per_arm=2), fake_backend)

    usage = outcome.manifest.usage
    assert usage is not None
    assert usage.prompt_tokens == 72
    assert usage.errors == 0
    assert usage.provider_refusals == 0


def test_usage_is_recorded_for_each_arm_as_well_as_the_run(
    fake_backend: FakeBackend,
) -> None:
    """Every arm carries its own share of what the run used, not the whole of it."""
    outcome = run(_plan(samples_per_arm=2), fake_backend)

    arms = outcome.manifest.arms
    assert [arm.lang for arm in arms] == ["en", "pl", "ja"]
    assert all(arm.usage is not None and arm.usage.prompt_tokens == 24 for arm in arms)
    assert outcome.manifest.usage is not None
    assert outcome.manifest.usage.prompt_tokens == 72


def test_arm_usage_separates_arms_that_share_a_language() -> None:
    """001d asks one language three ways, so an arm is its schema as much as it."""
    outcome = run(_plan(question="001d", samples_per_arm=2), PolishBackend())

    arms = outcome.manifest.arms
    assert [arm.schema_name for arm in arms] == ["FruitChoice", "WyborOwocu", None]
    assert all(arm.usage is not None and arm.usage.prompt_tokens == 24 for arm in arms)
    assert outcome.manifest.usage is not None
    assert outcome.manifest.usage.prompt_tokens == 72


def test_usage_counts_provider_refusals_and_keeps_their_cost_null() -> None:
    outcome = run(_plan(samples_per_arm=2), RefusingBackend())

    usage = outcome.manifest.usage
    assert usage is not None
    assert usage.provider_refusals == 6
    assert usage.total_tokens == 0
    assert usage.total_cost_usd == 0.0
