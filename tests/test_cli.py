"""Tests for the CLI surface: cost guardrails, dry-run plan and stage reports."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmango.cli import (
    _BAR_WIDTH,
    _report_normalize,
    _report_outcome,
    _report_progress,
    app,
)
from llmango.manifest import Manifest, UsageTotals
from llmango.normalize import NormalizeOutcome
from llmango.pricing import PricingEntry
from llmango.questions import load_question
from llmango.runner import RunOutcome
from llmango.spec import FREE_TEXT

runner = CliRunner()

_RUN_ID = "001a__20260720T101500000Z"


def _outcome(samples_written: int, tmp_path: Path) -> RunOutcome:
    """One run's outcome, as a run of six samples that got that far would report."""
    manifest = Manifest(
        run_id=_RUN_ID,
        question_id="001a",
        provider="openai",
        model="gpt-5.6-luna",
        temperature=1.0,
        samples_total=6,
        samples_per_arm=2,
        samples_written=samples_written,
        arms=[],
        pricing=PricingEntry(
            input=0.05, cached_input=0.005, output=0.4, last_updated="2026-07-24"
        ),
        usage=UsageTotals(),
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
    )

    return RunOutcome(
        manifest=manifest,
        results_path=tmp_path / f"{_RUN_ID}.jsonl",
        manifest_path=tmp_path / f"{_RUN_ID}.json",
    )


def test_large_run_is_refused_without_force() -> None:
    result = runner.invoke(app, ["run", "001a", "--samples", "100"])

    assert result.exit_code == 1
    assert "without --force" in result.output


def test_dry_run_reports_the_plan_and_writes_nothing(data_dirs: Path) -> None:
    result = runner.invoke(app, ["run", "001a", "--dry-run", "--samples", "3"])

    expected_total = len(load_question("001a").arms) * 3
    assert result.exit_code == 0
    assert "Plan for 001a via openai" in result.output
    assert "model:       gpt-5.6-luna" in result.output
    assert f"samples:     {expected_total} total, 3 per arm" in result.output
    assert not (data_dirs / "runs").exists()


def test_dry_run_reports_every_arm_of_one_plan(data_dirs: Path) -> None:
    """001d asks Polish three ways, which is three arms of a single run."""
    result = runner.invoke(app, ["run", "001d", "--dry-run"])

    assert result.exit_code == 0
    assert result.output.count("Plan for 001d") == 1
    assert "arms:        3" in result.output
    assert "FruitChoice  pl" in result.output
    assert "WyborOwocu  pl" in result.output
    assert f"{FREE_TEXT}  pl" in result.output


@pytest.mark.parametrize("command", ["normalize", "aggregate", "analyze"])
def test_an_experiment_reference_is_not_a_question(
    data_dirs: Path, command: str
) -> None:
    """No stage resolves '001', and each names the id it refused."""
    result = runner.invoke(app, [command, "001"])

    assert result.exit_code == 1
    assert "001" in result.output


def test_normalize_names_the_questions_that_exist(data_dirs: Path) -> None:
    """Only the stages that consult the registry can list the ids that do resolve."""
    result = runner.invoke(app, ["normalize", "001"])

    assert "Unknown question: '001'" in result.output
    assert "001a, 001b, 001c, 001d" in result.output


def test_report_progress_draws_one_bar_per_arm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bar redraws over itself until its arm is done, which ends its line."""
    arm = load_question("001a").arms[0]

    _report_progress(arm, 3, 5)
    _report_progress(arm, 5, 5)

    out = capsys.readouterr().out
    assert out.startswith("\r  FruitChoice  en  ")
    assert "3/5" in out
    assert f"{'#' * _BAR_WIDTH} 5/5\n" in out
    assert out.count("\n") == 1


def test_report_outcome_reports_a_finished_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _report_outcome(_outcome(6, tmp_path))

    out = capsys.readouterr().out
    assert f"Run {_RUN_ID}: wrote 6 rows." in out
    assert f"Results:  {tmp_path / f'{_RUN_ID}.jsonl'}" in out


def test_report_outcome_says_when_a_run_stopped_short(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial run is more samples short, not a failure, so it says how to finish."""
    _report_outcome(_outcome(3, tmp_path))

    out = capsys.readouterr().out
    assert "stopped after 3 of 6 rows" in out
    assert "Rerun to add the rest." in out


def test_report_normalize_omits_parquet_on_dry_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _report_normalize(
        NormalizeOutcome(parquet_path=None, rows=8, distinct=7, llm_calls=3)
    )

    out = capsys.readouterr().out
    assert "3 would be resolved by the LLM" in out
    assert "Parquet:" not in out


def test_report_normalize_shows_parquet_when_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _report_normalize(
        NormalizeOutcome(
            parquet_path=tmp_path / "001a.parquet",
            rows=8,
            distinct=7,
            llm_calls=0,
        )
    )

    out = capsys.readouterr().out
    assert "0 resolved by the LLM" in out
    assert "Parquet:" in out
