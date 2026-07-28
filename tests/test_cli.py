"""Tests for the CLI surface: cost guardrails, dry-run plan and normalize report."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmango.cli import _report_normalize, app
from llmango.normalize import NormalizeOutcome
from llmango.questions import load_question
from llmango.spec import FREE_TEXT

runner = CliRunner()


def test_large_run_is_refused_without_force() -> None:
    result = runner.invoke(app, ["run", "001a", "--samples", "100"])

    assert result.exit_code == 1
    assert "without --force" in result.output


def test_smoke_and_samples_cannot_be_combined() -> None:
    result = runner.invoke(app, ["run", "--smoke", "--samples", "3"])

    assert result.exit_code == 1
    assert "not both" in result.output


def test_dry_run_reports_the_plan_and_writes_nothing(data_dirs: Path) -> None:
    result = runner.invoke(app, ["run", "001a", "--dry-run", "--samples", "3"])

    expected_requests = len(load_question("001a").arms) * 3
    assert result.exit_code == 0
    assert "Plan for 001a via openai" in result.output
    assert "model:       gpt-5.6-luna" in result.output
    assert f"requests:    {expected_requests} total" in result.output
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
    """Only a question id resolves, and the error says which ones exist."""
    result = runner.invoke(app, [command, "001"])

    assert result.exit_code == 1
    assert "Unknown question: '001'" in result.output
    assert "001a, 001b, 001c, 001d" in result.output


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
            parquet_path=tmp_path / "001_fruit.parquet",
            rows=8,
            distinct=7,
            llm_calls=0,
        )
    )

    out = capsys.readouterr().out
    assert "0 resolved by the LLM" in out
    assert "Parquet:" in out
