"""Tests for the CLI surface: cost guardrails, dry-run plan and normalize report."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmango.cli import _report_normalize, app
from llmango.normalize import NormalizeOutcome
from llmango.questions import load_question

runner = CliRunner()


def test_large_run_is_refused_without_force() -> None:
    result = runner.invoke(app, ["run", "001_favorite_fruit", "--samples", "100"])

    assert result.exit_code == 1
    assert "without --force" in result.output


def test_smoke_and_samples_cannot_be_combined() -> None:
    result = runner.invoke(app, ["run", "--smoke", "--samples", "3"])

    assert result.exit_code == 1
    assert "not both" in result.output


def test_dry_run_reports_the_plan_and_writes_nothing(data_dirs: Path) -> None:
    result = runner.invoke(
        app, ["run", "001_favorite_fruit", "--dry-run", "--samples", "3"]
    )

    expected_requests = len(load_question("001_favorite_fruit").languages) * 3
    assert result.exit_code == 0
    assert "Dry run for 001_favorite_fruit" in result.output
    assert f"requests:  {expected_requests} total" in result.output
    assert not (data_dirs / "runs").exists()


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
            parquet_path=tmp_path / "001_favorite_fruit.parquet",
            rows=8,
            distinct=7,
            llm_calls=0,
        )
    )

    out = capsys.readouterr().out
    assert "0 resolved by the LLM" in out
    assert "Parquet:" in out
