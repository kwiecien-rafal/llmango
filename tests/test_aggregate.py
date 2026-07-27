"""Tests for aggregation: distributions and the 'other' share."""

import json
from pathlib import Path

import polars as pl
import pytest

from llmango.aggregate import aggregate_experiment
from llmango.storage import write_normalized

_QUESTION = "001a"
_FOLDER = "001_fruit"


@pytest.fixture
def env(data_dirs: Path) -> Path:
    return data_dirs


def _row(
    lang: str,
    answer: str,
    canonical: str,
    is_valid: bool,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "question_id": "001a",
        "schema_variant": "en",
        "lang": lang,
        "answer": answer,
        "canonical": canonical,
        "is_valid": is_valid,
        "multiple": False,
        "error": error,
    }


def _write_normalized(rows: list[dict[str, object]]) -> None:
    schema: dict[str, pl.DataType] = {
        "question_id": pl.String(),
        "schema_variant": pl.String(),
        "lang": pl.String(),
        "answer": pl.String(),
        "canonical": pl.String(),
        "is_valid": pl.Boolean(),
        "multiple": pl.Boolean(),
        "error": pl.String(),
    }
    write_normalized(pl.DataFrame(rows, schema=schema), _FOLDER)


def _read_langs(
    tmp_path: Path, name: str, schema_variant: str = "en"
) -> dict[str, object]:
    path = tmp_path / "aggregated" / _FOLDER / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["questions"]["001a"][schema_variant]


@pytest.fixture
def aggregated(env: Path) -> Path:
    _write_normalized(
        [
            _row("en", "apple", "apple", True),
            _row("en", "banana", "banana", True),
            _row("en", "", "", False),
            _row("pl", "jabłko", "apple", True),
            _row("pl", "apple", "apple", True),
            _row("pl", "coś dziwnego", "other", True),
        ]
    )
    aggregate_experiment(_QUESTION)
    return env


def test_distributions_count_valid_answers_and_report_other(aggregated: Path) -> None:
    languages = _read_langs(aggregated, "distributions.json")

    assert languages["en"] == {
        "n": 2,
        "counts": {"apple": 1, "banana": 1},
        "other_share": 0.0,
    }
    assert languages["pl"] == {
        "n": 3,
        "counts": {"apple": 2, "other": 1},
        "other_share": 0.3333,
    }


def test_answers_that_name_no_category_stay_out_of_the_distribution(
    env: Path,
) -> None:
    """A declined answer and a failed call are both absent, not an empty category."""
    _write_normalized(
        [
            _row("en", "apple", "apple", True),
            _row("en", "", "", False),
            _row("en", "", "", False, error="connection reset"),
        ]
    )

    aggregate_experiment(_QUESTION)

    distribution = _read_langs(env, "distributions.json")
    assert distribution["en"] == {
        "n": 1,
        "counts": {"apple": 1},
        "other_share": 0.0,
    }


def test_missing_normalized_parquet_raises(env: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No normalized parquet"):
        aggregate_experiment(_QUESTION)


def test_empty_normalized_parquet_raises(env: Path) -> None:
    _write_normalized([])

    with pytest.raises(ValueError, match="no rows"):
        aggregate_experiment(_QUESTION)
