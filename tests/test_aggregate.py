"""Tests for aggregation: distributions, arm labels and the 'other' share."""

import json
from pathlib import Path

import polars as pl
import pytest

from llmango.aggregate import aggregate_question
from llmango.storage import write_normalized

_QUESTION = "001a"
_ARMS = "001d"


@pytest.fixture
def env(data_dirs: Path) -> Path:
    return data_dirs


def _row(
    lang: str,
    answer: str,
    canonical: str,
    is_valid: bool,
    error: str | None = None,
    schema: str | None = "FruitChoice",
) -> dict[str, object]:
    return {
        "question_id": "001a",
        "response_schema": json.dumps({"title": schema}) if schema else None,
        "lang": lang,
        "answer": answer,
        "canonical": canonical,
        "is_valid": is_valid,
        "multiple": False,
        "error": error,
    }


def _write_normalized(rows: list[dict[str, object]], question: str = _QUESTION) -> None:
    schema: dict[str, pl.DataType] = {
        "question_id": pl.String(),
        "response_schema": pl.String(),
        "lang": pl.String(),
        "answer": pl.String(),
        "canonical": pl.String(),
        "is_valid": pl.Boolean(),
        "multiple": pl.Boolean(),
        "error": pl.String(),
    }
    write_normalized(pl.DataFrame(rows, schema=schema), question)


def _read_distributions(
    tmp_path: Path, question: str = _QUESTION
) -> dict[str, dict[str, object]]:
    path = tmp_path / "aggregated" / f"{question}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["question_id"] == question
    return payload["distributions"]


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
    aggregate_question(_QUESTION)
    return env


def test_distributions_count_valid_answers_and_report_other(aggregated: Path) -> None:
    languages = _read_distributions(aggregated)["FruitChoice"]

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

    aggregate_question(_QUESTION)

    distribution = _read_distributions(env)["FruitChoice"]
    assert distribution["en"] == {
        "n": 1,
        "counts": {"apple": 1},
        "other_share": 0.0,
    }


def test_each_schema_is_counted_as_its_own_arm(env: Path) -> None:
    """An arm is titled by the schema its rows carry, and the schemaless one 'none'."""
    _write_normalized(
        [
            _row("pl", "jabłko", "apple", True, schema="WyborOwocu"),
            _row("pl", "banan", "banana", True, schema="WyborOwocu"),
            _row("pl", "jabłko", "apple", True, schema=None),
        ],
        _ARMS,
    )

    aggregate_question(_ARMS)

    distributions = _read_distributions(env, _ARMS)
    assert distributions["WyborOwocu"]["pl"]["counts"] == {"apple": 1, "banana": 1}
    assert distributions["none"]["pl"]["counts"] == {"apple": 1}


def test_an_arm_that_named_nothing_has_no_entry(env: Path) -> None:
    """An arm whose answers all declined leaves no empty distribution behind."""
    _write_normalized(
        [
            _row("pl", "jabłko", "apple", True),
            _row("pl", "", "", False, schema=None),
        ],
        _ARMS,
    )

    aggregate_question(_ARMS)

    assert list(_read_distributions(env, _ARMS)) == ["FruitChoice"]


def test_missing_normalized_parquet_raises(env: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No normalized parquet"):
        aggregate_question(_QUESTION)


def test_empty_normalized_parquet_raises(env: Path) -> None:
    _write_normalized([])

    with pytest.raises(ValueError, match="no rows"):
        aggregate_question(_QUESTION)
