"""Tests for aggregation: distributions, arm labels and the 'other' share."""

import json
from pathlib import Path

import polars as pl
import pytest

from llmango.aggregate import aggregate_question
from llmango.config import get_aggregate_path
from llmango.storage import write_normalized

_FOLDER = "e001_fruit"
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
    chosen_position: int | None = None,
) -> dict[str, object]:
    return {
        "question_id": "001a",
        "response_schema": json.dumps({"title": schema}) if schema else None,
        "lang": lang,
        "answer": answer,
        "canonical": canonical,
        "is_valid": is_valid,
        "error": error,
        "chosen_position": chosen_position,
    }


def _write_normalized(rows: list[dict[str, object]], question: str = _QUESTION) -> None:
    schema: dict[str, pl.DataType] = {
        "question_id": pl.String(),
        "response_schema": pl.String(),
        "lang": pl.String(),
        "answer": pl.String(),
        "canonical": pl.String(),
        "is_valid": pl.Boolean(),
        "error": pl.String(),
        "chosen_position": pl.Int64(),
    }
    write_normalized(pl.DataFrame(rows, schema=schema), _FOLDER, question)


def _read_positions(question: str = _QUESTION) -> dict[str, object]:
    payload = json.loads(
        get_aggregate_path(_FOLDER, question).read_text(encoding="utf-8")
    )
    return payload["positions"]


def _read_distributions(question: str = _QUESTION) -> dict[str, dict[str, object]]:
    payload = json.loads(
        get_aggregate_path(_FOLDER, question).read_text(encoding="utf-8")
    )
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
    languages = _read_distributions()["FruitChoice"]

    assert languages["en"]["n"] == 2
    assert languages["en"]["counts"] == {"apple": 1, "banana": 1}
    assert languages["en"]["other_share"] == 0.0
    assert languages["pl"]["n"] == 3
    assert languages["pl"]["counts"] == {"apple": 2, "other": 1}
    assert languages["pl"]["other_share"] == 0.3333


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

    distribution = _read_distributions()["FruitChoice"]["en"]
    assert distribution["n"] == 1
    assert distribution["counts"] == {"apple": 1}
    assert distribution["other_share"] == 0.0


def test_an_arm_records_how_many_of_its_answers_were_invalid(env: Path) -> None:
    """Filtering invalids away before counting loses the schemaless arm's failure
    rate, which for 001d is a finding rather than noise."""
    _write_normalized(
        [
            _row("en", "apple", "apple", True),
            _row("en", "", "", False),
            _row("en", "", "", False, error="connection reset"),
        ]
    )

    aggregate_question(_QUESTION)

    assert _read_distributions()["FruitChoice"]["en"]["n_invalid"] == 2


def test_an_arm_carries_the_shape_of_its_answers(aggregated: Path) -> None:
    """The chart step never recomputes a number, so every statistic is stored."""
    english = _read_distributions()["FruitChoice"]["en"]

    assert english["coverage"] == 2
    assert 0.0 < english["entropy"] < 1.0
    assert 1.0 < english["effective_choices"] <= 10.0
    assert 0.0 < english["tvd_from_uniform"] < 1.0


def test_counts_are_stored_rather_than_the_intervals_drawn_from_them(
    aggregated: Path,
) -> None:
    """A stored interval could only cover the categories an arm picked, and would
    give the ones it never picked a flat cap; the counts derive all of them."""
    english = _read_distributions()["FruitChoice"]["en"]

    assert "intervals" not in english
    assert english["counts"] == {"apple": 1, "banana": 1}


def test_other_is_left_out_of_the_shape_it_would_distort(env: Path) -> None:
    """'other' is off-menu, so it is not one of the ten options an even spread
    is measured against; counting it would report a wider choice than was offered."""
    _write_normalized([_row("en", "x", "other", True) for _ in range(4)])

    aggregate_question(_QUESTION)

    distribution = _read_distributions()["FruitChoice"]["en"]
    assert distribution["counts"] == {"other": 4}
    assert distribution["coverage"] == 0
    assert distribution["entropy"] == 0.0


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

    distributions = _read_distributions(_ARMS)
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

    assert list(_read_distributions(_ARMS)) == ["FruitChoice"]


def test_where_a_pick_sat_is_counted_beside_which_fruit_it_was(env: Path) -> None:
    """Position is what a shuffled question actually isolates, and the run already
    paid for it; counting only the fruit throws that away."""
    _write_normalized(
        [
            _row("en", "apple", "apple", True, chosen_position=1),
            _row("en", "banana", "banana", True, chosen_position=1),
            _row("en", "mango", "mango", True, chosen_position=4),
        ]
    )

    aggregate_question(_QUESTION)

    positions = _read_positions()["FruitChoice"]["en"]
    assert positions["counts"] == {"1": 2, "4": 1}
    assert positions["n"] == 3


def test_a_run_that_recorded_no_position_leaves_the_block_empty(
    aggregated: Path,
) -> None:
    assert _read_positions() == {}


def test_the_options_offered_are_stored_beside_the_answers(aggregated: Path) -> None:
    """Ten fruits is what an even spread is measured against, and it comes from
    the experiment's own category set rather than from what happened to be picked."""
    payload = json.loads(
        get_aggregate_path(_FOLDER, _QUESTION).read_text(encoding="utf-8")
    )

    assert payload["support"] == 10


def test_missing_normalized_parquet_raises(env: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No data for question 001a"):
        aggregate_question(_QUESTION)


def test_a_parquet_with_nothing_to_count_raises(env: Path) -> None:
    """An empty file and one where every answer declined fail the same way."""
    _write_normalized([])
    with pytest.raises(ValueError, match="No valid answers"):
        aggregate_question(_QUESTION)

    _write_normalized([_row("en", "", "", False)])
    with pytest.raises(ValueError, match="No valid answers"):
        aggregate_question(_QUESTION)
