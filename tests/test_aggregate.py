"""Tests for aggregation: distributions, drift and the 'other' share."""

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from llmango import aggregate as aggregate_module
from llmango.aggregate import aggregate_experiment
from llmango.lang_detect import detect_language
from llmango.registry import get_experiment
from llmango.storage import write_normalized

_EXPERIMENT = "001_fruit"
_DETECTED = {"apple": "en", "banana": "en", "jabłko": "pl", "coś dziwnego": "pl"}


def _fake_detect(text: str, languages: tuple[str, ...]) -> str | None:
    return _DETECTED.get(text)


@pytest.fixture
def env(data_dirs: Path) -> Path:
    return data_dirs


@pytest.fixture
def drift_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn on the language-match metric for the experiment under test.

    The fruit experiment leaves drift detection off, so the tests that exercise
    the metric's logic enable it on a copy of the spec.
    """
    spec = replace(get_experiment(_EXPERIMENT), detect_language_drift=True)
    monkeypatch.setattr(aggregate_module, "get_experiment", lambda experiment_id: spec)


def _row(
    lang: str,
    raw: str,
    canonical: str,
    is_fruit: bool,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "question_id": "001a",
        "schema_variant": "en",
        "lang": lang,
        "fruit_raw": raw,
        "fruit_canonical": canonical,
        "is_fruit": is_fruit,
        "multiple": False,
        "error": error,
    }


def _write_normalized(rows: list[dict[str, object]]) -> None:
    schema: dict[str, pl.DataType] = {
        "question_id": pl.String(),
        "schema_variant": pl.String(),
        "lang": pl.String(),
        "fruit_raw": pl.String(),
        "fruit_canonical": pl.String(),
        "is_fruit": pl.Boolean(),
        "multiple": pl.Boolean(),
        "error": pl.String(),
    }
    write_normalized(pl.DataFrame(rows, schema=schema), _EXPERIMENT)


def _read_langs(
    tmp_path: Path, name: str, schema_variant: str = "en"
) -> dict[str, object]:
    path = tmp_path / "aggregated" / _EXPERIMENT / name
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
    aggregate_experiment(_EXPERIMENT, detect=_fake_detect)
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

    aggregate_experiment(_EXPERIMENT, detect=_fake_detect)

    distribution = _read_langs(env, "distributions.json")
    assert distribution["en"] == {
        "n": 1,
        "counts": {"apple": 1},
        "other_share": 0.0,
    }


def test_language_match_is_skipped_when_drift_detection_is_disabled(
    aggregated: Path,
) -> None:
    assert not (
        aggregated / "aggregated" / _EXPERIMENT / "language_match.json"
    ).exists()


def test_language_match_scores_in_and_out_of_language_answers(
    env: Path, drift_enabled: None
) -> None:
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

    aggregate_experiment(_EXPERIMENT, detect=_fake_detect)

    languages = _read_langs(env, "language_match.json")

    assert languages["en"] == {
        "total": 2,
        "matched": 2,
        "undetermined": 0,
        "rate": 1.0,
    }
    assert languages["pl"] == {
        "total": 3,
        "matched": 2,
        "undetermined": 0,
        "rate": 0.6667,
    }


def test_language_match_counts_undetermined_answers_apart(
    env: Path, drift_enabled: None
) -> None:
    _write_normalized(
        [
            _row("en", "apple", "apple", True),
            _row("en", "mango", "mango", True),
            _row("pl", "jabłko", "apple", True),
            _row("pl", "banan", "banana", True),
        ]
    )

    aggregate_experiment(_EXPERIMENT, detect=_fake_detect)

    languages = _read_langs(env, "language_match.json")
    assert languages["en"] == {
        "total": 2,
        "matched": 1,
        "undetermined": 1,
        "rate": 1.0,
    }
    assert languages["pl"] == {
        "total": 2,
        "matched": 1,
        "undetermined": 1,
        "rate": 1.0,
    }


def test_missing_normalized_parquet_raises(env: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No normalized parquet"):
        aggregate_experiment(_EXPERIMENT, detect=_fake_detect)


def test_empty_normalized_parquet_raises(env: Path) -> None:
    _write_normalized([])

    with pytest.raises(ValueError, match="no rows"):
        aggregate_experiment(_EXPERIMENT, detect=_fake_detect)


def test_detect_language_reads_obvious_sentences() -> None:
    languages = ("en", "pl")
    assert detect_language("this is an english sentence about fruit", languages) == "en"
    assert detect_language("to jest polskie zdanie o owocach", languages) == "pl"
