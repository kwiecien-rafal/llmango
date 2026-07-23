"""Tests for aggregation: distributions, refusal rate, drift and 'other' share."""

import json
from pathlib import Path

import polars as pl
import pytest

from llmango.analyze import analyze_question
from llmango.lang_detect import detect_language
from llmango.storage import write_normalized

_DETECTED = {"apple": "en", "banana": "en", "jabłko": "pl", "coś dziwnego": "pl"}


def _fake_detect(text: str, languages: tuple[str, ...]) -> str | None:
    return _DETECTED.get(text)


@pytest.fixture
def env(data_dirs: Path) -> Path:
    return data_dirs


def _row(lang: str, raw: str, canonical: str, is_fruit: bool) -> dict[str, object]:
    return {
        "lang": lang,
        "fruit_raw": raw,
        "fruit_canonical": canonical,
        "is_fruit": is_fruit,
        "multiple": False,
    }


def _write_normalized(rows: list[dict[str, object]]) -> None:
    schema: dict[str, pl.DataType] = {
        "lang": pl.String(),
        "fruit_raw": pl.String(),
        "fruit_canonical": pl.String(),
        "is_fruit": pl.Boolean(),
        "multiple": pl.Boolean(),
    }
    write_normalized(pl.DataFrame(rows, schema=schema), "favorite_fruit")


def _read(tmp_path: Path, name: str) -> dict[str, object]:
    path = tmp_path / "aggregated" / "favorite_fruit" / name
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def analyzed(env: Path) -> Path:
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
    analyze_question("favorite_fruit", detect=_fake_detect)
    return env


def test_distributions_count_valid_answers_and_report_other(analyzed: Path) -> None:
    languages = _read(analyzed, "distributions.json")["languages"]

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


def test_refusals_are_excluded_from_distribution_but_counted(analyzed: Path) -> None:
    distribution = _read(analyzed, "distributions.json")["languages"]
    refusal = _read(analyzed, "refusal_rate.json")["languages"]

    assert "" not in distribution["en"]["counts"]
    assert refusal["en"] == {"total": 3, "refusals": 1, "rate": 0.3333}
    assert refusal["pl"] == {"total": 3, "refusals": 0, "rate": 0.0}


def test_language_match_scores_in_and_out_of_language_answers(analyzed: Path) -> None:
    languages = _read(analyzed, "language_match.json")["languages"]

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


def test_language_match_counts_undetermined_answers_apart(env: Path) -> None:
    _write_normalized(
        [
            _row("en", "apple", "apple", True),
            _row("en", "mango", "mango", True),
            _row("pl", "jabłko", "apple", True),
            _row("pl", "banan", "banana", True),
        ]
    )

    analyze_question("favorite_fruit", detect=_fake_detect)

    languages = _read(env, "language_match.json")["languages"]
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
        analyze_question("favorite_fruit", detect=_fake_detect)


def test_empty_normalized_parquet_raises(env: Path) -> None:
    _write_normalized([])

    with pytest.raises(ValueError, match="no rows"):
        analyze_question("favorite_fruit", detect=_fake_detect)


def test_detect_language_reads_obvious_sentences() -> None:
    languages = ("en", "pl")
    assert detect_language("this is an english sentence about fruit", languages) == "en"
    assert detect_language("to jest polskie zdanie o owocach", languages) == "pl"
