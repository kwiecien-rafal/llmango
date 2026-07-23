"""End-to-end pipeline test: generate, normalize and analyze with a fake backend."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from llmango.analyze import analyze_question
from llmango.backends.base import GenerationBackend
from llmango.normalize import normalize_question
from llmango.runner import run

_SLUG = "001_favorite_fruit"

_ANSWERS = {
    "en": ["apple", "banana", "banana", ""],
    "pl": ["jabłko", "banan", "coś", ""],
}

_MAPPING = "apple: apple\nbanana: banana\njabłko: apple\nbanan: banana\n"

_CACHE = {"pl": {"coś": {"canonical": "other", "is_fruit": True, "multiple": False}}}

_DETECTED = {"apple": "en", "banana": "en", "jabłko": "pl", "banan": "pl", "coś": "pl"}


def _detect(text: str, languages: tuple[str, ...]) -> str | None:
    return _DETECTED.get(text)


@pytest.fixture
def pipeline(data_dirs: Path) -> Path:
    directory = data_dirs / "normalization" / _SLUG
    directory.mkdir(parents=True)
    (directory / "mapping.yaml").write_text(_MAPPING, encoding="utf-8")
    (directory / "normalization_cache.json").write_text(
        json.dumps(_CACHE), encoding="utf-8"
    )
    return data_dirs


def _aggregate(tmp_path: Path, name: str) -> dict[str, dict[str, object]]:
    path = tmp_path / "aggregated" / "favorite_fruit" / name
    return json.loads(path.read_text(encoding="utf-8"))["languages"]


def test_pipeline_generates_normalizes_and_aggregates(
    pipeline: Path, make_fake_backend: Callable[..., GenerationBackend]
) -> None:
    run_outcome = run(
        "favorite_fruit", make_fake_backend(_ANSWERS), samples=4, languages=["en", "pl"]
    )
    assert not run_outcome.skipped
    assert run_outcome.rows_written == 8

    normalize_outcome = normalize_question("favorite_fruit")
    assert normalize_outcome.rows == 8
    assert normalize_outcome.distinct == 7
    assert normalize_outcome.llm_calls == 0

    analyze_question("favorite_fruit", detect=_detect)

    distributions = _aggregate(pipeline, "distributions.json")
    refusals = _aggregate(pipeline, "refusal_rate.json")
    matches = _aggregate(pipeline, "language_match.json")

    assert distributions["en"]["counts"] == {"apple": 1, "banana": 2}
    assert distributions["pl"]["counts"] == {"apple": 1, "banana": 1, "other": 1}
    assert distributions["pl"]["other_share"] == 0.3333
    assert refusals["en"] == {"total": 4, "refusals": 1, "rate": 0.25}
    assert refusals["pl"] == {"total": 4, "refusals": 1, "rate": 0.25}

    for lang, distribution in distributions.items():
        counts: dict[str, int] = distribution["counts"]
        assert sum(counts.values()) == distribution["n"]
        assert distribution["n"] + refusals[lang]["refusals"] == refusals[lang]["total"]
        match = matches[lang]
        assert match["matched"] + match["undetermined"] <= match["total"]
