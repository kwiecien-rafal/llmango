"""End-to-end pipeline test: generate, normalize and analyze with a fake backend."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from llmango.analyze import analyze_experiment
from llmango.backends.base import GenerationBackend
from llmango.normalize import normalize_experiment
from llmango.runner import run
from llmango.storage import read_results

_EXPERIMENT = "001_fruit"

_ANSWERS = {
    "en": ["apple", "banana", "banana", ""],
    "pl": ["jabłko", "banan", "coś", ""],
}

_CACHE = {"pl": {"coś": {"canonical": "other", "is_fruit": True, "multiple": False}}}

_DETECTED = {"apple": "en", "banana": "en", "jabłko": "pl", "banan": "pl", "coś": "pl"}


def _detect(text: str, languages: tuple[str, ...]) -> str | None:
    return _DETECTED.get(text)


@pytest.fixture
def pipeline(data_dirs: Path) -> Path:
    directory = data_dirs / "normalization" / _EXPERIMENT
    directory.mkdir(parents=True)
    (directory / "normalization_cache.json").write_text(
        json.dumps(_CACHE), encoding="utf-8"
    )
    return data_dirs


def _aggregate(tmp_path: Path, name: str) -> dict[str, dict[str, object]]:
    path = tmp_path / "aggregated" / _EXPERIMENT / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["questions"]["001a"]["en"]


def test_pipeline_generates_normalizes_and_aggregates(
    pipeline: Path, make_fake_backend: Callable[..., GenerationBackend]
) -> None:
    run_outcome = run(
        "001a",
        make_fake_backend(_ANSWERS),
        samples=4,
        languages=["en", "pl"],
    )
    assert not run_outcome.skipped
    assert run_outcome.rows_written == 8

    raw = read_results("*.parquet")
    assert raw.height == 8
    assert all(rid == "chatcmpl-fake" for rid in raw["response_id"].to_list())
    assert all(
        cost is not None and cost > 0 for cost in raw["total_cost_usd"].to_list()
    )

    normalize_outcome = normalize_experiment("001")
    assert normalize_outcome.rows == 8
    assert normalize_outcome.distinct == 7
    assert normalize_outcome.llm_calls == 0

    analyze_experiment("001", detect=_detect)

    distributions = _aggregate(pipeline, "distributions.json")
    refusals = _aggregate(pipeline, "refusal_rate.json")

    assert distributions["en"]["counts"] == {"apple": 1, "banana": 2}
    assert distributions["pl"]["counts"] == {"apple": 1, "banana": 1, "other": 1}
    assert distributions["pl"]["other_share"] == 0.3333
    assert refusals["en"] == {"total": 4, "refusals": 1, "rate": 0.25}
    assert refusals["pl"] == {"total": 4, "refusals": 1, "rate": 0.25}
    assert not (pipeline / "aggregated" / _EXPERIMENT / "language_match.json").exists()

    for lang, distribution in distributions.items():
        counts: dict[str, int] = distribution["counts"]
        assert sum(counts.values()) == distribution["n"]
        assert distribution["n"] + refusals[lang]["refusals"] == refusals[lang]["total"]
