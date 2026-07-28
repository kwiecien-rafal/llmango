"""End-to-end pipeline: generate, normalize, aggregate and chart a fake backend."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from llmango.aggregate import aggregate_question
from llmango.backends.base import Backend
from llmango.charts import analyze_question
from llmango.normalize import normalize_question
from llmango.runner import plan, run
from llmango.storage import read_results

_QUESTION = "001a"
_FOLDER = "001_fruit"

_ANSWERS = {
    "en": ["apple", "banana", "banana", ""],
    "pl": ["jabłko", "banan", "coś", ""],
}

_CACHE = {"pl": {"coś": {"canonical": "other", "is_valid": True, "multiple": False}}}


@pytest.fixture
def pipeline(data_dirs: Path) -> Path:
    directory = data_dirs / "mappings" / _FOLDER
    directory.mkdir(parents=True)
    (directory / "normalization_cache.json").write_text(
        json.dumps(_CACHE), encoding="utf-8"
    )
    return data_dirs


def _aggregate(tmp_path: Path) -> dict[str, dict[str, object]]:
    path = tmp_path / "aggregated" / f"{_QUESTION}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["distributions"]["FruitChoice"]


def test_pipeline_generates_normalizes_aggregates_and_charts(
    pipeline: Path, make_fake_backend: Callable[..., Backend]
) -> None:
    backend: Backend = make_fake_backend(_ANSWERS)
    planned = plan(_QUESTION, samples_per_arm=4, languages=["en", "pl"])
    run_outcome = run(planned, backend)
    assert run_outcome.rows_written == 8

    raw = read_results("*.parquet")
    assert raw.height == 8
    assert all(rid == "chatcmpl-fake" for rid in raw["response_id"].to_list())
    assert all(
        cost is not None and cost > 0 for cost in raw["total_cost_usd"].to_list()
    )

    normalize_outcome = normalize_question(_QUESTION)
    assert normalize_outcome.rows == 8
    assert normalize_outcome.distinct == 7
    assert normalize_outcome.llm_calls == 0

    aggregate_question(_QUESTION)

    distributions = _aggregate(pipeline)

    assert distributions["en"]["counts"] == {"apple": 1, "banana": 2}
    assert distributions["pl"]["counts"] == {"apple": 1, "banana": 1, "other": 1}
    assert distributions["pl"]["other_share"] == 0.3333

    for distribution in distributions.values():
        counts: dict[str, int] = distribution["counts"]
        assert sum(counts.values()) == distribution["n"]
        assert distribution["n"] == 3

    analyze_outcome = analyze_question(_QUESTION)

    assert [chart.file for chart in analyze_outcome.charts] == ["distribution.svg"]
    charts = pipeline / "charts" / _QUESTION
    assert (charts / "distribution.svg").read_text(encoding="utf-8").count("<svg")
    index = json.loads((charts / "index.json").read_text(encoding="utf-8"))
    distribution = index["charts"][0]
    assert distribution["columns"] == ["en", "pl"]
    labels = [row["label"] for row in distribution["rows"]]
    assert labels == ["banana", "apple", "other"]
