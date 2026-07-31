"""End-to-end pipeline: generate, normalize, aggregate and chart a fake backend."""

import json
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from llmango.aggregate import aggregate_question
from llmango.analyze import analyze_question
from llmango.backends.base import Backend
from llmango.experiments.e001_fruit import experiment as fruit_module
from llmango.normalize import normalize_question
from llmango.rows import column_dtypes
from llmango.runner import plan, run
from llmango.storage import normalized_path, read_results

_QUESTION = "001a"
_FOLDER = "e001_fruit"

_EN_ANSWERS = ["apple", "banana", "banana", ""]
_PL_ANSWERS = ["jabłko", "banan", "coś", ""]
_JA_ANSWERS = ["りんご", "バナナ", "バナナ", ""]

_STORED_MAP = "pl:\n  coś: other\n"


@pytest.fixture
def pipeline(data_dirs: Path) -> Path:
    """Seed the one off-list answer, so the pipeline needs no normalization call."""
    fruit_module._NORMALIZATION_MAP.write_text(_STORED_MAP, encoding="utf-8")
    return data_dirs


def _aggregate(tmp_path: Path) -> dict[str, dict[str, object]]:
    path = tmp_path / "aggregated" / f"{_QUESTION}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["distributions"]["FruitChoice"]


def test_pipeline_generates_normalizes_aggregates_and_charts(
    pipeline: Path, make_fake_backend: Callable[..., Backend]
) -> None:
    backend: Backend = make_fake_backend(_EN_ANSWERS + _PL_ANSWERS + _JA_ANSWERS)
    planned = plan(_QUESTION, samples_per_arm=4)
    run_outcome = run(planned, backend)
    assert run_outcome.rows_written == 12

    raw = read_results("*.jsonl", column_dtypes({}))
    assert raw.height == 12
    assert all(rid == "chatcmpl-fake" for rid in raw["response_id"].to_list())
    assert all(
        cost is not None and cost > 0 for cost in raw["total_cost_usd"].to_list()
    )

    normalize_outcome = normalize_question(_QUESTION)
    assert normalize_outcome.rows == 12
    assert normalize_outcome.distinct == 10
    assert normalize_outcome.llm_calls == 0

    normalized = pl.read_parquet(normalized_path(_QUESTION))
    timestamp = pl.Datetime(time_unit="us", time_zone="UTC")
    assert normalized.schema["created_at"] == timestamp
    assert normalized.schema["provider_created_at"] == timestamp

    aggregate_question(_QUESTION)

    distributions = _aggregate(pipeline)

    assert distributions["en"]["counts"] == {"apple": 1, "banana": 2}
    assert distributions["pl"]["counts"] == {"apple": 1, "banana": 1, "other": 1}
    assert distributions["ja"]["counts"] == {"apple": 1, "banana": 2}
    assert distributions["pl"]["other_share"] == 0.3333

    for distribution in distributions.values():
        counts: dict[str, int] = distribution["counts"]
        assert sum(counts.values()) == distribution["n"]
        assert distribution["n"] == 3

    analyze_outcome = analyze_question(_QUESTION)

    assert [chart.file for chart in analyze_outcome.charts] == ["language_drift.svg"]
    assert analyze_outcome.skipped == ["order_effect", "schema_effect"]
    charts = pipeline / "charts" / _FOLDER
    assert (charts / "language_drift.svg").read_text(encoding="utf-8").count("<svg")
    index = json.loads((charts / "index.json").read_text(encoding="utf-8"))
    drift = index["charts"][0]
    assert drift["questions"] == [_QUESTION]
    assert drift["columns"] == ["en", "ja", "pl"]
    labels = [row["label"] for row in drift["rows"]]
    assert labels == ["banana", "apple", "other"]
