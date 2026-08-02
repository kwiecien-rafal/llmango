"""Tests for raw JSONL storage: appending, dtypes, truncation and how runs pool."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from llmango import storage as storage_module
from llmango.rows import column_dtypes
from llmango.storage import append_result, read_results, results_path

_RUN_ID = "001a__20260720T101500000Z"


def _row(sample_idx: int, fruit: str) -> dict[str, object]:
    return {
        "question_id": "001a",
        "lang": "en",
        "model": "gpt-5.6-luna",
        "provider": "fake",
        "run_id": _RUN_ID,
        "sample_idx": sample_idx,
        "temperature": 1.0,
        "prompt_sha256": "deadbeef",
        "prompt": "Pick one random fruit from this list: apple, mango",
        "prompt_inputs": '{"fruit_list": ["apple", "mango"]}',
        "raw_json": f'{{"fruit": "{fruit}"}}',
        "answer": fruit,
        "model_snapshot": "gpt-5.6-luna-2026-01-01",
        "finish_reason": "stop",
        "refusal": None,
        "error": None,
        "response_id": "chatcmpl-fake",
        "service_tier": "default",
        "provider_created_at": datetime(2026, 7, 20, 10, 15, 0, 123456, tzinfo=UTC),
        "response_schema": '{"title": "FruitChoice"}',
        "request_envelope": '{"model": "gpt-5.6-luna"}',
        "response_envelope": '{"id": "chatcmpl-fake"}',
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "cached_tokens": 4,
        "reasoning_tokens": 1,
        "input_cost_usd": 0.0001,
        "output_cost_usd": 0.0002,
        "total_cost_usd": 0.0003,
        "pricing_version": "2026-07-24",
        "generation_seconds": 0.812,
        "created_at": datetime(2026, 7, 20, tzinfo=UTC),
    }


def _append(rows: list[dict[str, object]], run_id: str = _RUN_ID) -> None:
    for row in rows:
        append_result(row, run_id)


def _read(pattern: str = "*.jsonl") -> pl.DataFrame:
    return read_results(pattern, column_dtypes({}))


@pytest.fixture(autouse=True)
def _raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write every run of these tests into tmp_path."""
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)
    return tmp_path


def test_append_then_read_round_trips() -> None:
    _append([_row(0, "apple"), _row(1, "mango")])

    frame = _read()
    assert frame.height == 2
    assert frame["answer"].to_list() == ["apple", "mango"]


def test_every_result_is_on_disk_before_the_next_one_is_asked_for() -> None:
    """Persisting per call is the point: what came back survives what follows."""
    results_file = append_result(_row(0, "apple"), _RUN_ID)

    assert results_file == results_path(_RUN_ID)
    assert results_file.read_text(encoding="utf-8").count("\n") == 1
    assert json.loads(results_file.read_text(encoding="utf-8"))["answer"] == "apple"


def test_column_dtypes_are_pinned() -> None:
    _append([_row(0, "apple")])

    frame = _read()
    assert frame.schema["sample_idx"] == pl.Int64
    assert frame.schema["temperature"] == pl.Float64
    assert frame.schema["raw_json"] == pl.String
    assert frame.schema["prompt_tokens"] == pl.Int64
    assert frame.schema["total_cost_usd"] == pl.Float64
    assert frame.schema["generation_seconds"] == pl.Float64
    assert frame.schema["request_envelope"] == pl.String
    assert frame.schema["response_envelope"] == pl.String
    assert frame.schema["provider_created_at"] == pl.Datetime(
        time_unit="us", time_zone="UTC"
    )
    assert frame.schema["created_at"] == pl.Datetime(time_unit="us", time_zone="UTC")


def test_a_timestamp_survives_json_to_the_microsecond() -> None:
    """JSON has no datetime, so the storage seam encodes and restores both ways."""
    _append([_row(0, "apple")])

    frame = _read().select(
        pl.col("provider_created_at").dt.microsecond().alias("micros"),
        (
            pl.col("provider_created_at")
            == datetime(2026, 7, 20, 10, 15, 0, 123456, tzinfo=UTC)
        ).alias("exact"),
    )
    assert frame["micros"].to_list() == [123456]
    assert frame["exact"].to_list() == [True]


def test_a_null_timestamp_stays_null() -> None:
    _append([{**_row(0, "apple"), "provider_created_at": None}])

    frame = _read()
    assert frame.schema["provider_created_at"] == pl.Datetime(
        time_unit="us", time_zone="UTC"
    )
    assert frame.select(pl.col("provider_created_at").is_null())["provider_created_at"][
        0
    ]


def test_declared_dtypes_pin_a_column_that_is_null_in_every_row() -> None:
    """Declared dtypes keep a frame's schema from varying with one run's data."""
    _append([{**_row(0, "apple"), "ripeness": None}])

    frame = read_results("*.jsonl", column_dtypes({"ripeness": pl.String()}))
    assert frame.schema["ripeness"] == pl.String


def test_a_half_written_final_line_is_dropped_rather_than_failing_the_read() -> None:
    """A killed run can cut its last line mid-write; the rest is still every result."""
    _append([_row(0, "apple"), _row(1, "mango")])
    results_file = results_path(_RUN_ID)
    text = results_file.read_text(encoding="utf-8")
    results_file.write_text(text + '{"question_id": "001a", "lan', encoding="utf-8")

    frame = _read()
    assert frame.height == 2
    assert frame["answer"].to_list() == ["apple", "mango"]


def test_result_file_is_named_after_its_run() -> None:
    assert results_path(_RUN_ID).name == f"{_RUN_ID}.jsonl"


def test_read_results_pools_every_run_of_one_question() -> None:
    """Repeating a configuration is how a sample grows, so runs of it add up.

    Each run writes its own file and a question's glob reads them all, so a
    second run of the same arm is more samples of it rather than a replacement.
    """
    later = "001a__20260721T101500000Z"

    _append([_row(0, "apple"), _row(1, "mango")])
    _append([_row(0, "pear")], run_id=later)

    frame = _read("001a__*.jsonl")
    assert frame.height == 3
    assert frame["answer"].to_list() == ["apple", "mango", "pear"]


def test_read_results_is_empty_when_no_files() -> None:
    assert _read().is_empty()
