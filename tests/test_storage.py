"""Tests for Parquet storage: round-trip, dtypes and how runs pool."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from llmango import storage as storage_module
from llmango.rows import column_dtypes
from llmango.storage import read_results, results_path, write_results

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
        "provider_created_at": datetime(2026, 7, 20, tzinfo=UTC),
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
        "created_at": datetime(2026, 7, 20, tzinfo=UTC),
    }


@pytest.fixture(autouse=True)
def _raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write every run of these tests into tmp_path."""
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)
    return tmp_path


def test_write_then_read_round_trips() -> None:
    rows = [_row(0, "apple"), _row(1, "mango")]

    path = write_results(rows, _RUN_ID, column_dtypes({}))

    assert path == results_path(_RUN_ID)
    frame = read_results("*.parquet")
    assert frame.height == 2
    assert frame["answer"].to_list() == ["apple", "mango"]


def test_column_dtypes_are_pinned() -> None:
    write_results([_row(0, "apple")], _RUN_ID, column_dtypes({}))

    frame = read_results("*.parquet")
    assert frame.schema["sample_idx"] == pl.Int64
    assert frame.schema["temperature"] == pl.Float64
    assert frame.schema["raw_json"] == pl.String
    assert frame.schema["prompt_tokens"] == pl.Int64
    assert frame.schema["total_cost_usd"] == pl.Float64
    assert frame.schema["request_envelope"] == pl.String
    assert frame.schema["response_envelope"] == pl.String
    assert frame.schema["provider_created_at"] == pl.Datetime(
        time_unit="us", time_zone="UTC"
    )
    assert frame.schema["created_at"] == pl.Datetime(time_unit="us", time_zone="UTC")


def test_declared_dtypes_pin_a_column_that_is_null_in_every_row() -> None:
    """Declared dtypes keep a parquet schema from varying with one run's data."""
    row = {**_row(0, "apple"), "ripeness": None}

    write_results([row], _RUN_ID, column_dtypes({"ripeness": pl.String()}))

    assert read_results("*.parquet").schema["ripeness"] == pl.String


def test_result_file_is_named_after_its_run() -> None:
    assert results_path(_RUN_ID).name == f"{_RUN_ID}.parquet"


def test_read_results_pools_every_run_of_one_question() -> None:
    """Repeating a configuration is how a sample grows, so runs of it add up.

    Each run writes its own file and a question's glob reads them all, so a
    second run of the same arm is more samples of it rather than a replacement.
    """
    later = "001a__20260721T101500000Z"

    write_results([_row(0, "apple"), _row(1, "mango")], _RUN_ID, column_dtypes({}))
    write_results([_row(0, "pear")], later, column_dtypes({}))

    frame = read_results("001a__*.parquet")
    assert frame.height == 3
    assert frame["answer"].to_list() == ["apple", "mango", "pear"]


def test_read_results_is_empty_when_no_files() -> None:
    assert read_results("*.parquet").is_empty()
