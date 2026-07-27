"""Tests for Parquet storage: round-trip, column order and dtypes."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from llmango import storage as storage_module
from llmango.storage import read_results, results_path, write_results

_RUN_ID = "001a__en__20260720T101500Z__c3f9a1"


def _row(sample_idx: int, fruit: str) -> dict[str, object]:
    return {
        "question_id": "001a",
        "lang": "en",
        "schema_variant": "en",
        "schema_name": "FruitChoice",
        "model": "gpt-5.6-luna",
        "backend": "fake",
        "run_id": _RUN_ID,
        "sample_idx": sample_idx,
        "seed": 7,
        "temperature": 1.0,
        "prompt_sha256": "deadbeef",
        "prompt": "Pick one random fruit from this list: apple, mango",
        "prompt_inputs": '{"fruit_list": ["apple", "mango"]}',
        "raw_json": f'{{"fruit": "{fruit}"}}',
        "fruit_raw": fruit,
        "model_snapshot": "gpt-5.6-luna-2026-01-01",
        "finish_reason": "stop",
        "refusal": None,
        "error": None,
        "response_id": "chatcmpl-fake",
        "system_fingerprint": "fp_fake",
        "service_tier": "default",
        "provider_created_at": datetime(2026, 7, 20, tzinfo=UTC),
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


def test_write_then_read_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)
    rows = [_row(0, "apple"), _row(1, "mango")]

    path = write_results(rows, _RUN_ID, "gpt-5.6-luna")
    assert path == results_path(_RUN_ID, "gpt-5.6-luna")
    assert path.exists()

    frame = read_results("*.parquet")
    assert frame.height == 2
    assert frame["fruit_raw"].to_list() == ["apple", "mango"]


def test_columns_follow_the_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)

    write_results([_row(0, "apple")], _RUN_ID, "gpt-5.6-luna")

    frame = read_results("*.parquet")
    assert frame.columns == [
        "question_id",
        "lang",
        "schema_variant",
        "schema_name",
        "model",
        "backend",
        "run_id",
        "sample_idx",
        "seed",
        "temperature",
        "prompt_sha256",
        "prompt",
        "prompt_inputs",
        "raw_json",
        "fruit_raw",
        "model_snapshot",
        "finish_reason",
        "refusal",
        "error",
        "response_id",
        "system_fingerprint",
        "service_tier",
        "provider_created_at",
        "request_envelope",
        "response_envelope",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "input_cost_usd",
        "output_cost_usd",
        "total_cost_usd",
        "pricing_version",
        "created_at",
    ]


def test_column_dtypes_are_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)

    write_results([_row(0, "apple")], _RUN_ID, "gpt-5.6-luna")

    frame = read_results("*.parquet")
    assert frame.schema["sample_idx"] == pl.Int64
    assert frame.schema["seed"] == pl.Int64
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


def test_result_file_is_named_after_its_run_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)

    path = results_path(_RUN_ID, "vendor/model-1")

    assert path.name == f"{_RUN_ID}__vendor-model-1.parquet"
    assert path.name.startswith("001a__")


def test_read_results_is_empty_when_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)
    assert read_results("*.parquet").is_empty()
