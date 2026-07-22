"""Tests for Parquet storage: round-trip, column order and dtypes."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from llmango import storage as storage_module
from llmango.storage import read_results, results_path, write_results


def _row(sample_idx: int, fruit: str) -> dict[str, object]:
    return {
        "question_id": "favorite_fruit",
        "lang": "en",
        "model": "gpt-5.6-luna",
        "backend": "fake",
        "run_id": "run-001",
        "sample_idx": sample_idx,
        "seed": 7,
        "temperature": 1.0,
        "prompt_sha256": "deadbeef",
        "raw_json": f'{{"fruit": "{fruit}"}}',
        "fruit_raw": fruit,
        "created_at": datetime(2026, 7, 20, tzinfo=UTC),
    }


def test_write_then_read_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)
    rows = [_row(0, "apple"), _row(1, "mango")]

    path = write_results(rows, "favorite_fruit", "gpt-5.6-luna", "run-001")
    assert path == results_path("favorite_fruit", "gpt-5.6-luna", "run-001")
    assert path.exists()

    frame = read_results("*.parquet")
    assert frame.height == 2
    assert frame["fruit_raw"].to_list() == ["apple", "mango"]


def test_columns_follow_the_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)

    write_results([_row(0, "apple")], "favorite_fruit", "gpt-5.6-luna", "run-001")

    frame = read_results("*.parquet")
    assert frame.columns == [
        "question_id",
        "lang",
        "model",
        "backend",
        "run_id",
        "sample_idx",
        "seed",
        "temperature",
        "prompt_sha256",
        "raw_json",
        "fruit_raw",
        "created_at",
    ]


def test_column_dtypes_are_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)

    write_results([_row(0, "apple")], "favorite_fruit", "gpt-5.6-luna", "run-001")

    frame = read_results("*.parquet")
    assert frame.schema["sample_idx"] == pl.Int64
    assert frame.schema["seed"] == pl.Int64
    assert frame.schema["temperature"] == pl.Float64
    assert frame.schema["raw_json"] == pl.String
    assert frame.schema["created_at"] == pl.Datetime(time_unit="us", time_zone="UTC")


def test_read_results_is_empty_when_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path)
    assert read_results("*.parquet").is_empty()
