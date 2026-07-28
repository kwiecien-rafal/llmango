"""Parquet storage for raw generation results, and the column set they carry."""

from collections.abc import Mapping
from pathlib import Path

import polars as pl

from llmango.config import NORMALIZED_DIR, RAW_DIR

LEADING_COLUMNS: dict[str, pl.DataType] = {
    "question_id": pl.String(),
    "lang": pl.String(),
    "model": pl.String(),
    "provider": pl.String(),
    "run_id": pl.String(),
    "sample_idx": pl.Int64(),
    "temperature": pl.Float64(),
    "prompt_sha256": pl.String(),
    "prompt": pl.String(),
    "prompt_inputs": pl.String(),
    "raw_json": pl.String(),
    "answer": pl.String(),
}

TRAILING_COLUMNS: dict[str, pl.DataType] = {
    "model_snapshot": pl.String(),
    "finish_reason": pl.String(),
    "refusal": pl.String(),
    "error": pl.String(),
    "response_id": pl.String(),
    "service_tier": pl.String(),
    "provider_created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "response_schema": pl.String(),
    "request_envelope": pl.String(),
    "response_envelope": pl.String(),
    "prompt_tokens": pl.Int64(),
    "completion_tokens": pl.Int64(),
    "total_tokens": pl.Int64(),
    "cached_tokens": pl.Int64(),
    "reasoning_tokens": pl.Int64(),
    "input_cost_usd": pl.Float64(),
    "output_cost_usd": pl.Float64(),
    "total_cost_usd": pl.Float64(),
    "pricing_version": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}


def _slugify(value: str) -> str:
    """Make a model id safe to use inside a file name."""
    return value.replace("/", "-").replace("\\", "-")


def results_path(run_id: str, model: str) -> Path:
    """Return the Parquet path for one run."""
    return RAW_DIR / f"{run_id}__{_slugify(model)}.parquet"


def _ordered_columns(row: Mapping[str, object]) -> list[str]:
    """Order a run's columns: the core, the experiment's extras, then the rest."""
    extras = [
        column
        for column in row
        if column not in LEADING_COLUMNS and column not in TRAILING_COLUMNS
    ]
    return [*LEADING_COLUMNS, *extras, *TRAILING_COLUMNS]


def write_results(
    rows: list[dict[str, object]],
    run_id: str,
    model: str,
    extra_dtypes: Mapping[str, pl.DataType] | None = None,
) -> Path:
    """Write every declared column of each row to one Parquet file, filling nulls."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    columns = _ordered_columns(rows[0]) if rows else []
    ordered = [{column: row.get(column) for column in columns} for row in rows]
    dtypes = {**LEADING_COLUMNS, **TRAILING_COLUMNS, **(extra_dtypes or {})}
    path = results_path(run_id, model)
    pl.DataFrame(ordered, schema_overrides=dtypes).write_parquet(path)
    return path


def read_results(pattern: str = "*.parquet") -> pl.DataFrame:
    """Read and concatenate raw result files matching a glob under data/raw/."""
    paths = sorted(RAW_DIR.glob(pattern))
    if not paths:
        return pl.DataFrame()
    return pl.concat(pl.read_parquet(path) for path in paths)


def normalized_path(question_id: str) -> Path:
    """Return the normalized Parquet path for a question under data/normalized/."""
    return NORMALIZED_DIR / f"{question_id}.parquet"


def write_normalized(frame: pl.DataFrame, question_id: str) -> Path:
    """Write a question's normalized frame to a single Parquet file and return it."""
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    path = normalized_path(question_id)
    frame.write_parquet(path)
    return path


def read_normalized(question_id: str) -> pl.DataFrame:
    """Read a question's normalized results, or an empty frame if none exist."""
    path = normalized_path(question_id)
    if not path.is_file():
        return pl.DataFrame()
    return pl.read_parquet(path)
