"""Parquet storage for raw generation results.

The common core is byte-identical in every experiment and ends with answer, the
one name every experiment's extracted answer goes by. An experiment's own columns
sit in the single slot between that core and the fixed provenance, usage and cost
blocks, so a consumer reading across experiments always finds the same columns in
the same places.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

from llmango.config import NORMALIZED_DIR, RAW_DIR

COMMON_LEADING_COLUMNS = [
    "question_id",
    "lang",
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
    "answer",
]

PROVENANCE_COLUMNS = [
    "model_snapshot",
    "finish_reason",
    "refusal",
    "error",
    "response_id",
    "system_fingerprint",
    "service_tier",
    "provider_created_at",
    "response_schema",
    "request_envelope",
    "response_envelope",
]

USAGE_COLUMNS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
]

COST_COLUMNS = [
    "input_cost_usd",
    "output_cost_usd",
    "total_cost_usd",
    "pricing_version",
]

FIXED_TRAILING_COLUMNS = PROVENANCE_COLUMNS + USAGE_COLUMNS + COST_COLUMNS

TRAILING_COLUMNS = ["created_at"]

_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "question_id": pl.String(),
    "lang": pl.String(),
    "schema_name": pl.String(),
    "model": pl.String(),
    "backend": pl.String(),
    "run_id": pl.String(),
    "sample_idx": pl.Int64(),
    "seed": pl.Int64(),
    "temperature": pl.Float64(),
    "prompt_sha256": pl.String(),
    "prompt": pl.String(),
    "prompt_inputs": pl.String(),
    "raw_json": pl.String(),
    "answer": pl.String(),
    "model_snapshot": pl.String(),
    "finish_reason": pl.String(),
    "refusal": pl.String(),
    "error": pl.String(),
    "response_id": pl.String(),
    "system_fingerprint": pl.String(),
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


def _ordered_columns(columns: Iterable[str]) -> list[str]:
    """Order present columns: the common core, extras, the fixed block, created_at."""
    present = set(columns)
    known = (
        set(COMMON_LEADING_COLUMNS)
        | set(FIXED_TRAILING_COLUMNS)
        | set(TRAILING_COLUMNS)
    )
    extras = [column for column in columns if column not in known]
    ordered = (
        COMMON_LEADING_COLUMNS + extras + FIXED_TRAILING_COLUMNS + TRAILING_COLUMNS
    )
    return [column for column in ordered if column in present]


def write_results(
    rows: list[dict[str, object]],
    run_id: str,
    model: str,
    extra_dtypes: Mapping[str, pl.DataType] | None = None,
) -> Path:
    """Write result rows to a single Parquet file and return its path.

    Every row of a run carries the same columns, so the order is resolved once
    from the first row rather than rebuilt for each one. An experiment's extra
    columns declare their dtypes here, so a column that happens to be null in
    every row of one run still lands with the type it has in every other run.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    columns = _ordered_columns(rows[0]) if rows else []
    ordered = [{column: row[column] for column in columns} for row in rows]
    overrides = {**_SCHEMA_OVERRIDES, **(extra_dtypes or {})}
    frame = pl.DataFrame(ordered, schema_overrides=overrides)
    path = results_path(run_id, model)
    frame.write_parquet(path)
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
