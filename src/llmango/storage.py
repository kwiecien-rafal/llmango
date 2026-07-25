"""Parquet storage for raw generation results.

Raw results are written one Parquet file per (question, model, run). The common
columns are fixed here per CLAUDE.md; each experiment contributes its own parsed
fields via its to_row hook, which land between raw_json and the provenance block.
After the parsed fields comes a fixed block recording the provenance, token usage
and cost of each generation, then created_at.
"""

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from llmango.config import NORMALIZED_DIR, RAW_DIR

COMMON_LEADING_COLUMNS = [
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
    "option_order",
    "raw_json",
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
    "schema_variant": pl.String(),
    "schema_name": pl.String(),
    "model": pl.String(),
    "backend": pl.String(),
    "run_id": pl.String(),
    "sample_idx": pl.Int64(),
    "seed": pl.Int64(),
    "temperature": pl.Float64(),
    "prompt_sha256": pl.String(),
    "prompt": pl.String(),
    "option_order": pl.String(),
    "raw_json": pl.String(),
    "model_snapshot": pl.String(),
    "finish_reason": pl.String(),
    "refusal": pl.String(),
    "error": pl.String(),
    "response_id": pl.String(),
    "system_fingerprint": pl.String(),
    "service_tier": pl.String(),
    "provider_created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
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
    """Return the Parquet path for one run.

    The name is the run id plus the model, so a raw file and its manifest share a
    prefix and sort into the same order, and a question's files still group under
    its id.
    """
    return RAW_DIR / f"{run_id}__{_slugify(model)}.parquet"


def _ordered_columns(columns: Iterable[str]) -> list[str]:
    """Order present columns: leading common, parsed, the fixed block, created_at."""
    present = set(columns)
    known = (
        set(COMMON_LEADING_COLUMNS)
        | set(FIXED_TRAILING_COLUMNS)
        | set(TRAILING_COLUMNS)
    )
    parsed = [column for column in columns if column not in known]
    ordered = (
        COMMON_LEADING_COLUMNS + parsed + FIXED_TRAILING_COLUMNS + TRAILING_COLUMNS
    )
    return [column for column in ordered if column in present]


def write_results(rows: list[dict[str, object]], run_id: str, model: str) -> Path:
    """Write result rows to a single Parquet file and return its path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ordered = [_reorder(row) for row in rows]
    frame = pl.DataFrame(ordered, schema_overrides=_SCHEMA_OVERRIDES)
    path = results_path(run_id, model)
    frame.write_parquet(path)
    return path


def _reorder(row: dict[str, object]) -> dict[str, object]:
    """Return the row with its columns in canonical order."""
    return {column: row[column] for column in _ordered_columns(row.keys())}


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
