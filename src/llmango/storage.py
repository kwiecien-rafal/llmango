"""Parquet storage for raw generation results.

Raw results are written one Parquet file per (question, model, run). The common
columns are fixed here per CLAUDE.md; each experiment contributes its own parsed
fields via its to_row hook, which land between raw_json and created_at.
"""

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from llmango.config import RAW_DIR

COMMON_LEADING_COLUMNS = [
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
]

TRAILING_COLUMNS = ["created_at"]

_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "question_id": pl.String(),
    "lang": pl.String(),
    "model": pl.String(),
    "backend": pl.String(),
    "run_id": pl.String(),
    "sample_idx": pl.Int64(),
    "seed": pl.Int64(),
    "temperature": pl.Float64(),
    "prompt_sha256": pl.String(),
    "raw_json": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}


def _slugify(value: str) -> str:
    """Make a model id safe to use inside a file name."""
    return value.replace("/", "-").replace("\\", "-")


def results_path(question_id: str, model: str, run_id: str) -> Path:
    """Return the Parquet path for one question, model and run."""
    return RAW_DIR / f"{question_id}__{_slugify(model)}__{run_id}.parquet"


def _ordered_columns(columns: Iterable[str]) -> list[str]:
    """Order columns as leading common, then parsed fields, then created_at."""
    present = list(columns)
    parsed = [
        column
        for column in present
        if column not in COMMON_LEADING_COLUMNS and column not in TRAILING_COLUMNS
    ]
    return COMMON_LEADING_COLUMNS + parsed + TRAILING_COLUMNS


def write_results(
    rows: list[dict[str, object]],
    question_id: str,
    model: str,
    run_id: str,
) -> Path:
    """Write result rows to a single Parquet file and return its path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ordered = [_reorder(row) for row in rows]
    frame = pl.DataFrame(ordered, schema_overrides=_SCHEMA_OVERRIDES)
    path = results_path(question_id, model, run_id)
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
