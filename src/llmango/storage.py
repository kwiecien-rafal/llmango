"""Parquet storage: where a run's results land, and how they are read back."""

from collections.abc import Mapping
from pathlib import Path

import polars as pl

from llmango.config import NORMALIZED_DIR, RAW_DIR


def _slugify(value: str) -> str:
    """Make a model id safe to use inside a file name."""
    return value.replace("/", "-").replace("\\", "-")


def results_path(run_id: str, model: str) -> Path:
    """Return the Parquet path for one run."""
    return RAW_DIR / f"{run_id}__{_slugify(model)}.parquet"


def write_results(
    rows: list[dict[str, object]],
    run_id: str,
    model: str,
    dtypes: Mapping[str, pl.DataType],
) -> Path:
    """Write one run's rows to a single Parquet file under the declared dtypes."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = results_path(run_id, model)
    pl.DataFrame(rows, schema_overrides=dtypes).write_parquet(path)
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
