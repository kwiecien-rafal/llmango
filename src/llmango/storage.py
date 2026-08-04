"""Result storage: raw appended a call at a time, normalized written whole."""

import io
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import polars as pl

from llmango.config import get_normalized_path, get_raw_dir, get_raw_results_path


def append_result(raw_row: dict[str, object], folder: str, run_id: str) -> Path:
    """Append one result to its run's JSONL, closed so a kill keeps what came back."""
    get_raw_dir(folder).mkdir(parents=True, exist_ok=True)
    results_file = get_raw_results_path(folder, run_id)
    with results_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(raw_row, ensure_ascii=False, default=_json_default) + "\n"
        )

    return results_file


def read_results(
    folder: str, question_id: str, dtypes: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    """Pool every run of one question, back under the dtypes raw was written with."""
    results_files = sorted(get_raw_dir(folder).glob(f"{question_id}__*.jsonl"))
    if not results_files:
        return pl.DataFrame()

    frame = pl.concat(
        pl.read_ndjson(_complete_lines(results_file), schema=_text_timestamps(dtypes))
        for results_file in results_files
    )

    return frame.with_columns(
        pl.col(_timestamp_columns(dtypes)).str.to_datetime(
            time_unit="us", time_zone="UTC"
        )
    )


def write_normalized(frame: pl.DataFrame, folder: str, question_id: str) -> Path:
    """Write a question's normalized frame whole, so what it publishes is complete."""
    normalized_file = get_normalized_path(folder, question_id)
    normalized_file.parent.mkdir(parents=True, exist_ok=True)
    partial = normalized_file.with_suffix(".partial")
    frame.write_parquet(partial)
    partial.replace(normalized_file)

    return normalized_file


def _json_default(value: object) -> str:
    """Encode what JSON carries no type for, which is every timestamp column."""
    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(f"A raw row holds no {type(value).__name__}.")


def _text_timestamps(dtypes: Mapping[str, pl.DataType]) -> dict[str, pl.DataType]:
    """The declared dtypes as JSON holds them, every timestamp a string."""
    return {
        name: pl.String() if isinstance(dtype, pl.Datetime) else dtype
        for name, dtype in dtypes.items()
    }


def _timestamp_columns(dtypes: Mapping[str, pl.DataType]) -> list[str]:
    """The columns read back as text, which the declared dtypes make datetimes."""
    return [name for name, dtype in dtypes.items() if isinstance(dtype, pl.Datetime)]


def _complete_lines(results_file: Path) -> io.StringIO:
    """Drop the final line a killed run left half-written, which no reader can parse."""
    text = results_file.read_text(encoding="utf-8")

    return io.StringIO(text[: text.rfind("\n") + 1])
