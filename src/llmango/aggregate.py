"""Aggregate one question's normalized answers into the JSON the chart step reads."""

import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

import polars as pl

from llmango.config import AGG_DIR
from llmango.spec import FREE_TEXT, OTHER_CATEGORY
from llmango.storage import normalized_path


class Distribution(TypedDict):
    """One arm's answers counted over the canonical categories they named."""

    n: int
    counts: dict[str, int]
    other_share: float


def aggregate_question(question_id: str) -> Path:
    """Count each arm's canonical answers into data/aggregated/<question_id>.json."""
    path = normalized_path(question_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"No normalized data for question {question_id} to aggregate from. "
        )
    arms = (
        pl.read_parquet(path)
        .filter(pl.col("is_valid"))
        .group_by(_arm_label(), "lang")
        .agg(pl.col("canonical"))
    )
    if arms.is_empty():
        raise ValueError(f"No valid answers to aggregate for {question_id}.")

    distributions: dict[str, dict[str, Distribution]] = {}
    for arm, lang, canonical in arms.iter_rows():
        distributions.setdefault(arm, {})[lang] = _distribution(canonical)
    return _write_aggregate(question_id, distributions)


def _arm_label() -> pl.Expr:
    """Name a row's arm after the title of the schema it was asked under."""
    return (
        pl.col("response_schema")
        .str.json_path_match("$.title")
        .fill_null(FREE_TEXT)
        .alias("arm")
    )


def _distribution(canonical: list[str]) -> Distribution:
    """Count one arm's answers over the canonical categories they named."""
    counts = Counter(canonical)
    total = counts.total()
    return {
        "n": total,
        "counts": dict(counts),
        "other_share": _rate(counts.get(OTHER_CATEGORY, 0), total),
    }


def _rate(part: int, whole: int) -> float:
    """Return part over whole rounded for a compact, stable file, 0.0 if empty."""
    return round(part / whole, 4) if whole else 0.0


def _write_aggregate(
    question_id: str, distributions: dict[str, dict[str, Distribution]]
) -> Path:
    """Write one question's numbers to data/aggregated/<question_id>.json."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    path = AGG_DIR / f"{question_id}.json"
    body = {"question_id": question_id, "distributions": distributions}
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
