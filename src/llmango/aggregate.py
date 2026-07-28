"""Aggregate normalized answers into the small JSON the chart step reads.

Takes a question id, reads that question's normalized Parquet and, per schema and
language, computes the distribution over canonical categories. It is written as
one compact JSON file, data/aggregated/<question_id>.json, nested schema ->
language, so the question is the file name rather than a level inside it. An arm
asked under no schema is reported under FREE_TEXT, the one name it can go by. The
share that fell into 'other' is reported alongside the distribution as a
first-class number, not hidden.

Answers that named no category, whether the call errored or the model declined,
are simply absent from the distribution. Their share is not measured here, and an
arm whose answers all named nothing carries no entry rather than an empty one.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from llmango.config import AGG_DIR
from llmango.spec import FREE_TEXT, OTHER_CATEGORY
from llmango.storage import normalized_path, read_normalized


@dataclass(frozen=True)
class AggregateOutcome:
    """The aggregated JSON file one aggregation run wrote."""

    path: Path


def aggregate_question(question_id: str) -> AggregateOutcome:
    """Aggregate one question's normalized answers into its committed JSON file.

    One grouping collects the categories each arm named, in one pass over the
    answers that named one, and the arms are nested schema over language from
    there. Rows carry the schema they were asked under, so nothing has to be
    looked up in the question's config to know which arm a row belongs to.
    """
    if not normalized_path(question_id).is_file():
        raise FileNotFoundError(
            f"No normalized parquet for {question_id}. Run 'llmango normalize' first."
        )
    frame = read_normalized(question_id)
    if frame.is_empty():
        raise ValueError(f"Normalized results for {question_id} contain no rows.")

    arms = (
        frame.filter(pl.col("is_valid") & pl.col("canonical").is_not_null())
        .group_by("schema_name", "lang")
        .agg(pl.col("canonical"))
        .sort("schema_name", "lang")
    )

    distributions: dict[str, dict[str, object]] = {}
    for schema_name, lang, canonical in arms.iter_rows():
        arm = schema_name or FREE_TEXT
        distributions.setdefault(arm, {})[lang] = _distribution(canonical)
    return AggregateOutcome(path=_write_aggregate(question_id, distributions))


def _distribution(canonical: list[str]) -> dict[str, object]:
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
    question_id: str, distributions: dict[str, dict[str, object]]
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
