"""Aggregate normalized answers into the small JSON the chart step reads.

Takes a question id, reads that question's normalized Parquet and, per schema
variant and language, computes the distribution over canonical categories. It is
written as one compact JSON file, data/aggregated/<question_id>.json, nested
schema_variant -> language, so the question is the file name rather than a level
inside it. The share that fell into 'other' is reported alongside the
distribution as a first-class number, not hidden.

Answers that named no category, whether the call errored or the model declined,
are simply absent from the distribution. Their share is not measured here.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from llmango.config import AGG_DIR
from llmango.spec import OTHER_CATEGORY
from llmango.storage import normalized_path, read_normalized


@dataclass(frozen=True)
class Answer:
    """One normalized answer, reduced to the fields aggregation needs."""

    schema_variant: str
    lang: str
    canonical: str
    is_valid: bool


@dataclass(frozen=True)
class AggregateOutcome:
    """The aggregated JSON file one aggregation run wrote."""

    path: Path


Head = dict[str, list[Answer]]
Metric = Callable[[Head], Mapping[str, object]]


def aggregate_question(question_id: str) -> AggregateOutcome:
    """Aggregate one question's normalized answers into its committed JSON file."""
    if not normalized_path(question_id).is_file():
        raise FileNotFoundError(
            f"No normalized parquet for {question_id}. Run 'llmango normalize' first."
        )
    frame = read_normalized(question_id)
    if frame.is_empty():
        raise ValueError(f"Normalized results for {question_id} contain no rows.")

    heads = _group_heads(_answers(frame))
    distributions = _nest(
        heads,
        lambda head: {lang: _distribution(subset) for lang, subset in head.items()},
    )
    return AggregateOutcome(path=_write_aggregate(question_id, distributions))


def _answers(frame: pl.DataFrame) -> list[Answer]:
    """Reduce the normalized frame to the answer records aggregation reads."""
    columns = {
        name: frame.get_column(name).to_list()
        for name in ("schema_variant", "lang", "canonical", "is_valid")
    }
    return [
        Answer(
            schema_variant=str(schema_variant),
            lang=str(lang),
            canonical=_text(canonical),
            is_valid=bool(valid),
        )
        for schema_variant, lang, canonical, valid in zip(
            columns["schema_variant"],
            columns["lang"],
            columns["canonical"],
            columns["is_valid"],
            strict=True,
        )
    ]


def _text(value: object) -> str:
    """Render a possibly-null cell as a string, treating null as empty."""
    return "" if value is None else str(value)


def _group_heads(answers: list[Answer]) -> dict[str, Head]:
    """Group answers by schema variant, then by language."""
    heads: dict[str, Head] = {}
    for answer in answers:
        head = heads.setdefault(answer.schema_variant, {})
        head.setdefault(answer.lang, []).append(answer)
    return {key: heads[key] for key in sorted(heads)}


def _nest(heads: dict[str, Head], metric: Metric) -> Mapping[str, object]:
    """Apply a metric to each head, nested schema_variant -> language."""
    return {variant: dict(metric(head)) for variant, head in heads.items()}


def _distribution(answers: list[Answer]) -> dict[str, object]:
    """Count one group's valid answers over their canonical categories."""
    counts = Counter(answer.canonical for answer in answers if answer.is_valid)
    total = counts.total()
    return {
        "n": total,
        "counts": dict(counts),
        "other_share": _rate(counts.get(OTHER_CATEGORY, 0), total),
    }


def _rate(part: int, whole: int) -> float:
    """Return part over whole rounded for a compact, stable file, 0.0 if empty."""
    return round(part / whole, 4) if whole else 0.0


def _write_aggregate(question_id: str, distributions: Mapping[str, object]) -> Path:
    """Write one question's numbers to data/aggregated/<question_id>.json."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    path = AGG_DIR / f"{question_id}.json"
    body = {"question_id": question_id, "distributions": distributions}
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
