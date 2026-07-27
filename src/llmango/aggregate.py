"""Aggregate normalized answers into the small JSON the chart step reads.

Reads an experiment's normalized Parquet and, per question and schema variant and
language, computes the distribution over canonical categories. It is written as a
compact JSON file under data/aggregated/<experiment_id>/, nested question ->
schema_variant -> language. The share that fell into 'other' is reported
alongside the distribution as a first-class number, not hidden.

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
from llmango.registry import OTHER_CATEGORY, resolve_experiment_id
from llmango.storage import normalized_path, read_normalized


@dataclass(frozen=True)
class Answer:
    """One normalized answer, reduced to the fields aggregation needs."""

    question_id: str
    schema_variant: str
    lang: str
    canonical: str
    is_valid: bool


@dataclass(frozen=True)
class AggregateOutcome:
    """The aggregated JSON files one aggregation run wrote."""

    paths: list[Path]


Head = dict[str, list[Answer]]
Metric = Callable[[Head], Mapping[str, object]]


def aggregate_experiment(experiment_id: str) -> AggregateOutcome:
    """Aggregate an experiment's normalized answers into the committed JSON files."""
    experiment_id = resolve_experiment_id(experiment_id)
    if not normalized_path(experiment_id).is_file():
        raise FileNotFoundError(
            f"No normalized parquet for {experiment_id}. Run 'llmango normalize' first."
        )
    frame = read_normalized(experiment_id)
    if frame.is_empty():
        raise ValueError(f"Normalized results for {experiment_id} contain no rows.")

    heads = _group_heads(_answers(frame))
    distributions = _nest(
        heads,
        lambda head: {lang: _distribution(subset) for lang, subset in head.items()},
    )
    return AggregateOutcome(
        paths=[_write_json(experiment_id, "distributions.json", distributions)]
    )


def _answers(frame: pl.DataFrame) -> list[Answer]:
    """Reduce the normalized frame to the answer records aggregation reads."""
    columns = {
        name: frame.get_column(name).to_list()
        for name in (
            "question_id",
            "schema_variant",
            "lang",
            "canonical",
            "is_valid",
        )
    }
    return [
        Answer(
            question_id=str(question_id),
            schema_variant=str(schema_variant),
            lang=str(lang),
            canonical=_text(canonical),
            is_valid=bool(valid),
        )
        for question_id, schema_variant, lang, canonical, valid in zip(
            columns["question_id"],
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


def _group_heads(answers: list[Answer]) -> dict[tuple[str, str], Head]:
    """Group answers by (question_id, schema_variant), then by language."""
    heads: dict[tuple[str, str], Head] = {}
    for answer in answers:
        head = heads.setdefault((answer.question_id, answer.schema_variant), {})
        head.setdefault(answer.lang, []).append(answer)
    return {key: heads[key] for key in sorted(heads)}


def _nest(heads: dict[tuple[str, str], Head], metric: Metric) -> Mapping[str, object]:
    """Apply a metric to each head, nested question -> schema_variant -> language."""
    nested: dict[str, dict[str, object]] = {}
    for (question_id, schema_variant), head in heads.items():
        nested.setdefault(question_id, {})[schema_variant] = dict(metric(head))
    return nested


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


def _write_json(experiment_id: str, name: str, payload: Mapping[str, object]) -> Path:
    """Write one metric to data/aggregated/<experiment_id>/<name> and return it."""
    directory = AGG_DIR / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    body = {"experiment_id": experiment_id, "questions": payload}
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
