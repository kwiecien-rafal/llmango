"""Aggregate normalized answers into the small JSON the chart step reads.

Reads an experiment's normalized Parquet and, per question and schema variant and
language, computes the distribution over canonical categories. Each metric is
written as a compact JSON file under data/aggregated/<experiment_id>/, nested
question -> schema_variant -> language. The share that fell into 'other' is
reported alongside the distribution as a first-class number, not hidden.

Answers that named no category, whether the call errored or the model declined,
are simply absent from the distribution. Their share is not measured here.

Experiments whose answers carry enough free text to detect drift can opt into an
output language-match rate by setting detect_language_drift on their spec. When
enabled, the metric detects the language of each answer against the set actually
present in the data; short answers that are too ambiguous to place confidently
are counted as undetermined. Single-token experiments like fruit leave it off,
since a lone fruit word is a cross-language cognate and too short to detect.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from llmango.config import AGG_DIR
from llmango.lang_detect import detect_language, primary_subtag
from llmango.registry import (
    OTHER_CATEGORY,
    ExperimentSpec,
    get_experiment,
    resolve_experiment_id,
)
from llmango.storage import normalized_path, read_normalized

DetectFn = Callable[[str, tuple[str, ...]], str | None]


@dataclass(frozen=True)
class Answer:
    """One normalized answer, reduced to the fields aggregation needs."""

    question_id: str
    schema_variant: str
    lang: str
    raw: str
    canonical: str
    is_fruit: bool


@dataclass(frozen=True)
class AggregateOutcome:
    """The aggregated JSON files one aggregation run wrote."""

    paths: list[Path]


Head = dict[str, list[Answer]]
Metric = Callable[[Head], Mapping[str, object]]


def aggregate_experiment(
    experiment_id: str,
    *,
    detect: DetectFn = detect_language,
) -> AggregateOutcome:
    """Aggregate an experiment's normalized answers into the committed JSON files.

    The detector is injectable so tests can run offline; by default it uses the
    lingua-backed detector restricted to the languages present in the data.
    """
    experiment_id = resolve_experiment_id(experiment_id)
    spec = get_experiment(experiment_id)
    if not normalized_path(experiment_id).is_file():
        raise FileNotFoundError(
            f"No normalized parquet for {experiment_id}. Run 'llmango normalize' first."
        )
    frame = read_normalized(experiment_id)
    if frame.is_empty():
        raise ValueError(f"Normalized results for {experiment_id} contain no rows.")

    heads = _group_heads(_answers(frame, spec))
    metrics: dict[str, Metric] = {
        "distributions.json": lambda head: {
            lang: _distribution(subset) for lang, subset in head.items()
        },
    }
    if spec.detect_language_drift:
        metrics["language_match.json"] = lambda head: {
            lang: _match(subset, lang, tuple(head), detect)
            for lang, subset in head.items()
        }

    paths = [
        _write_json(experiment_id, name, _nest(heads, metric))
        for name, metric in metrics.items()
    ]
    return AggregateOutcome(paths=paths)


def _answers(frame: pl.DataFrame, spec: ExperimentSpec) -> list[Answer]:
    """Reduce the normalized frame to the answer records aggregation reads."""
    columns = {
        name: frame.get_column(name).to_list()
        for name in (
            "question_id",
            "schema_variant",
            "lang",
            spec.raw_column,
            spec.canonical_column,
            "is_fruit",
        )
    }
    return [
        Answer(
            question_id=str(question_id),
            schema_variant=str(schema_variant),
            lang=str(lang),
            raw=_text(raw),
            canonical=_text(canonical),
            is_fruit=bool(fruit),
        )
        for question_id, schema_variant, lang, raw, canonical, fruit in zip(
            columns["question_id"],
            columns["schema_variant"],
            columns["lang"],
            columns[spec.raw_column],
            columns[spec.canonical_column],
            columns["is_fruit"],
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
    counts = Counter(answer.canonical for answer in answers if answer.is_fruit)
    total = counts.total()
    return {
        "n": total,
        "counts": dict(counts),
        "other_share": _rate(counts.get(OTHER_CATEGORY, 0), total),
    }


def _match(
    answers: list[Answer],
    lang: str,
    languages: tuple[str, ...],
    detect: DetectFn,
) -> dict[str, object]:
    """How one group's valid answers split across in-language, other, unsure."""
    texts = Counter(answer.raw for answer in answers if answer.is_fruit and answer.raw)
    expected = primary_subtag(lang)
    matched = 0
    undetermined = 0
    for text, count in texts.items():
        detected = detect(text, languages)
        if detected is None:
            undetermined += count
        elif detected == expected:
            matched += count
    total = texts.total()
    return {
        "total": total,
        "matched": matched,
        "undetermined": undetermined,
        "rate": _rate(matched, total - undetermined),
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
