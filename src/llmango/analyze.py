"""Aggregate normalized answers into the small JSON the site reads.

Reads a question's normalized Parquet and, per language, computes the
distribution over canonical categories, the refusal rate, and the output
language-match rate that measures drift away from the language that was asked.
Each metric is written as a compact JSON file under
data/aggregated/<question_id>/. The share that fell into 'other' is reported
alongside the distribution as a first-class number, not hidden.

The language-match metric detects the language of each answer against the set
actually present in the data. Short answers are often too ambiguous to place
confidently, so those are counted as undetermined and reported alongside the
match rate rather than forced into a match or a mismatch.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from llmango.config import AGG_DIR
from llmango.lang_detect import detect_language, primary_subtag
from llmango.registry import ExperimentSpec, get_experiment
from llmango.storage import normalized_path, read_normalized

_OTHER = "other"

DetectFn = Callable[[str, tuple[str, ...]], str | None]


@dataclass(frozen=True)
class Answer:
    """One normalized answer, reduced to the fields aggregation needs."""

    lang: str
    raw: str
    canonical: str
    is_fruit: bool


@dataclass(frozen=True)
class AnalyzeOutcome:
    """The aggregated JSON files one analysis run wrote."""

    paths: list[Path]


def analyze_question(
    question_id: str,
    *,
    detect: DetectFn = detect_language,
) -> AnalyzeOutcome:
    """Aggregate a question's normalized answers into the committed JSON files.

    The detector is injectable so tests can run offline; by default it uses the
    lingua-backed detector restricted to the languages present in the data.
    """
    from llmango.experiments import ensure_registered

    ensure_registered()
    spec = get_experiment(question_id)
    if not normalized_path(question_id).is_file():
        raise FileNotFoundError(
            f"No normalized parquet for {question_id}. Run 'llmango normalize' first."
        )
    frame = read_normalized(question_id)
    if frame.is_empty():
        raise ValueError(f"Normalized results for {question_id} contain no rows.")

    grouped = _by_language(_answers(frame, spec))
    languages = tuple(grouped)
    metrics = {
        "distributions.json": {
            lang: _distribution(subset) for lang, subset in grouped.items()
        },
        "refusal_rate.json": {
            lang: _refusal(subset) for lang, subset in grouped.items()
        },
        "language_match.json": {
            lang: _match(subset, lang, languages, detect)
            for lang, subset in grouped.items()
        },
    }
    paths = [
        _write_json(question_id, name, per_language)
        for name, per_language in metrics.items()
    ]
    return AnalyzeOutcome(paths=paths)


def _answers(frame: pl.DataFrame, spec: ExperimentSpec) -> list[Answer]:
    """Reduce the normalized frame to the answer records aggregation reads."""
    langs = frame.get_column("lang").to_list()
    raws = frame.get_column(spec.raw_column).to_list()
    canonicals = frame.get_column(spec.canonical_column).to_list()
    is_fruit = frame.get_column("is_fruit").to_list()
    return [
        Answer(str(lang), _text(raw), _text(canonical), bool(fruit))
        for lang, raw, canonical, fruit in zip(
            langs, raws, canonicals, is_fruit, strict=True
        )
    ]


def _text(value: object) -> str:
    """Render a possibly-null cell as a string, treating null as empty."""
    return "" if value is None else str(value)


def _by_language(answers: list[Answer]) -> dict[str, list[Answer]]:
    """Group answers by language, ordered for a stable file."""
    groups: dict[str, list[Answer]] = {}
    for answer in answers:
        groups.setdefault(answer.lang, []).append(answer)
    return {lang: groups[lang] for lang in sorted(groups)}


def _distribution(answers: list[Answer]) -> dict[str, object]:
    """Count one language's valid answers over their canonical categories."""
    counts = Counter(answer.canonical for answer in answers if answer.is_fruit)
    total = counts.total()
    return {
        "n": total,
        "counts": dict(counts),
        "other_share": _rate(counts.get(_OTHER, 0), total),
    }


def _refusal(answers: list[Answer]) -> dict[str, object]:
    """The share of one language's answers that were refusals or non-answers."""
    refusals = sum(1 for answer in answers if not answer.is_fruit)
    return {
        "total": len(answers),
        "refusals": refusals,
        "rate": _rate(refusals, len(answers)),
    }


def _match(
    answers: list[Answer],
    lang: str,
    languages: tuple[str, ...],
    detect: DetectFn,
) -> dict[str, object]:
    """How one language's valid answers split across in-language, other, unsure."""
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


def _write_json(
    question_id: str, name: str, per_language: Mapping[str, object]
) -> Path:
    """Write one metric to data/aggregated/<question_id>/<name> and return it."""
    directory = AGG_DIR / question_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    payload = {"question_id": question_id, "languages": per_language}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
