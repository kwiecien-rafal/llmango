"""Aggregate one question's normalized answers into the JSON the chart step reads."""

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import polars as pl

from llmango.config import AGG_DIR
from llmango.experiments import spec_for
from llmango.spec import FREE_TEXT, OTHER_CATEGORY, canonical_values
from llmango.stats import distance_from_uniform, effective_choices, normalized_entropy
from llmango.storage import normalized_path

POSITION_COLUMN = "chosen_position"


class Distribution(TypedDict):
    """One arm's answers counted over its categories, and how evenly they spread."""

    n: int
    n_invalid: int
    counts: dict[str, int]
    other_share: float
    entropy: float
    effective_choices: float
    tvd_from_uniform: float
    coverage: int


class Aggregate(TypedDict):
    """One question's committed numbers: what each arm answered, and from where."""

    question_id: str
    support: int
    distributions: dict[str, dict[str, Distribution]]
    positions: dict[str, dict[str, Distribution]]


def aggregate_question(question_id: str) -> Path:
    """Count each arm's canonical answers into data/aggregated/<question_id>.json."""
    normalized_file = normalized_path(question_id)
    if not normalized_file.is_file():
        raise FileNotFoundError(
            f"No data for question {question_id} to aggregate. "
            f"Run 'llmango normalize {question_id}' first."
        )

    frame = pl.read_parquet(normalized_file)
    support = _support(question_id, frame)
    distributions = _by_arm(frame, support)
    if not distributions:
        raise ValueError(f"No valid answers to aggregate for {question_id}.")

    return _write_aggregate(
        question_id, support, distributions, _by_position(frame, support)
    )


def _support(question_id: str, frame: pl.DataFrame) -> int:
    """How many categories an answer could have named, 'other' not among them."""
    schema = spec_for(question_id).normalization_schema
    if schema is None:
        return frame.get_column("canonical").drop_nulls().n_unique()

    return len(canonical_values(schema) - {OTHER_CATEGORY})


def _by_arm(frame: pl.DataFrame, support: int) -> dict[str, dict[str, Distribution]]:
    """Count what every arm answered, one distribution per schema and language."""
    counted = (
        frame.group_by(_arm_label(), "lang")
        .agg(
            pl.col("canonical").filter(pl.col("is_valid")),
            (~pl.col("is_valid")).sum().alias("n_invalid"),
        )
        .sort("arm", "lang")
    )

    return _nest(
        (arm, lang, _distribution(canonical, support, n_invalid))
        for arm, lang, canonical, n_invalid in counted.iter_rows()
        if canonical
    )


def _by_position(
    frame: pl.DataFrame, support: int
) -> dict[str, dict[str, Distribution]]:
    """Count where in the shown list each arm's pick sat, when a run recorded it."""
    if POSITION_COLUMN not in frame.columns:
        return {}

    counted = (
        frame.filter(pl.col("is_valid") & pl.col(POSITION_COLUMN).is_not_null())
        .group_by(_arm_label(), "lang")
        .agg(pl.col(POSITION_COLUMN).cast(pl.String))
        .sort("arm", "lang")
    )

    return _nest(
        (arm, lang, _distribution(positions, support, 0))
        for arm, lang, positions in counted.iter_rows()
        if positions
    )


def _arm_label() -> pl.Expr:
    """Name a row's arm after the title of the schema it was asked under."""
    return (
        pl.col("response_schema")
        .str.json_path_match("$.title")
        .fill_null(FREE_TEXT)
        .alias("arm")
    )


def _distribution(answers: list[str], support: int, n_invalid: int) -> Distribution:
    """Count one arm's answers, and describe how evenly they spread over support."""
    counts = dict(sorted(Counter(answers).items()))
    total = sum(counts.values())
    picked = [count for name, count in counts.items() if name != OTHER_CATEGORY]

    return {
        "n": total,
        "n_invalid": n_invalid,
        "counts": counts,
        "other_share": _rate(counts.get(OTHER_CATEGORY, 0), total),
        "entropy": normalized_entropy(picked, support),
        "effective_choices": effective_choices(picked, support),
        "tvd_from_uniform": distance_from_uniform(picked, support),
        "coverage": len(picked),
    }


def _nest(
    entries: Iterable[tuple[str, str, Distribution]],
) -> dict[str, dict[str, Distribution]]:
    """Nest arm, language and numbers into the schema-then-language shape stored."""
    nested: dict[str, dict[str, Distribution]] = {}
    for arm, lang, distribution in entries:
        nested.setdefault(arm, {})[lang] = distribution

    return nested


def _rate(part: int, whole: int) -> float:
    """Return part over whole rounded for a compact, stable file, 0.0 if empty."""
    return round(part / whole, 4) if whole else 0.0


def _write_aggregate(
    question_id: str,
    support: int,
    distributions: dict[str, dict[str, Distribution]],
    positions: dict[str, dict[str, Distribution]],
) -> Path:
    """Write one question's numbers to data/aggregated/<question_id>.json."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_file = AGG_DIR / f"{question_id}.json"
    body: Aggregate = {
        "question_id": question_id,
        "support": support,
        "distributions": distributions,
        "positions": positions,
    }

    aggregate_file.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return aggregate_file
