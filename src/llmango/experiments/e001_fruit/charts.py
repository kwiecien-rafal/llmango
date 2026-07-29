"""Experiment 001's charts: what each one compares, and which questions it reads.

Each chart is named once here and referenced by that name from the site, so a
page decides where a figure sits and never what it shows.
"""

from llmango.aggregate import Aggregate, Distribution
from llmango.plot import ChartDef, Drawn, distribution, question_distribution

_SCHEMA = "FruitChoice"
_ENGLISH = "en"


def language_drift(aggregates: dict[str, Aggregate]) -> Drawn:
    """001a's three language arms: the baseline distribution."""
    return question_distribution(aggregates["001a"])


def order_effect(aggregates: dict[str, Aggregate]) -> Drawn:
    """001a against 001b: one English prompt over two fixed orders of one list.

    The pair is the whole point of 001b, so option position is read here as the
    only difference between two arms that are otherwise the same question.
    """
    return distribution(
        cells={
            "001a order": _english(aggregates["001a"]),
            "001b order": _english(aggregates["001b"]),
        },
        title="001a / 001b: answer distribution by option order (en)",
    )


def schema_effect(aggregates: dict[str, Aggregate]) -> Drawn:
    """001d's three schema arms, all asked by the same Polish prompt."""
    return question_distribution(aggregates["001d"])


def _english(aggregate: Aggregate) -> Distribution:
    """One question's English FruitChoice arm, the arm both fixed orders share."""
    return aggregate["distributions"][_SCHEMA][_ENGLISH]


CHARTS = (
    ChartDef("language_drift", ("001a",), draw=language_drift),
    ChartDef("order_effect", ("001a", "001b"), draw=order_effect),
    ChartDef("schema_effect", ("001d",), draw=schema_effect),
)
