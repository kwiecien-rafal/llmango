"""Experiment 001's charts: what each one compares, and which questions it reads."""

from llmango.aggregate import Aggregate, Distribution
from llmango.plot import (
    ChartDef,
    Drawn,
    distribution,
    question_distribution,
    summary,
)
from llmango.stats import total_variation

_SCHEMA = "FruitChoice"
_ENGLISH = "en"
_UNIFORM = 1.0

FRUIT_EMOJI = {
    "apple": "\U0001f34e",
    "banana": "\U0001f34c",
    "orange": "\U0001f34a",
    "mango": "\U0001f96d",
    "grape": "\U0001f347",
    "strawberry": "\U0001f353",
    "watermelon": "\U0001f349",
    "pineapple": "\U0001f34d",
    "lychee": "\U0001f35a",
    "pomegranate": "\U0001f534",
}


def fruit_label(category: str) -> str:
    """Write a fruit on an axis with its emoji, or bare when Unicode has none."""
    emoji = FRUIT_EMOJI.get(category)
    if emoji is None:
        return category

    return f"{category} {emoji}"


def randomness(aggregates: dict[str, Aggregate]) -> Drawn:
    """Every arm's entropy against a fair ten-sided die: the headline number."""
    entropies = {
        _arm_name(question_id, arm, lang): cell["entropy"]
        for question_id, aggregate in sorted(aggregates.items())
        for arm, langs in sorted(aggregate["distributions"].items())
        for lang, cell in sorted(langs.items())
    }
    counts = {
        _arm_name(question_id, arm, lang): cell["n"]
        for question_id, aggregate in sorted(aggregates.items())
        for arm, langs in sorted(aggregate["distributions"].items())
        for lang, cell in sorted(langs.items())
    }

    return summary(
        cells=entropies,
        title="how much of a fair die's randomness each arm reached",
        value_label="share of uniform entropy",
        row_label="arm",
        reference=_UNIFORM,
        counts=counts,
    )


def language_drift(aggregates: dict[str, Aggregate]) -> Drawn:
    """001a's three language arms: the baseline distribution."""
    return question_distribution(aggregates["001a"], category_label=fruit_label)


def order_effect(aggregates: dict[str, Aggregate]) -> Drawn:
    """001a against 001b: one English prompt over two fixed orders of one list."""
    return distribution(
        cells={
            "001a order": _english(aggregates["001a"]),
            "001b order": _english(aggregates["001b"]),
        },
        title="001a / 001b: answer distribution by option order (en)",
        category_label=fruit_label,
    )


def position_bias(aggregates: dict[str, Aggregate]) -> Drawn:
    """001c's shuffled arms read by where the pick sat, not by which fruit it was."""
    aggregate = aggregates["001c"]
    support = aggregate["support"]
    arms = {
        lang: cell
        for _, langs in sorted(aggregate["positions"].items())
        for lang, cell in sorted(langs.items())
    }

    return distribution(
        cells=arms,
        title="001c: answer distribution by position in the shown list",
        row_label="position",
        categories=[str(place) for place in range(1, support + 1)],
        reference=1.0 / support if support else None,
    )


def schema_effect(aggregates: dict[str, Aggregate]) -> Drawn:
    """001d's three schema arms, all asked by the same Polish prompt."""
    return question_distribution(aggregates["001d"], category_label=fruit_label)


def shuffle_effect(aggregates: dict[str, Aggregate]) -> Drawn:
    """How far 001a's fixed order moved each language, measured against 001c."""
    fixed = aggregates["001a"]["distributions"][_SCHEMA]
    shuffled = aggregates["001c"]["distributions"][_SCHEMA]
    langs = sorted(set(fixed) & set(shuffled))
    categories = sorted(
        {
            name
            for lang in langs
            for name in fixed[lang]["counts"] | shuffled[lang]["counts"]
        }
    )

    return summary(
        cells={
            lang: total_variation(
                _aligned(fixed[lang], categories), _aligned(shuffled[lang], categories)
            )
            for lang in langs
        },
        title="001a / 001c: how much of the fixed order was position",
        value_label="total variation, fixed against shuffled",
        row_label="language",
        counts={lang: fixed[lang]["n"] + shuffled[lang]["n"] for lang in langs},
    )


def _english(aggregate: Aggregate) -> Distribution:
    """One question's English FruitChoice arm, the arm both fixed orders share."""
    return aggregate["distributions"][_SCHEMA][_ENGLISH]


def _aligned(cell: Distribution, categories: list[str]) -> list[int]:
    """One arm's counts over a shared category order, so two arms can be compared."""
    return [cell["counts"].get(category, 0) for category in categories]


def _arm_name(question_id: str, arm: str, lang: str) -> str:
    """Name one arm across every question, since this chart spans all of them."""
    return f"{question_id} {lang}" if arm == _SCHEMA else f"{question_id} {lang}/{arm}"


CHARTS = (
    ChartDef("randomness", ("001a", "001b", "001c", "001d"), draw=randomness),
    ChartDef("language_drift", ("001a",), draw=language_drift),
    ChartDef("order_effect", ("001a", "001b"), draw=order_effect),
    ChartDef("position_bias", ("001c",), draw=position_bias),
    ChartDef("schema_effect", ("001d",), draw=schema_effect),
    ChartDef("shuffle_effect", ("001a", "001c"), draw=shuffle_effect),
)
