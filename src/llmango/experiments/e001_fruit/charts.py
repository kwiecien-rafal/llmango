"""Experiment 001's charts: what each one compares, and which questions it reads."""

from collections.abc import Iterable
from pathlib import Path

from llmango.aggregate import Aggregate, Distribution
from llmango.plot import (
    COUNT,
    ChartDef,
    Drawn,
    distribution,
    estimates,
    question_distribution,
    summary,
)
from llmango.spec import FREE_TEXT, OTHER_CATEGORY
from llmango.stats import (
    effective_choices_interval,
    total_variation,
    total_variation_interval,
)

_SCHEMA = "FruitChoice"
_SCHEMA_LANGUAGES = {_SCHEMA: "en", "WyborOwocu": "pl"}
_ENGLISH = "en"
_ONE_FRUIT_ALWAYS = 1.0
_EMOJI_DIR = Path(__file__).parent / "emoji"

FRUIT_EMOJI = {
    "apple": "\U0001f34e",
    "banana": "\U0001f34c",
    "orange": "\U0001f34a",
    "mango": "\U0001f96d",
    "grape": "\U0001f347",
    "strawberry": "\U0001f353",
    "watermelon": "\U0001f349",
    "pineapple": "\U0001f34d",
    "lychee": "\U0001f330",
    "pomegranate": "\U0001f534",
}
_FRUITS_OFFERED = len(FRUIT_EMOJI)


def fruit_icon(category: str) -> Path | None:
    """Find the picture a fruit is drawn with, or nothing when it is not one."""
    emoji = FRUIT_EMOJI.get(category)
    if emoji is None:
        return None

    return _EMOJI_DIR / f"emoji_u{ord(emoji):x}.png"


def schema_label(schema: str) -> str:
    """Name a schema arm by the language it is written in, which is what 001d varies."""
    if schema == FREE_TEXT:
        return "no schema"

    return f"{_SCHEMA_LANGUAGES[schema]} schema"


def language_drift(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """001a's three language arms: the baseline distribution."""
    return _fruit_distribution(aggregates["001a"], title)


def order_effect(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """001a against 001b: one English prompt over two fixed orders of one list."""
    cells = {
        "001a order": _english(aggregates["001a"]),
        "001b order": _english(aggregates["001b"]),
    }

    return distribution(
        cells=cells,
        title=title,
        category_icon=fruit_icon,
        categories=_fruit_categories(cells.values()),
    )


def shuffled_choice(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """001c's shuffled arms read by which fruit was picked, not by where it sat."""
    return _fruit_distribution(aggregates["001c"], title)


def position_bias(aggregates: dict[str, Aggregate], title: str) -> Drawn:
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
        title=title,
        row_label="position",
        categories=[str(place) for place in range(1, support + 1)],
        reference=1.0 / support if support else None,
        horizontal=True,
    )


def shuffle_effect(aggregates: dict[str, Aggregate], title: str) -> Drawn:
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
    pairs = {
        lang: (_aligned(fixed[lang], categories), _aligned(shuffled[lang], categories))
        for lang in langs
    }

    return summary(
        cells={lang: total_variation(*pair) for lang, pair in pairs.items()},
        title=title,
        value_label="share of answers that moved",
        row_label="language",
        counts={lang: fixed[lang]["n"] + shuffled[lang]["n"] for lang in langs},
        intervals={
            lang: total_variation_interval(*pair) for lang, pair in pairs.items()
        },
    )


def schema_effect(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """001d's three schema arms, all asked by the same Polish prompt."""
    return _fruit_distribution(aggregates["001d"], title)


def randomness(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """How many fruits each arm behaved as though it was choosing between."""
    arms = {
        _arm_name(question_id, aggregate, schema, lang): (aggregate["support"], cell)
        for question_id, aggregate in sorted(aggregates.items())
        for schema, langs in sorted(aggregate["distributions"].items())
        for lang, cell in sorted(langs.items())
    }

    return estimates(
        cells={name: cell["effective_choices"] for name, (_, cell) in arms.items()},
        title=title,
        value_label=(
            f"effective choices (1 = one fruit always, of {_FRUITS_OFFERED} offered)"
        ),
        row_label="arm",
        counts={name: cell["n"] for name, (_, cell) in arms.items()},
        intervals={
            name: effective_choices_interval(_picked(cell), support)
            for name, (support, cell) in arms.items()
        },
        unit=COUNT,
        floor=_ONE_FRUIT_ALWAYS,
    )


def _fruit_distribution(aggregate: Aggregate, title: str) -> Drawn:
    """Draw one question's arms over the fruits any of them picked."""
    return question_distribution(
        aggregate,
        title,
        schema_label=schema_label,
        category_icon=fruit_icon,
        categories=_fruit_categories(_cells(aggregate)),
    )


def _fruit_categories(cells: Iterable[Distribution]) -> list[str]:
    """The fruits some arm picked, in the canonical order every chart shares."""
    picked = {name for cell in cells for name, count in cell["counts"].items() if count}
    shown = [fruit for fruit in FRUIT_EMOJI if fruit in picked]
    if OTHER_CATEGORY in picked:
        shown.append(OTHER_CATEGORY)

    return shown


def _cells(aggregate: Aggregate) -> list[Distribution]:
    """Every arm of one question, for a decision that spans all of them."""
    return [
        cell for langs in aggregate["distributions"].values() for cell in langs.values()
    ]


def _picked(cell: Distribution) -> list[int]:
    """One arm's counts over the canonical fruits, which is what its entropy reads."""
    return [count for name, count in cell["counts"].items() if name != OTHER_CATEGORY]


def _english(aggregate: Aggregate) -> Distribution:
    """One question's English FruitChoice arm, the arm both fixed orders share."""
    return aggregate["distributions"][_SCHEMA][_ENGLISH]


def _aligned(cell: Distribution, categories: list[str]) -> list[int]:
    """One arm's counts over a shared category order, so two arms can be compared."""
    return [cell["counts"].get(category, 0) for category in categories]


def _arm_name(question_id: str, aggregate: Aggregate, schema: str, lang: str) -> str:
    """Name one arm across questions, by whichever of schema and language varies."""
    if len(aggregate["distributions"]) > 1:
        return f"{question_id} {schema_label(schema)}"

    return f"{question_id} {lang}"


CHARTS = (
    ChartDef(
        "language_drift",
        number="1.1",
        title="Answer distribution by language in 001a",
        questions=("001a",),
        draw=language_drift,
    ),
    ChartDef(
        "order_effect",
        number="1.2",
        title="Answer distribution by option order in 001b vs 001a",
        questions=("001a", "001b"),
        draw=order_effect,
    ),
    ChartDef(
        "shuffled_choice",
        number="1.3",
        title="Answer distribution by language in 001c",
        questions=("001c",),
        draw=shuffled_choice,
    ),
    ChartDef(
        "position_bias",
        number="1.4",
        title="Answer distribution by position in 001c's shown list",
        questions=("001c",),
        draw=position_bias,
    ),
    ChartDef(
        "shuffle_effect",
        number="1.5",
        title="How much of the fixed order was position in 001a vs 001c",
        questions=("001a", "001c"),
        draw=shuffle_effect,
    ),
    ChartDef(
        "schema_effect",
        number="1.6",
        title="Answer distribution by schema in 001d",
        questions=("001d",),
        draw=schema_effect,
    ),
    ChartDef(
        "randomness",
        number="1.7",
        title=f"How many of the {_FRUITS_OFFERED} fruits each arm was choosing between",
        questions=("001a", "001b", "001c", "001d"),
        draw=randomness,
    ),
)
