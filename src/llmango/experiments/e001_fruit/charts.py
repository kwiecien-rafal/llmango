"""Experiment 001's charts: what each one compares, and which questions it reads."""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from llmango.aggregate import Aggregate, Distribution
from llmango.experiments.e001_fruit.palette import language_color
from llmango.plot import (
    COUNT,
    ChartDef,
    Drawn,
    Tabled,
    TableDef,
    distribution,
    estimates,
    panels,
    question_distribution,
    table,
)
from llmango.spec import FREE_TEXT, OTHER_CATEGORY
from llmango.stats import effective_choices_interval

_SCHEMA = "FruitChoice"
_SCHEMA_LANGUAGES = {_SCHEMA: "en", "WyborOwocu": "pl", "KudamonoSentaku": "ja"}
_ENGLISH = "en"
_NATIVE_SCHEMA = "native schema"
_ONE_FRUIT_ALWAYS = 1.0
_WHOLE_SHARE = 1.0
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
    return _fruit_distribution(aggregates["001a"], title, zeros_written=True)


def order_effect(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """001a against 001b: three prompts over two fixed orders of one list."""
    return _fruit_panels(
        {
            "001a order": _by_language(aggregates["001a"]),
            "001b order": _by_language(aggregates["001b"]),
        },
        title,
    )


def shuffled_choice(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """001c's shuffled arms read by which fruit was picked, not by where it sat."""
    return _fruit_distribution(aggregates["001c"], title, ceiling=_WHOLE_SHARE)


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
        series_color=language_color,
        row_label="position",
        categories=[str(place) for place in range(1, support + 1)],
        reference=1.0 / support if support else None,
        horizontal=True,
    )


def schema_effect(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """001d's eight arms: every language under the English schema, its own, and none."""
    return _fruit_panels(_by_schema(aggregates["001d"]), title)


def randomness(aggregates: dict[str, Aggregate], title: str) -> Drawn:
    """How many fruits each arm behaved as though it was choosing between."""
    arms = _named_arms(aggregates)

    return estimates(
        cells={name: cell["effective_choices"] for name, (_, _, cell) in arms.items()},
        title=title,
        value_label="number of effective fruit choices",
        row_label="arm",
        counts={name: cell["n"] for name, (_, _, cell) in arms.items()},
        intervals={
            name: effective_choices_interval(_picked(cell), support)
            for name, (_, support, cell) in arms.items()
        },
        series_color=lambda name: language_color(arms[name][0]),
        key=_language_key(arms),
        unit=COUNT,
        floor=_ONE_FRUIT_ALWAYS,
    )


def fruit_totals(aggregates: dict[str, Aggregate], title: str) -> Tabled:
    """Every arm of every question pooled: what each fruit was picked in total."""
    counted: Counter[str] = Counter()
    answered = 0
    for aggregate in aggregates.values():
        for cell in _cells(aggregate):
            counted.update(cell["counts"])
            answered += cell["n"]

    return table(
        cells={name: counted[name] for name in _every_fruit(counted)},
        total=answered,
        title=title,
        row_label="fruit",
        count_column="times picked",
        share_column="share of all answers",
        row_icon=fruit_icon,
    )


def _every_fruit(counted: Counter[str]) -> list[str]:
    """Every fruit offered, most picked first, so one never picked is still read."""
    ranked = sorted(FRUIT_EMOJI, key=lambda fruit: (-counted[fruit], fruit))
    if counted[OTHER_CATEGORY]:
        ranked.append(OTHER_CATEGORY)

    return ranked


def _fruit_distribution(
    aggregate: Aggregate,
    title: str,
    zeros_written: bool = False,
    ceiling: float | None = None,
) -> Drawn:
    """Draw one question's arms over the fruits any of them picked."""
    return question_distribution(
        aggregate,
        title,
        series_color=language_color,
        schema_label=schema_label,
        category_icon=fruit_icon,
        categories=_fruit_categories(_cells(aggregate)),
        zeros_written=zeros_written,
        ceiling=ceiling,
    )


def _fruit_panels(facets: dict[str, dict[str, Distribution]], title: str) -> Drawn:
    """Draw one panel per thing a question varies, over the fruits its arms picked."""
    return panels(
        cells=facets,
        title=title,
        series_color=language_color,
        category_icon=fruit_icon,
        categories=_fruit_categories(
            cell for panel in facets.values() for cell in panel.values()
        ),
    )


def _by_language(aggregate: Aggregate) -> dict[str, Distribution]:
    """One question's arms keyed by the language each of them was asked in."""
    return dict(sorted(aggregate["distributions"][_SCHEMA].items()))


def _by_schema(aggregate: Aggregate) -> dict[str, dict[str, Distribution]]:
    """001d's arms as one panel per schema asked: English, the prompt's own, none."""
    distributions = aggregate["distributions"]

    return {
        schema_label(_SCHEMA): dict(sorted(distributions[_SCHEMA].items())),
        _NATIVE_SCHEMA: {
            lang: distributions[schema][lang]
            for schema, lang in sorted(_SCHEMA_LANGUAGES.items(), key=_by_value)
            if lang != _ENGLISH and lang in distributions.get(schema, {})
        },
        schema_label(FREE_TEXT): dict(sorted(distributions[FREE_TEXT].items())),
    }


def _by_value(entry: tuple[str, str]) -> str:
    """Sort a schema by the language it is written in rather than by its name."""
    return entry[1]


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


def _named_arms(
    aggregates: dict[str, Aggregate],
) -> dict[str, tuple[str, int, Distribution]]:
    """Every arm of every question, named across them and kept beside its numbers."""
    return {
        _arm_name(question_id, aggregate, schema, lang): (
            lang,
            aggregate["support"],
            cell,
        )
        for question_id, aggregate in sorted(aggregates.items())
        for schema, langs in sorted(aggregate["distributions"].items())
        for lang, cell in sorted(langs.items())
    }


def _language_key(arms: dict[str, tuple[str, int, Distribution]]) -> dict[str, str]:
    """What a dot's color stands for, since the arm it names carries no swatch."""
    return {
        lang: language_color(lang)
        for lang in sorted({lang for lang, _, _ in arms.values()})
    }


def _arm_name(question_id: str, aggregate: Aggregate, schema: str, lang: str) -> str:
    """Name one arm across questions, by every dimension its question varies."""
    if len(aggregate["distributions"]) > 1:
        return f"{question_id} {lang} / {schema_label(schema)}"

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
        title="Answer distribution by option order in 001a vs 001b",
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
        "schema_effect",
        number="1.5",
        title="Answer distribution by schema in 001d",
        questions=("001d",),
        draw=schema_effect,
    ),
    ChartDef(
        "randomness",
        number="1.6",
        title=f"How many of the {_FRUITS_OFFERED} fruits each arm was choosing between",
        questions=("001a", "001b", "001c", "001d"),
        draw=randomness,
    ),
)

TABLES = (
    TableDef(
        "fruit_totals",
        number="1.1",
        title="How many times was each fruit picked",
        questions=("001a", "001b", "001c", "001d"),
        build=fruit_totals,
    ),
)
