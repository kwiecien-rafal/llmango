"""Tests for experiment 001's declared charts: what each one names and reads."""

import io
import subprocess
import sys

from conftest import SUPPORT, build_distribution
from llmango.aggregate import Aggregate, Distribution
from llmango.experiments.e001_fruit.charts import (
    CHARTS,
    FRUIT_EMOJI,
    fruit_icon,
    language_drift,
    order_effect,
    randomness,
    schema_effect,
)
from llmango.experiments.e001_fruit.experiment import FruitEnum
from llmango.plot import styled

_TITLE = "Chart 1.1: a title the chart is drawn under"


def _aggregate(question_id: str, langs: dict[str, Distribution]) -> Aggregate:
    return {
        "question_id": question_id,
        "support": SUPPORT,
        "distributions": {"FruitChoice": langs},
        "positions": {},
    }


def _cell(counts: dict[str, int]) -> Distribution:
    return build_distribution(counts)


def _schemas() -> dict[str, Aggregate]:
    """001d as three schema arms of one Polish prompt, which is all it varies."""
    return {
        "001d": {
            "question_id": "001d",
            "support": SUPPORT,
            "distributions": {
                "FruitChoice": {"pl": _cell({"apple": 2})},
                "WyborOwocu": {"pl": _cell({"apple": 1, "banana": 1})},
                "none": {"pl": _cell({"banana": 2})},
            },
            "positions": {},
        }
    }


def test_every_chart_is_named_once_and_declares_what_it_reads() -> None:
    """The tuple order is index.json's order and the order the article reads in."""
    declared = [(chart.name, chart.number, chart.questions) for chart in CHARTS]

    assert declared == [
        ("language_drift", "1.1", ("001a",)),
        ("order_effect", "1.2", ("001a", "001b")),
        ("shuffled_choice", "1.3", ("001c",)),
        ("position_bias", "1.4", ("001c",)),
        ("shuffle_effect", "1.5", ("001a", "001c")),
        ("schema_effect", "1.6", ("001d",)),
        ("randomness", "1.7", ("001a", "001b", "001c", "001d")),
    ]
    assert len({name for name, _, _ in declared}) == len(CHARTS)
    assert len({number for _, number, _ in declared}) == len(CHARTS)


def test_a_chart_opens_its_title_with_the_number_a_page_cites_it_by() -> None:
    """A number is how the article refers to a chart from anywhere but beside it,
    so it is drawn into the figure rather than written around it on the page."""
    numbered = [chart.numbered_title() for chart in CHARTS]

    assert numbered[:2] == [
        "Chart 1.1: Answer distribution by language in 001a",
        "Chart 1.2: Answer distribution by option order in 001b vs 001a",
    ]


def test_a_fruit_chart_draws_only_the_fruits_some_arm_picked() -> None:
    """Six of the ten were never picked, and an empty slot is not a finding."""
    with styled():
        drawn = language_drift(
            {
                "001a": _aggregate(
                    "001a",
                    {
                        "en": _cell({"lychee": 4}),
                        "pl": _cell({"lychee": 3, "banana": 1}),
                    },
                )
            },
            _TITLE,
        )

    assert [row["label"] for row in drawn.rows] == ["banana", "lychee"]


def test_a_fruit_chart_keeps_the_canonical_order_the_axis_shares() -> None:
    """Sorting the survivors by frequency would give every chart its own axis."""
    with styled():
        drawn = language_drift(
            {"001a": _aggregate("001a", {"en": _cell({"lychee": 9, "apple": 1})})},
            _TITLE,
        )

    assert [row["label"] for row in drawn.rows] == ["apple", "lychee"]


def test_the_order_comparison_reads_one_language_from_two_questions() -> None:
    """001b exists only to be read against 001a, so the chart labels by question."""
    with styled():
        drawn = order_effect(
            {
                "001a": _aggregate("001a", {"en": _cell({"apple": 3, "banana": 1})}),
                "001b": _aggregate("001b", {"en": _cell({"apple": 1, "banana": 3})}),
            },
            _TITLE,
        )

    assert drawn.columns == ["001a order", "001b order"]


def test_the_randomness_chart_counts_options_rather_than_shares_of_entropy() -> None:
    """A share of entropy reads as twice the randomness a count of options does."""
    with styled():
        drawn = randomness(
            {"001a": _aggregate("001a", {"en": _cell({"lychee": 5, "apple": 5})})},
            _TITLE,
        )

    assert drawn.unit == "count"
    assert drawn.columns == ["effective choices (1 = one fruit always, of 10 offered)"]
    assert drawn.rows[0]["cells"][0]["value"] > 2.0


def test_the_randomness_chart_starts_where_its_statistic_does() -> None:
    """Effective choices is floored at 1, not 0: an arm that always answers the
    same fruit scores exactly 1. A column from zero would spend a third of every
    bar on ink that cannot vary, and read total determinism as a real quantity."""
    with styled():
        drawn = randomness(
            {"001a": _aggregate("001a", {"en": _cell({"lychee": 300})})}, _TITLE
        )

    assert drawn.rows[0]["cells"][0]["value"] == 1.0
    assert drawn.figure.axes[0].get_xlim()[0] == 1.0


def test_the_randomness_chart_names_each_arm_by_what_its_question_varies() -> None:
    """001d asks one language under three schemas, so its rows read by schema and
    the language they share is noise. Every other question is the other way round."""
    with styled():
        drawn = randomness(
            {"001a": _aggregate("001a", {"en": _cell({"apple": 2})})} | _schemas(),
            _TITLE,
        )

    assert [row["label"] for row in drawn.rows] == [
        "001a en",
        "001d en schema",
        "001d pl schema",
        "001d no schema",
    ]


def test_a_schema_arm_reads_by_the_language_its_schema_is_written_in() -> None:
    """FruitChoice and WyborOwocu are what the code calls 001d's two schemas, and
    a reader comparing them is comparing the language each one is written in."""
    with styled():
        drawn = schema_effect(_schemas(), _TITLE)

    assert drawn.columns == ["en schema", "pl schema", "no schema"]


def test_the_experiment_package_does_not_pull_in_matplotlib() -> None:
    """Importing an experiment must not cost every command matplotlib's import.

    Run in a subprocess because the test session has already imported it. The
    charts module is where matplotlib is reached, and analyze imports that lazily.
    """
    probe = (
        "import sys; import llmango.experiments.e001_fruit; "
        "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_every_canonical_fruit_is_written_with_an_emoji() -> None:
    """A fruit added to the list without one would silently draw a bare word
    beside nine decorated ones, which reads as a rendering failure."""
    canonical = {value for value in FruitEnum if value != FruitEnum.OTHER}

    assert set(FRUIT_EMOJI) == {str(fruit) for fruit in canonical}


def test_only_a_canonical_fruit_is_pictured() -> None:
    assert fruit_icon("nectarine") is None


def test_every_pictured_fruit_has_its_image_vendored() -> None:
    """A system emoji font would draw a different picture on every machine."""
    for fruit in FRUIT_EMOJI:
        icon = fruit_icon(fruit)
        assert icon is not None and icon.is_file(), fruit


def test_a_drawn_axis_carries_its_pictures_inside_the_svg() -> None:
    """A page must not fetch anything to see them, and must see the same ones."""
    aggregates = {
        "001a": _aggregate("001a", {"en": _cell({"apple": 3, "lychee": 1})}),
    }
    buffer = io.BytesIO()
    with styled():
        drawn = language_drift(aggregates, _TITLE)
        drawn.figure.savefig(buffer, format="svg", transparent=True)

    body = buffer.getvalue().decode("utf-8")
    assert body.count("data:image/png;base64,") == 2
