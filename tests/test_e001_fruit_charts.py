"""Tests for experiment 001's declared charts: what each one names and reads."""

import io
import subprocess
import sys

from matplotlib.ft2font import FT2Font

from conftest import SUPPORT, build_distribution
from llmango.aggregate import Aggregate, Distribution
from llmango.experiments.e001_fruit.charts import (
    CHARTS,
    FRUIT_EMOJI,
    fruit_label,
    language_drift,
    order_effect,
)
from llmango.experiments.e001_fruit.experiment import FruitEnum
from llmango.plot import FONTS_DIR, styled


def _aggregate(question_id: str, langs: dict[str, Distribution]) -> Aggregate:
    return {
        "question_id": question_id,
        "support": SUPPORT,
        "distributions": {"FruitChoice": langs},
        "positions": {},
    }


def _cell(counts: dict[str, int]) -> Distribution:
    return build_distribution(counts)


def test_every_chart_is_named_once_and_declares_what_it_reads() -> None:
    declared = {chart.name: chart.questions for chart in CHARTS}

    assert declared == {
        "randomness": ("001a", "001b", "001c", "001d"),
        "language_drift": ("001a",),
        "order_effect": ("001a", "001b"),
        "position_bias": ("001c",),
        "schema_effect": ("001d",),
        "shuffle_effect": ("001a", "001c"),
    }
    assert len(declared) == len(CHARTS)


def test_the_order_comparison_reads_one_language_from_two_questions() -> None:
    """001b exists only to be read against 001a, so the chart labels by question."""
    with styled():
        drawn = order_effect(
            {
                "001a": _aggregate("001a", {"en": _cell({"apple": 3, "banana": 1})}),
                "001b": _aggregate("001b", {"en": _cell({"apple": 1, "banana": 3})}),
            }
        )

    assert drawn.columns == ["001a order", "001b order"]
    assert "option order" in drawn.title


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


def test_a_fruit_is_written_with_its_emoji_beside_the_word() -> None:
    """The word carries the meaning and the emoji is the scanning aid, which is
    why lychee and pomegranate can take a stand-in glyph without misnaming data."""
    assert fruit_label("apple") == "apple \U0001f34e"
    assert fruit_label("lychee") == "lychee \U0001f35a"
    assert fruit_label("nectarine") == "nectarine"


def test_the_vendored_font_carries_every_emoji_an_axis_asks_for() -> None:
    """Segoe UI Emoji is Windows-only and proprietary; leaning on a system font
    would draw tofu boxes wherever the charts were regenerated next."""
    font_file = FONTS_DIR / "NotoEmoji-Regular.ttf"
    face = FT2Font(str(font_file))

    assert font_file.is_file()
    for fruit, emoji in FRUIT_EMOJI.items():
        assert face.get_char_index(ord(emoji)) != 0, fruit


def test_a_drawn_axis_reaches_the_emoji_font_rather_than_a_tofu_box() -> None:
    aggregates = {
        "001a": _aggregate("001a", {"en": _cell({"apple": 3, "lychee": 1})}),
    }
    buffer = io.BytesIO()
    with styled():
        drawn = language_drift(aggregates)
        drawn.figure.savefig(buffer, format="svg", transparent=True)

    body = buffer.getvalue().decode("utf-8")
    assert "NotoEmoji" in body
    assert "LastResort" not in body
