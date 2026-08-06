"""Tests for the analyze stage: what it draws, what it skips, and what it writes."""

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

import pytest

from conftest import SUPPORT, build_distribution
from llmango.aggregate import Distribution, _write_aggregate
from llmango.analyze import AnalyzeOutcome, analyze_all
from llmango.experiments import EXPERIMENTS

_EXPERIMENT = "e001_fruit"
_SVG_ROOT = "{http://www.w3.org/2000/svg}svg"


def _analyze() -> AnalyzeOutcome:
    """Analyze everything and return the one experiment these tests are about."""
    return next(
        outcome for outcome in analyze_all() if outcome.experiment == _EXPERIMENT
    )


def _cell(counts: dict[str, int]) -> Distribution:
    return build_distribution(counts)


def _aggregate(question_id: str, langs: dict[str, Distribution]) -> None:
    """Write one question's aggregate through the writer aggregate itself uses.

    Hand-rolling the envelope here would let these tests keep passing against a
    layout nothing produces, so the real writer builds it. It resolves its path
    through config, which data_dirs redirects into tmp_path.
    """
    _write_aggregate(_EXPERIMENT, question_id, SUPPORT, {"FruitChoice": langs}, {})


def _charts(root: Path) -> Path:
    return root / "charts" / _EXPERIMENT


def _index(root: Path) -> dict[str, Any]:
    return json.loads((_charts(root) / "index.json").read_text(encoding="utf-8"))


@pytest.fixture
def baseline(data_dirs: Path) -> Path:
    """Only 001a aggregated, which is every chart 001a alone can support."""
    _aggregate("001a", {"en": _cell({"apple": 3, "banana": 1})})
    return data_dirs


@pytest.fixture
def both_orders(data_dirs: Path) -> Path:
    """001a and 001b aggregated, which is what order_effect needs to be drawn."""
    _aggregate("001a", {"en": _cell({"apple": 3, "banana": 1})})
    _aggregate("001b", {"en": _cell({"apple": 1, "banana": 3})})
    return data_dirs


@pytest.fixture
def every_question(data_dirs: Path) -> Path:
    """All four questions aggregated, which is what the pooled table reads."""
    arm = {"en": _cell({"lychee": 3, "mango": 1})}
    for question_id in ("001a", "001b", "001c"):
        _aggregate(question_id, arm)
    _write_aggregate(
        _EXPERIMENT, "001d", SUPPORT, {"FruitChoice": arm, "none": arm}, {}
    )
    return data_dirs


def test_the_index_is_keyed_by_experiment_and_chart_name(baseline: Path) -> None:
    outcome = _analyze()

    index = _index(baseline)
    assert index["experiment"] == _EXPERIMENT
    assert [(chart["name"], chart["file"]) for chart in index["charts"]] == [
        ("language_drift", "language_drift.svg"),
    ]
    assert outcome.index_path == _charts(baseline) / "index.json"


def test_a_chart_carries_the_number_it_is_cited_by_into_its_title(
    baseline: Path,
) -> None:
    """The number is the article's citation, so it reaches both the drawing and
    the table beside it rather than being written on the page by hand."""
    _analyze()

    chart = _index(baseline)["charts"][0]
    assert chart["number"] == "1.1"
    assert chart["title"] == "Chart 1.1: Answer distribution by language in 001a"


def test_a_chart_cites_the_questions_it_was_drawn_from(both_orders: Path) -> None:
    """Provenance a page can quote, now that a chart is not keyed by one question."""
    _analyze()

    charts = _index(both_orders)["charts"]
    assert {chart["name"]: chart["questions"] for chart in charts} == {
        "language_drift": ["001a"],
        "order_effect": ["001a", "001b"],
    }


def test_a_chart_whose_questions_lack_aggregates_is_skipped(baseline: Path) -> None:
    """001b and 001d have not been run, so two of the three charts cannot be drawn."""
    outcome = _analyze()

    assert [chart.name for chart in outcome.charts] == ["language_drift"]
    assert outcome.skipped == [
        "order_effect",
        "shuffled_choice",
        "position_bias",
        "schema_effect",
        "randomness",
        "fruit_totals",
    ]
    assert not (_charts(baseline) / "order_effect.svg").exists()


def test_a_chart_over_two_questions_is_drawn_once_both_are_there(
    both_orders: Path,
) -> None:
    """The comparison per-question keying could not express, drawn in one pass."""
    outcome = _analyze()

    assert [chart.name for chart in outcome.charts] == [
        "language_drift",
        "order_effect",
    ]
    assert outcome.skipped == [
        "shuffled_choice",
        "position_bias",
        "schema_effect",
        "randomness",
        "fruit_totals",
    ]
    assert (_charts(both_orders) / "order_effect.svg").is_file()

    order = next(chart for chart in outcome.charts if chart.name == "order_effect")
    assert order.questions == ["001a", "001b"]
    assert order.columns == ["en / 001a order", "en / 001b order"]


def test_a_table_reaches_the_index_with_no_drawing_beside_it(
    every_question: Path,
) -> None:
    """A table the site prints on its own is an index entry with no file to embed."""
    outcome = _analyze()

    assert [table.name for table in outcome.tables] == ["fruit_totals"]
    tables = _index(every_question)["tables"]
    assert [table["name"] for table in tables] == ["fruit_totals"]
    assert tables[0]["questions"] == ["001a", "001b", "001c", "001d"]
    assert not list(_charts(every_question).glob("fruit_totals*"))


def test_a_pictured_row_has_its_picture_written_out_beside_the_charts(
    every_question: Path,
) -> None:
    """The site fetches the picture, so the file has to be somewhere it serves."""
    _analyze()

    rows = _index(every_question)["tables"][0]["rows"]
    lychee = next(row for row in rows if row["label"] == "lychee")
    assert lychee["icon"] == "icons/emoji_u1f330.png"
    assert (_charts(every_question) / "icons" / "emoji_u1f330.png").is_file()


def test_a_table_carries_the_number_it_is_cited_by_into_its_title(
    every_question: Path,
) -> None:
    """A table is cited from the page the way a chart is, and numbered apart."""
    _analyze()

    tabled = _index(every_question)["tables"][0]
    assert tabled["number"] == "1.1"
    assert tabled["title"] == "Table 1.1: How many times was each fruit picked"


def test_a_table_pools_every_arm_of_every_question_it_reads(
    every_question: Path,
) -> None:
    """Five arms over four questions, and no interval around what they add up to."""
    _analyze()

    rows = _index(every_question)["tables"][0]["rows"]
    lychee = next(row for row in rows if row["label"] == "lychee")
    assert lychee["cells"] == [
        {"value": 15, "count": 15, "n": 20, "written": "15"},
        {"value": 0.75, "count": 15, "n": 20, "written": "75.0%"},
    ]


def test_every_experiment_is_analyzed_in_one_pass(both_orders: Path) -> None:
    """analyze takes no id: what it draws is every experiment there is."""
    outcomes = analyze_all()

    assert [outcome.experiment for outcome in outcomes] == [
        experiment.folder for experiment in EXPERIMENTS
    ]


def test_every_chart_is_written_as_a_transparent_svg(baseline: Path) -> None:
    _analyze()

    path = _charts(baseline) / "language_drift.svg"
    assert ElementTree.parse(path).getroot().tag == _SVG_ROOT
    assert "dc:date" not in path.read_text(encoding="utf-8")


def test_redrawing_unchanged_aggregates_rewrites_an_identical_file(
    baseline: Path,
) -> None:
    """A rerun that churned bytes would put a meaningless diff in every commit."""
    _analyze()
    first = (_charts(baseline) / "language_drift.svg").read_bytes()

    _analyze()

    assert (_charts(baseline) / "language_drift.svg").read_bytes() == first


def test_missing_aggregates_point_at_the_aggregate_command(data_dirs: Path) -> None:
    with pytest.raises(FileNotFoundError, match="llmango aggregate"):
        analyze_all()


def test_an_experiment_with_no_aggregates_leaves_its_committed_index_alone(
    data_dirs: Path,
) -> None:
    """Writing an empty index would clobber the committed file the site reads."""
    index = _charts(data_dirs) / "index.json"
    index.parent.mkdir(parents=True)
    index.write_text("committed", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        analyze_all()

    assert index.read_text(encoding="utf-8") == "committed"
