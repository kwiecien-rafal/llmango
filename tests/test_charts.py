"""Tests for the drawn charts: arm labels, shares, ordering, index and output."""

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

import pytest
from matplotlib.figure import Figure

from llmango.aggregate import _write_aggregate
from llmango.charts import (
    _distribution_figure,
    _distribution_title,
    _labels,
    _load,
    _peak_labels,
    analyze_question,
)

_SVG_ROOT = "{http://www.w3.org/2000/svg}svg"


def _cell(counts: dict[str, int]) -> dict[str, object]:
    total = sum(counts.values())
    return {
        "n": total,
        "counts": counts,
        "other_share": round(counts.get("other", 0) / total, 4) if total else 0.0,
    }


def _write_distributions(
    question_id: str, distributions: dict[str, dict[str, object]]
) -> None:
    """Write a fixture through the writer aggregate itself uses.

    Hand-rolling the envelope here would let these tests keep passing against a
    layout nothing produces, so the real writer builds it. It resolves AGG_DIR
    through the aggregate module, which data_dirs redirects into tmp_path.
    """
    _write_aggregate(question_id, distributions)


def _charts(root: Path, question_id: str) -> Path:
    return root / "charts" / question_id


def _index(root: Path, question_id: str) -> dict[str, Any]:
    path = _charts(root, question_id) / "index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _chart(root: Path, question_id: str, metric: str) -> dict[str, Any]:
    charts = _index(root, question_id)["charts"]
    return next(entry for entry in charts if entry["metric"] == metric)


def _figure(question_id: str) -> tuple[Figure, list[dict[str, Any]]]:
    """Draw one question straight from the fixture aggregates, without writing."""
    arms = _load(question_id)
    assert arms is not None
    labels = _labels(list(arms))
    title = _distribution_title(question_id, list(arms), labels)
    return _distribution_figure(arms, labels, title)


@pytest.fixture
def languages(data_dirs: Path) -> Path:
    """One question whose arms vary by language only, as 001a and 001c do."""
    _write_distributions(
        "001a",
        {
            "en": {
                "en": _cell({"apple": 3, "banana": 1}),
                "pl": _cell({"apple": 1, "banana": 2, "other": 1}),
            }
        },
    )
    return data_dirs


def test_arms_that_differ_by_language_are_labeled_by_language(languages: Path) -> None:
    analyze_question("001a")

    chart = _chart(languages, "001a", "distribution")
    assert chart["columns"] == ["en", "pl"]
    assert chart["title"] == "001a: answer distribution by language"


def test_arms_that_differ_by_schema_are_labeled_by_schema(data_dirs: Path) -> None:
    _write_distributions(
        "001d",
        {
            "en": {"pl": _cell({"apple": 2})},
            "none": {"pl": _cell({"banana": 2})},
            "pl": {"pl": _cell({"apple": 1, "banana": 1})},
        },
    )

    analyze_question("001d")

    chart = _chart(data_dirs, "001d", "distribution")
    assert chart["columns"] == ["en schema", "no schema", "pl schema"]
    assert chart["title"] == "001d: answer distribution by schema"


def test_both_dimensions_varying_are_named_together(data_dirs: Path) -> None:
    _write_distributions(
        "001e",
        {
            "en": {"en": _cell({"apple": 1}), "pl": _cell({"apple": 1})},
            "pl": {"pl": _cell({"banana": 1})},
        },
    )

    analyze_question("001e")

    chart = _chart(data_dirs, "001e", "distribution")
    assert chart["columns"] == ["en / en schema", "pl / en schema", "pl / pl schema"]
    assert chart["title"] == "001e: answer distribution by arm"


def test_shares_and_counts_come_from_the_aggregate(languages: Path) -> None:
    analyze_question("001a")

    rows = _chart(languages, "001a", "distribution")["rows"]
    apple = next(row for row in rows if row["label"] == "apple")
    assert apple["cells"] == [
        {"value": 0.75, "count": 3, "n": 4},
        {"value": 0.25, "count": 1, "n": 4},
    ]


def test_unpicked_categories_are_dropped_and_other_sorts_last(languages: Path) -> None:
    analyze_question("001a")

    labels = [row["label"] for row in _chart(languages, "001a", "distribution")["rows"]]
    assert "lychee" not in labels
    assert labels == ["apple", "banana", "other"]


def test_rows_are_written_in_the_order_they_are_drawn(languages: Path) -> None:
    """The y axis is inverted so row 0 is drawn on top, matching the row order.

    Sorting the rows backwards to compensate would put the table and the chart
    in opposite orders, which is the trap the old spec's reversed axis was.
    """
    figure, rows = _figure("001a")

    axes = figure.axes[0]
    drawn = [text.get_text() for text in axes.get_yticklabels()]
    assert drawn == [row["label"] for row in rows]
    assert axes.get_ylim()[0] > axes.get_ylim()[1]


def test_several_arms_are_keyed_by_a_legend(languages: Path) -> None:
    figure, _ = _figure("001a")

    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["en", "pl"]


def test_a_single_arm_needs_no_legend(data_dirs: Path) -> None:
    _write_distributions("001b", {"en": {"en": _cell({"apple": 2, "banana": 1})}})
    figure, _ = _figure("001b")

    assert figure.axes[0].get_legend() is None
    analyze_question("001b")
    assert _chart(data_dirs, "001b", "distribution")["title"] == (
        "001b: answer distribution (en)"
    )


def test_more_arms_than_the_palette_is_refused(data_dirs: Path) -> None:
    """Wrapping the palette would draw two arms in one color under a legend
    that claims they differ, so outgrowing it has to be a decision, not a wrap."""
    _write_distributions(
        "001f",
        {
            "en": {"en": _cell({"apple": 1})},
            "pl": {"en": _cell({"apple": 1})},
            "ja": {"en": _cell({"apple": 1})},
            "none": {"en": _cell({"banana": 1})},
        },
    )

    with pytest.raises(ValueError, match="palette"):
        analyze_question("001f")


def test_only_a_series_peak_is_directly_labeled() -> None:
    """A value beside every bar goes unread, so the table carries the rest."""
    assert _peak_labels([0.2, 0.8, 0.4], ["a", "b", "c"]) == [None, "b", None]
    assert _peak_labels([0.0, 0.0], ["a", "b"]) == [None, None]


def test_index_lists_every_chart_with_its_file_and_arms(languages: Path) -> None:
    outcome = analyze_question("001a")

    index = _index(languages, "001a")
    assert index["question_id"] == "001a"
    assert [(chart["metric"], chart["file"]) for chart in index["charts"]] == [
        ("distribution", "distribution.svg"),
    ]
    assert index["charts"][0]["arms"] == ["en", "pl"]
    assert outcome.index_path == _charts(languages, "001a") / "index.json"


def test_every_chart_is_written_as_a_transparent_svg(languages: Path) -> None:
    analyze_question("001a")

    path = _charts(languages, "001a") / "distribution.svg"
    assert ElementTree.parse(path).getroot().tag == _SVG_ROOT
    assert "dc:date" not in path.read_text(encoding="utf-8")


def test_redrawing_unchanged_aggregates_rewrites_an_identical_file(
    languages: Path,
) -> None:
    """A rerun that churned bytes would put a meaningless diff in every commit."""
    analyze_question("001a")
    first = (_charts(languages, "001a") / "distribution.svg").read_bytes()

    analyze_question("001a")

    assert (_charts(languages, "001a") / "distribution.svg").read_bytes() == first


def test_missing_aggregates_point_at_the_aggregate_command(data_dirs: Path) -> None:
    with pytest.raises(FileNotFoundError, match="llmango aggregate"):
        analyze_question("001a")
