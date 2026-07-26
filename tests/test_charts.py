"""Tests for the drawn charts: arm labels, shares, ordering, index and output."""

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

import pytest
from matplotlib.figure import Figure

from llmango.aggregate import _write_json
from llmango.charts import (
    _distribution_figure,
    _distribution_title,
    _labels,
    _load,
    _peak_labels,
    analyze_experiment,
)

_EXPERIMENT = "001_fruit"
_SVG_ROOT = "{http://www.w3.org/2000/svg}svg"


def _cell(counts: dict[str, int]) -> dict[str, object]:
    total = sum(counts.values())
    return {
        "n": total,
        "counts": counts,
        "other_share": round(counts.get("other", 0) / total, 4) if total else 0.0,
    }


def _write_aggregate(
    name: str, questions: dict[str, dict[str, dict[str, object]]]
) -> None:
    """Write a fixture through the writer aggregate itself uses.

    Hand-rolling the envelope here would let these tests keep passing against a
    layout nothing produces, so the real writer builds it. It resolves AGG_DIR
    through the aggregate module, which data_dirs redirects into tmp_path.
    """
    _write_json(_EXPERIMENT, name, questions)


def _match_cell(
    total: int, matched: int, undetermined: int, rate: float
) -> dict[str, object]:
    return {
        "total": total,
        "matched": matched,
        "undetermined": undetermined,
        "rate": rate,
    }


def _charts(root: Path) -> Path:
    return root / "charts" / _EXPERIMENT


def _index(root: Path) -> dict[str, Any]:
    path = _charts(root) / "index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _chart(root: Path, metric: str) -> dict[str, Any]:
    return next(entry for entry in _index(root)["charts"] if entry["metric"] == metric)


def _figure(question_id: str) -> tuple[Figure, list[dict[str, Any]]]:
    """Draw one question straight from the fixture aggregates, without writing."""
    questions = _load(_EXPERIMENT, "distributions.json")
    assert questions is not None
    arms = questions[question_id]
    labels = _labels(list(arms))
    title = _distribution_title(question_id, list(arms), labels)
    return _distribution_figure(arms, labels, title)


@pytest.fixture
def languages(data_dirs: Path) -> Path:
    """One question whose arms vary by language only, as 001a and 001c do."""
    _write_aggregate(
        "distributions.json",
        {
            "001a": {
                "en": {
                    "en": _cell({"apple": 3, "banana": 1}),
                    "pl": _cell({"apple": 1, "banana": 2, "other": 1}),
                }
            }
        },
    )
    return data_dirs


def test_arms_that_differ_by_language_are_labeled_by_language(languages: Path) -> None:
    analyze_experiment("001")

    chart = _chart(languages, "distribution")
    assert chart["columns"] == ["en", "pl"]
    assert chart["title"] == "001a: answer distribution by language"


def test_arms_that_differ_by_schema_are_labeled_by_schema(data_dirs: Path) -> None:
    _write_aggregate(
        "distributions.json",
        {
            "001d": {
                "en": {"pl": _cell({"apple": 2})},
                "none": {"pl": _cell({"banana": 2})},
                "pl": {"pl": _cell({"apple": 1, "banana": 1})},
            }
        },
    )

    analyze_experiment("001")

    chart = _chart(data_dirs, "distribution")
    assert chart["columns"] == ["en schema", "no schema", "pl schema"]
    assert chart["title"] == "001d: answer distribution by schema"


def test_both_dimensions_varying_are_named_together(data_dirs: Path) -> None:
    _write_aggregate(
        "distributions.json",
        {
            "001e": {
                "en": {"en": _cell({"apple": 1}), "pl": _cell({"apple": 1})},
                "pl": {"pl": _cell({"banana": 1})},
            }
        },
    )

    analyze_experiment("001")

    chart = _chart(data_dirs, "distribution")
    assert chart["columns"] == ["en / en schema", "pl / en schema", "pl / pl schema"]
    assert chart["title"] == "001e: answer distribution by arm"


def test_shares_and_counts_come_from_the_aggregate(languages: Path) -> None:
    analyze_experiment("001")

    rows = _chart(languages, "distribution")["rows"]
    apple = next(row for row in rows if row["label"] == "apple")
    assert apple["cells"] == [
        {"value": 0.75, "count": 3, "n": 4},
        {"value": 0.25, "count": 1, "n": 4},
    ]


def test_unpicked_categories_are_dropped_and_other_sorts_last(languages: Path) -> None:
    analyze_experiment("001")

    labels = [row["label"] for row in _chart(languages, "distribution")["rows"]]
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
    _write_aggregate(
        "distributions.json",
        {"001b": {"en": {"en": _cell({"apple": 2, "banana": 1})}}},
    )
    figure, _ = _figure("001b")

    assert figure.axes[0].get_legend() is None
    analyze_experiment("001")
    assert _chart(data_dirs, "distribution")["title"] == (
        "001b: answer distribution (en)"
    )


def test_more_arms_than_the_palette_is_refused(data_dirs: Path) -> None:
    """Wrapping the palette would draw two arms in one color under a legend
    that claims they differ, so outgrowing it has to be a decision, not a wrap."""
    _write_aggregate(
        "distributions.json",
        {
            "001f": {
                "en": {"en": _cell({"apple": 1})},
                "pl": {"en": _cell({"apple": 1})},
                "ja": {"en": _cell({"apple": 1})},
                "none": {"en": _cell({"banana": 1})},
            }
        },
    )

    with pytest.raises(ValueError, match="palette"):
        analyze_experiment("001")


def test_only_a_series_peak_is_directly_labeled() -> None:
    """A value beside every bar goes unread, so the table carries the rest."""
    assert _peak_labels([0.2, 0.8, 0.4], ["a", "b", "c"]) == [None, "b", None]
    assert _peak_labels([0.0, 0.0], ["a", "b"]) == [None, None]


def test_a_rate_holds_every_arm_of_every_question_in_one_chart(
    languages: Path,
) -> None:
    _write_aggregate(
        "language_match.json",
        {
            "001a": {"en": {"en": _match_cell(5, 4, 0, 0.8)}},
            "001b": {"en": {"en": _match_cell(4, 3, 1, 1.0)}},
        },
    )

    analyze_experiment("001")

    chart = _chart(languages, "language_match")
    assert [row["label"] for row in chart["rows"]] == ["001a en", "001b en"]
    assert chart["rows"][1]["cells"] == [
        {"value": 1.0, "count": 3, "n": 4, "undetermined": 1}
    ]


def test_language_match_chart_appears_only_when_aggregated(languages: Path) -> None:
    analyze_experiment("001")
    assert not (_charts(languages) / "language_match.svg").exists()

    _write_aggregate(
        "language_match.json",
        {"001a": {"en": {"en": _match_cell(4, 4, 0, 1.0)}}},
    )
    analyze_experiment("001")

    assert (_charts(languages) / "language_match.svg").is_file()
    assert _chart(languages, "language_match")["rows"][0]["cells"][0]["value"] == 1.0


def test_index_lists_every_chart_with_its_file_and_arms(languages: Path) -> None:
    _write_aggregate(
        "language_match.json",
        {"001a": {"en": {"en": _match_cell(4, 4, 0, 1.0)}}},
    )

    outcome = analyze_experiment("001")

    index = _index(languages)
    assert index["experiment_id"] == _EXPERIMENT
    assert [(chart["metric"], chart["file"]) for chart in index["charts"]] == [
        ("distribution", "001a__distribution.svg"),
        ("language_match", "language_match.svg"),
    ]
    assert index["charts"][0]["arms"] == ["en", "pl"]
    assert index["charts"][1]["question_id"] is None
    assert outcome.index_path == _charts(languages) / "index.json"


def test_every_chart_is_written_as_a_transparent_svg(languages: Path) -> None:
    _write_aggregate(
        "language_match.json",
        {"001a": {"en": {"en": _match_cell(4, 4, 0, 1.0)}}},
    )

    analyze_experiment("001")

    for name in ("001a__distribution.svg", "language_match.svg"):
        path = _charts(languages) / name
        assert ElementTree.parse(path).getroot().tag == _SVG_ROOT
        assert "dc:date" not in path.read_text(encoding="utf-8")


def test_redrawing_unchanged_aggregates_rewrites_an_identical_file(
    languages: Path,
) -> None:
    """A rerun that churned bytes would put a meaningless diff in every commit."""
    analyze_experiment("001")
    first = (_charts(languages) / "001a__distribution.svg").read_bytes()

    analyze_experiment("001")

    assert (_charts(languages) / "001a__distribution.svg").read_bytes() == first


def test_missing_aggregates_point_at_the_aggregate_command(data_dirs: Path) -> None:
    with pytest.raises(FileNotFoundError, match="llmango aggregate"):
        analyze_experiment("001")
