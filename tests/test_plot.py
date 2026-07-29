"""Tests for the drawing itself: arm labels, shares, ordering, legend and palette."""

from typing import Any, cast

import pytest

from llmango.aggregate import Aggregate, Distribution
from llmango.plot import Drawn, _peak_labels, distribution, question_distribution


def _cell(counts: dict[str, int]) -> Distribution:
    total = sum(counts.values())
    return {
        "n": total,
        "counts": counts,
        "other_share": round(counts.get("other", 0) / total, 4) if total else 0.0,
    }


def _aggregate(
    question_id: str, distributions: dict[str, dict[str, Distribution]]
) -> Aggregate:
    return {"question_id": question_id, "distributions": distributions}


def _cells(drawn: Drawn, label: str) -> list[dict[str, Any]]:
    row = next(entry for entry in drawn.rows if entry["label"] == label)
    return cast(list[dict[str, Any]], row["cells"])


@pytest.fixture
def languages() -> Aggregate:
    """One question whose arms vary by language only, as 001a and 001c do."""
    return _aggregate(
        "001a",
        {
            "en": {
                "en": _cell({"apple": 3, "banana": 1}),
                "pl": _cell({"apple": 1, "banana": 2, "other": 1}),
            }
        },
    )


def test_arms_that_differ_by_language_are_labeled_by_language(
    languages: Aggregate,
) -> None:
    drawn = question_distribution(languages)

    assert drawn.columns == ["en", "pl"]
    assert drawn.title == "001a: answer distribution by language"


def test_arms_that_differ_by_schema_are_labeled_by_schema() -> None:
    drawn = question_distribution(
        _aggregate(
            "001d",
            {
                "en": {"pl": _cell({"apple": 2})},
                "none": {"pl": _cell({"banana": 2})},
                "pl": {"pl": _cell({"apple": 1, "banana": 1})},
            },
        )
    )

    assert drawn.columns == ["en schema", "no schema", "pl schema"]
    assert drawn.title == "001d: answer distribution by schema"


def test_both_dimensions_varying_are_named_together() -> None:
    drawn = question_distribution(
        _aggregate(
            "001e",
            {
                "en": {"en": _cell({"apple": 1}), "pl": _cell({"apple": 1})},
                "pl": {"pl": _cell({"banana": 1})},
            },
        )
    )

    assert drawn.columns == ["en / en schema", "pl / en schema", "pl / pl schema"]
    assert drawn.title == "001e: answer distribution by arm"


def test_shares_and_counts_come_from_the_aggregate(languages: Aggregate) -> None:
    drawn = question_distribution(languages)

    assert _cells(drawn, "apple") == [
        {"value": 0.75, "count": 3, "n": 4},
        {"value": 0.25, "count": 1, "n": 4},
    ]


def test_unpicked_categories_are_dropped_and_other_sorts_last(
    languages: Aggregate,
) -> None:
    drawn = question_distribution(languages)

    labels = [row["label"] for row in drawn.rows]
    assert "lychee" not in labels
    assert labels == ["apple", "banana", "other"]


def test_rows_are_written_in_the_order_they_are_drawn(languages: Aggregate) -> None:
    """The y axis is inverted so row 0 is drawn on top, matching the row order.

    Sorting the rows backwards to compensate would put the table and the chart
    in opposite orders, which is the trap the old spec's reversed axis was.
    """
    drawn = question_distribution(languages)

    axes = drawn.figure.axes[0]
    labels = [text.get_text() for text in axes.get_yticklabels()]
    assert labels == [row["label"] for row in drawn.rows]
    assert axes.get_ylim()[0] > axes.get_ylim()[1]


def test_several_arms_are_keyed_by_a_legend(languages: Aggregate) -> None:
    legend = question_distribution(languages).figure.axes[0].get_legend()

    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["en", "pl"]


def test_a_single_arm_needs_no_legend() -> None:
    drawn = question_distribution(
        _aggregate("001b", {"en": {"en": _cell({"apple": 2, "banana": 1})}})
    )

    assert drawn.figure.axes[0].get_legend() is None
    assert drawn.title == "001b: answer distribution (en)"


def test_more_arms_than_the_palette_is_refused() -> None:
    """Wrapping the palette would draw two arms in one color under a legend
    that claims they differ, so outgrowing it has to be a decision, not a wrap."""
    with pytest.raises(ValueError, match="palette"):
        question_distribution(
            _aggregate(
                "001f",
                {
                    "en": {"en": _cell({"apple": 1})},
                    "ja": {"en": _cell({"apple": 1})},
                    "none": {"en": _cell({"banana": 1})},
                    "pl": {"en": _cell({"apple": 1})},
                },
            )
        )


def test_a_chart_may_label_its_own_arms_across_questions() -> None:
    """An order comparison names its own series, since the arms it draws are
    identical in schema and language and differ only by the question asked."""
    drawn = distribution(
        cells={
            "001a order": _cell({"apple": 3, "banana": 1}),
            "001b order": _cell({"apple": 1, "banana": 3}),
        },
        title="001a / 001b: answer distribution by option order (en)",
    )

    assert drawn.columns == ["001a order", "001b order"]
    assert drawn.row_label == "category"
    assert _cells(drawn, "apple") == [
        {"value": 0.75, "count": 3, "n": 4},
        {"value": 0.25, "count": 1, "n": 4},
    ]


def test_only_a_series_peak_is_directly_labeled() -> None:
    """A value beside every bar goes unread, so the table carries the rest."""
    assert _peak_labels([0.2, 0.8, 0.4], ["a", "b", "c"]) == [None, "b", None]
    assert _peak_labels([0.0, 0.0], ["a", "b"]) == [None, None]
