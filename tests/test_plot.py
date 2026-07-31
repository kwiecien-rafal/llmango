"""Tests for the drawing itself: arm labels, shares, ordering, legend and palette."""

from itertools import combinations
from typing import Any, cast

import pytest

from colorimetry import contrast, delta_e, lightness_and_chroma
from conftest import SUPPORT, build_distribution
from llmango.aggregate import Aggregate, Distribution
from llmango.plot import (
    ARM_COLORS,
    DARK_SURFACE,
    INK,
    LIGHT_SURFACE,
    Drawn,
    _peak_labels,
    distribution,
    question_distribution,
)

_DUAL_BAND = (0.48, 0.67)
_CHROMA_FLOOR = 0.10
_CVD_TARGET = 8.0
_NORMAL_FLOOR = 15.0
_CONTRAST_MIN = 3.0


def _cell(counts: dict[str, int]) -> Distribution:
    return build_distribution(counts)


def _aggregate(
    question_id: str, distributions: dict[str, dict[str, Distribution]]
) -> Aggregate:
    return {
        "question_id": question_id,
        "support": SUPPORT,
        "distributions": distributions,
        "positions": {},
    }


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
    english, polish = _cells(drawn := question_distribution(languages), "apple")

    assert (english["value"], english["count"], english["n"]) == (0.75, 3, 4)
    assert (polish["value"], polish["count"], polish["n"]) == (0.25, 1, 4)
    assert drawn.row_label == "category"


def test_every_plotted_share_carries_the_interval_drawn_over_it(
    languages: Aggregate,
) -> None:
    """The caps are data, so the table has to carry them too or it is the lossier
    of the two views the site claims are twins."""
    english = _cells(question_distribution(languages), "apple")[0]

    assert english["lo"] < english["value"] < english["hi"]


def test_unpicked_categories_are_dropped_and_other_sorts_last(
    languages: Aggregate,
) -> None:
    drawn = question_distribution(languages)

    labels = [row["label"] for row in drawn.rows]
    assert "lychee" not in labels
    assert labels == ["apple", "banana", "other"]


def test_rows_are_written_in_the_order_they_are_drawn(languages: Aggregate) -> None:
    """Columns run left to right along x and the table runs down in the same order,
    so a reader moving between the two never has to reverse anything."""
    drawn = question_distribution(languages)

    axes = drawn.figure.axes[0]
    labels = [text.get_text() for text in axes.get_xticklabels()]
    assert labels == [row["label"] for row in drawn.rows]
    assert axes.get_ylim()[0] < axes.get_ylim()[1]


def test_categories_are_written_along_x_at_an_angle(languages: Aggregate) -> None:
    """Ten fruit names side by side do not fit at 360px unless they are turned."""
    labels = question_distribution(languages).figure.axes[0].get_xticklabels()

    assert {label.get_rotation() for label in labels} == {45.0}
    assert {label.get_horizontalalignment() for label in labels} == {"right"}


def test_a_chart_may_write_its_categories_however_its_experiment_names_them(
    languages: Aggregate,
) -> None:
    """The emoji beside a fruit is experiment knowledge, so the toolkit takes it
    as a hook rather than knowing that a category is ever a fruit."""
    drawn = question_distribution(languages, category_label=lambda name: f"{name} X")

    labels = [text.get_text() for text in drawn.figure.axes[0].get_xticklabels()]
    assert labels == ["apple X", "banana X", "other X"]
    assert [row["label"] for row in drawn.rows] == ["apple", "banana", "other"]


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

    first, second = _cells(drawn, "apple")
    assert drawn.columns == ["001a order", "001b order"]
    assert drawn.row_label == "category"
    assert (first["value"], first["count"]) == (0.75, 3)
    assert (second["value"], second["count"]) == (0.25, 1)


def test_a_chart_may_fix_the_order_its_categories_are_drawn_in() -> None:
    """Positions run 1 to 10 whatever their counts; sorting them by frequency
    would scramble the one axis whose order is the finding."""
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="001c: answer distribution by position",
        categories=["1", "2", "3"],
        row_label="position",
    )

    assert [row["label"] for row in drawn.rows] == ["1", "2", "3"]
    assert [row["cells"][0]["value"] for row in drawn.rows] == [0.8, 0.0, 0.2]


def test_a_chart_may_draw_the_line_it_is_read_against() -> None:
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="001c: answer distribution by position",
        categories=["1", "2", "3"],
        reference=0.1,
    )

    assert any(
        list(line.get_ydata()) == [0.1, 0.1]
        for line in drawn.figure.axes[0].get_lines()
    )


def test_only_a_series_peak_is_directly_labeled() -> None:
    """A value beside every bar goes unread, so the table carries the rest."""
    assert _peak_labels([0.2, 0.8, 0.4], ["a", "b", "c"]) == [None, "b", None]
    assert _peak_labels([0.0, 0.0], ["a", "b"]) == [None, None]


def test_every_arm_color_stays_inside_the_dual_surface_lightness_band() -> None:
    """A transparent export is read on both pages, so the usable band is the
    intersection of the two: too light vanishes on bone, too dark on ink."""
    for color in ARM_COLORS:
        lightness, _ = lightness_and_chroma(color)
        assert _DUAL_BAND[0] <= lightness <= _DUAL_BAND[1], color


def test_every_arm_color_carries_enough_chroma_to_do_identity_work() -> None:
    for color in ARM_COLORS:
        _, chroma = lightness_and_chroma(color)
        assert chroma >= _CHROMA_FLOOR, color


def test_every_arm_color_pair_separates_under_red_green_colorblindness() -> None:
    """All pairs, not adjacent ones: a grouped column chart puts any two series
    side by side, so a collapse anywhere in the palette is a collapse on screen."""
    for first, second in combinations(ARM_COLORS, 2):
        for deficiency in ("protan", "deutan"):
            assert delta_e(first, second, deficiency) >= _CVD_TARGET, (
                first,
                second,
                deficiency,
            )


def test_every_arm_color_pair_separates_under_ordinary_vision_too() -> None:
    for first, second in combinations(ARM_COLORS, 2):
        assert delta_e(first, second) >= _NORMAL_FLOOR, (first, second)


def test_every_arm_color_reads_against_both_page_surfaces() -> None:
    for color in ARM_COLORS:
        for surface in (LIGHT_SURFACE, DARK_SURFACE):
            assert contrast(color, surface) >= _CONTRAST_MIN, (color, surface)


def test_the_ink_reads_against_both_page_surfaces() -> None:
    """No neutral reaches 4.5:1 on both, so chart text is held to the large-text
    threshold and the index.json table is the accessible twin that carries it."""
    for surface in (LIGHT_SURFACE, DARK_SURFACE):
        assert contrast(INK, surface) >= _CONTRAST_MIN


def test_the_palette_has_no_room_for_a_fourth_arm() -> None:
    """The three-series cap is a measured property of a transparent export, not
    a stylistic choice, so the refusal is the honest behaviour rather than a wrap."""
    assert len(ARM_COLORS) == 3


def test_series_peaking_on_one_category_have_their_labels_stacked() -> None:
    """Every language picking the same fruit is the finding, not an edge case, so
    the three labels that land on one column are spread up it rather than overdrawn."""
    drawn = distribution(
        cells={
            "en": _cell({"lychee": 9, "apple": 1}),
            "pl": _cell({"lychee": 8, "apple": 2}),
            "ja": _cell({"lychee": 7, "apple": 3}),
        },
        title="001a: answer distribution by language",
    )

    written = drawn.figure.axes[0].texts
    anchors = {round(float(text.xy[1]), 6) for text in written}
    lifts = sorted(float(text.get_position()[1]) for text in written)

    assert sorted(text.get_text() for text in written) == ["70%", "80%", "90%"]
    assert len(anchors) == 1
    assert lifts == sorted(set(lifts))


def test_a_value_axis_ends_just_above_the_data_rather_than_at_full_scale() -> None:
    """A 0-100% axis when nothing clears 35% spends most of the plot on white space."""
    drawn = distribution(
        cells={"en": _cell({"apple": 2, "banana": 3, "grape": 5})},
        title="001a: answer distribution (en)",
    )

    assert drawn.figure.axes[0].get_ylim()[1] < 1.0


def test_short_categories_are_not_turned_when_they_already_fit() -> None:
    drawn = distribution(
        cells={"en": _cell({"1": 4, "2": 1})},
        title="001c: answer distribution by position",
        categories=["1", "2"],
    )

    labels = drawn.figure.axes[0].get_xticklabels()
    assert {label.get_rotation() for label in labels} == {0.0}


def test_a_category_an_arm_never_picked_still_carries_its_uncertainty() -> None:
    """0 of 5 is 0-43%, not certainty. A flat cap there would claim precision
    exactly where a run knows least, which is the reader's likeliest misreading."""
    drawn = distribution(
        cells={
            "en": _cell({"lychee": 5}),
            "pl": _cell({"apple": 5}),
        },
        title="001a: answer distribution by language",
    )

    unpicked = _cells(drawn, "apple")[0]
    assert unpicked["value"] == 0.0
    assert unpicked["lo"] == 0.0
    assert unpicked["hi"] > 0.4
