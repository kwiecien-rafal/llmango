"""Tests for the drawing itself: arm labels, shares, ordering, legend and palette."""

from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.image import imsave
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import Patch

from colorimetry import contrast, delta_e, lightness_and_chroma
from conftest import SUPPORT, build_distribution
from llmango.aggregate import Aggregate, Distribution
from llmango.plot import (
    _ARTICLE_WIDTH_IN,
    _ICON_GAP_PT,
    _MIN_MARK_PX,
    _ZERO_MARK_PT,
    ARM_COLORS,
    COUNT,
    DARK_SURFACE,
    INK,
    LIGHT_SURFACE,
    Drawn,
    _dots_across,
    distribution,
    estimates,
    question_distribution,
    summary,
)

_TITLE = "Chart 1.1: answer distribution by language"
_DUAL_BAND = (0.48, 0.67)
_CHROMA_FLOOR = 0.10
_CVD_TARGET = 8.0
_NORMAL_FLOOR = 15.0
_CONTRAST_MIN = 3.0
_INLINE_SLACK_PX = 3.0


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
    drawn = question_distribution(languages, _TITLE)

    assert drawn.columns == ["en", "pl"]


def test_arms_that_differ_by_schema_are_labeled_by_schema() -> None:
    drawn = question_distribution(
        _aggregate(
            "001d",
            {
                "en": {"pl": _cell({"apple": 2})},
                "none": {"pl": _cell({"banana": 2})},
                "pl": {"pl": _cell({"apple": 1, "banana": 1})},
            },
        ),
        _TITLE,
    )

    assert drawn.columns == ["en schema", "no schema", "pl schema"]


def test_a_chart_may_name_a_schema_the_way_its_experiment_writes_it() -> None:
    """A schema's class name is what the code calls it, not what a reader compares,
    so the toolkit takes the naming as a hook rather than writing the name it has."""
    drawn = question_distribution(
        _aggregate(
            "001d",
            {
                "FruitChoice": {"pl": _cell({"apple": 2})},
                "none": {"pl": _cell({"banana": 2})},
            },
        ),
        _TITLE,
        schema_label=lambda schema: "no schema" if schema == "none" else "en schema",
    )

    assert drawn.columns == ["en schema", "no schema"]


def test_both_dimensions_varying_are_named_together() -> None:
    drawn = question_distribution(
        _aggregate(
            "001e",
            {
                "en": {"en": _cell({"apple": 1}), "pl": _cell({"apple": 1})},
                "pl": {"pl": _cell({"banana": 1})},
            },
        ),
        _TITLE,
    )

    assert drawn.columns == ["en / en schema", "pl / en schema", "pl / pl schema"]


def test_shares_and_counts_come_from_the_aggregate(languages: Aggregate) -> None:
    english, polish = _cells(drawn := question_distribution(languages, _TITLE), "apple")

    assert (english["value"], english["count"], english["n"]) == (0.75, 3, 4)
    assert (polish["value"], polish["count"], polish["n"]) == (0.25, 1, 4)
    assert drawn.row_label == "category"


def test_every_plotted_share_carries_its_interval_into_the_table(
    languages: Aggregate,
) -> None:
    """The columns draw no caps, so the table is where the uncertainty is read."""
    english = _cells(question_distribution(languages, _TITLE), "apple")[0]

    assert english["lo"] < english["value"] < english["hi"]


def test_no_column_is_drawn_with_a_cap_over_it(languages: Aggregate) -> None:
    """A grouped chart of ten fruits spends more ink on caps than on columns."""
    axes = question_distribution(languages, _TITLE).figure.axes[0]

    assert axes.get_lines() == []


def test_unpicked_categories_are_dropped_and_other_sorts_last(
    languages: Aggregate,
) -> None:
    drawn = question_distribution(languages, _TITLE)

    labels = [row["label"] for row in drawn.rows]
    assert "lychee" not in labels
    assert labels == ["apple", "banana", "other"]


def test_rows_are_written_in_the_order_they_are_drawn(languages: Aggregate) -> None:
    """Columns run left to right along x and the table runs down in the same order,
    so a reader moving between the two never has to reverse anything."""
    drawn = question_distribution(languages, _TITLE)

    axes = drawn.figure.axes[0]
    labels = [text.get_text() for text in axes.get_xticklabels()]
    assert labels == [row["label"] for row in drawn.rows]
    assert axes.get_ylim()[0] < axes.get_ylim()[1]


def test_bare_categories_are_written_along_x_at_an_angle(languages: Aggregate) -> None:
    """A word long enough to collide with its neighbour is turned rather than
    shrunk, since a name that has to be read is the one thing a chart cannot cut."""
    labels = question_distribution(languages, _TITLE).figure.axes[0].get_xticklabels()

    assert {label.get_rotation() for label in labels} == {45.0}
    assert {label.get_horizontalalignment() for label in labels} == {"right"}


def test_a_chart_may_write_its_categories_however_its_experiment_names_them(
    languages: Aggregate,
) -> None:
    """The emoji beside a fruit is experiment knowledge, so the toolkit takes it
    as a hook rather than knowing that a category is ever a fruit."""
    drawn = question_distribution(
        languages, _TITLE, category_label=lambda name: f"{name} X"
    )

    labels = [text.get_text() for text in drawn.figure.axes[0].get_xticklabels()]
    assert labels == ["apple X", "banana X", "other X"]
    assert [row["label"] for row in drawn.rows] == ["apple", "banana", "other"]


def test_a_chart_may_picture_a_category_the_way_its_experiment_illustrates_it(
    languages: Aggregate, tmp_path: Path
) -> None:
    """A color emoji is an image, since a font's glyphs are outlined into paths."""
    icon = tmp_path / "apple.png"
    imsave(icon, np.zeros((8, 8, 4)))

    drawn = question_distribution(
        languages, _TITLE, category_icon=lambda name: icon if name == "apple" else None
    )

    pictured = [
        child
        for child in drawn.figure.axes[0].get_children()
        if isinstance(child, AnnotationBbox)
    ]
    assert len(pictured) == 1


def test_a_pictured_category_is_written_flat_with_its_picture_ahead_of_the_word(
    languages: Aggregate, tmp_path: Path
) -> None:
    """A picture is what makes a category scannable without turning it, so a
    pictured axis reads flat, and reads as one label: the word follows its picture
    by the one gap the toolkit sets, and the two sit on one line."""
    icon = tmp_path / "fruit.png"
    imsave(icon, np.zeros((8, 8, 4)))

    drawn = question_distribution(languages, _TITLE, category_icon=lambda _: icon)
    figure = drawn.figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    labels = figure.axes[0].get_xticklabels()
    assert {label.get_rotation() for label in labels} == {0.0}
    gap = _ICON_GAP_PT * figure.dpi / 72.0
    for label, picture in zip(labels, _pictures(figure), strict=True):
        word = label.get_window_extent(renderer)
        shown = picture.get_window_extent(renderer)
        assert abs(word.x0 - shown.x1 - gap) < _INLINE_SLACK_PX
        assert abs((shown.y0 + shown.y1) - (word.y0 + word.y1)) < _INLINE_SLACK_PX


def _pictures(figure: Figure) -> list[AnnotationBbox]:
    return [
        child
        for child in figure.axes[0].get_children()
        if isinstance(child, AnnotationBbox)
    ]


def test_a_chart_writes_its_numbers_in_the_unit_it_plots() -> None:
    """A share and a count cannot share a formatter: 2.64 choices is not 264%."""
    drawn = summary(
        cells={"001a en": 1.0594, "001d pl/none": 2.6413},
        title="how many of the 10 fruits each arm was choosing between",
        value_label="effective choices (of 10)",
        row_label="arm",
        counts={"001a en": 300, "001d pl/none": 300},
        intervals={"001a en": (1.02, 1.12), "001d pl/none": (2.41, 2.88)},
        unit=COUNT,
    )

    written = drawn.figure.axes[0].texts
    assert drawn.unit == "count"
    assert sorted(text.get_text() for text in written) == ["1.06", "2.64"]


def test_a_summary_writes_its_columns_the_way_every_other_chart_does() -> None:
    """One number per named thing is still a column carrying a number, so it is
    written whole unless a decimal is what tells it from the column beside it."""
    drawn = summary(
        cells={"en": 0.208, "ja": 0.721, "pl": 0.741},
        title="how much of the fixed order was position",
        value_label="share of answers that moved",
        row_label="language",
        counts={"en": 300, "ja": 300, "pl": 300},
        intervals={"en": (0.18, 0.24), "ja": (0.69, 0.75), "pl": (0.71, 0.77)},
    )

    written = drawn.figure.axes[0].texts
    tabled = [cell["written"] for row in drawn.rows for cell in row["cells"]]
    assert [text.get_text() for text in written] == ["21%", "72%", "74%"]
    assert tabled == ["20.8%", "72.1%", "74.1%"]


def test_an_estimate_is_drawn_as_a_dot_on_the_interval_it_carries() -> None:
    """Ten single numbers is the one chart whose finding is which differences
    survive their intervals, so here the interval is the mark rather than a cap
    on one. Each row is a line from low to high with the estimate sitting on it."""
    drawn = estimates(
        cells={"001a en": 1.0594, "001d no schema": 2.6413},
        title="how many of the 10 fruits each arm was choosing between",
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300, "001d no schema": 300},
        intervals={"001a en": (1.0, 1.119), "001d no schema": (2.4628, 2.7936)},
        unit=COUNT,
        floor=1.0,
    )

    axes = drawn.figure.axes[0]
    spans = [line for line in axes.get_lines() if len(line.get_xdata()) == 2]
    assert [list(line.get_xdata()) for line in spans] == [
        [1.0, 1.119],
        [2.4628, 2.7936],
    ]
    assert [line.get_marker() for line in axes.get_lines() if line.get_marker() == "o"]
    assert sorted(text.get_text() for text in axes.texts) == ["1.06", "2.64"]


def test_an_estimate_chart_may_start_at_the_floor_its_statistic_has() -> None:
    """A dot stands on nothing, so unlike a column it does not have to be read
    from a zero the statistic can never reach."""
    drawn = estimates(
        cells={"001a en": 1.0594},
        title="how many of the 10 fruits each arm was choosing between",
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300},
        intervals={"001a en": (1.0, 1.119)},
        unit=COUNT,
        floor=1.0,
    )

    axes = drawn.figure.axes[0]
    assert axes.get_xlim()[0] == 1.0
    assert axes.get_ylim()[0] > axes.get_ylim()[1]
    assert [text.get_text() for text in axes.get_yticklabels()] == ["001a en"]


def test_every_plotted_estimate_carries_its_interval_into_the_table() -> None:
    drawn = estimates(
        cells={"001a en": 1.0594},
        title="how many of the 10 fruits each arm was choosing between",
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300},
        intervals={"001a en": (1.0, 1.119)},
        unit=COUNT,
    )

    cell = drawn.rows[0]["cells"][0]
    assert drawn.unit == "count"
    assert drawn.row_label == "arm"
    assert (cell["value"], cell["n"], cell["lo"], cell["hi"]) == (
        1.0594,
        300,
        1.0,
        1.119,
    )


def test_a_distribution_is_always_written_as_a_share(languages: Aggregate) -> None:
    assert question_distribution(languages, _TITLE).unit == "share"


def test_several_arms_are_keyed_by_a_legend(languages: Aggregate) -> None:
    legend = question_distribution(languages, _TITLE).figure.axes[0].get_legend()

    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["en", "pl"]


def test_a_title_is_centred_and_its_legend_keyed_off_to_the_right(
    languages: Aggregate,
) -> None:
    """The two sit on their own rows above the plot, so a long title never has to
    negotiate width with the key that names its series."""
    figure = question_distribution(languages, _TITLE).figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    axes = figure.axes[0]
    legend = axes.get_legend()
    assert axes.title.get_horizontalalignment() == "center"
    assert legend is not None
    keyed = legend.get_window_extent(renderer)
    assert abs(keyed.x1 - axes.get_window_extent().x1) < _INLINE_SLACK_PX
    assert keyed.y0 >= axes.get_window_extent().y1


def test_a_single_arm_needs_no_legend() -> None:
    drawn = question_distribution(
        _aggregate("001b", {"en": {"en": _cell({"apple": 2, "banana": 1})}}), _TITLE
    )

    assert drawn.figure.axes[0].get_legend() is None


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
            ),
            _TITLE,
        )


def test_a_chart_may_label_its_own_arms_across_questions() -> None:
    """An order comparison names its own series, since the arms it draws are
    identical in schema and language and differ only by the question asked."""
    drawn = distribution(
        cells={
            "001a order": _cell({"apple": 3, "banana": 1}),
            "001b order": _cell({"apple": 1, "banana": 3}),
        },
        title="answer distribution by option order (en)",
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
        title="answer distribution by position",
        categories=["1", "2", "3"],
        row_label="position",
    )

    assert [row["label"] for row in drawn.rows] == ["1", "2", "3"]
    assert [row["cells"][0]["value"] for row in drawn.rows] == [0.8, 0.0, 0.2]


def test_a_chart_with_too_many_categories_for_x_lays_its_bars_on_their_side() -> None:
    """Ten positions times three languages is thirty columns fighting over one x
    axis. Turned, every category name reads flat and the axis grows down the page
    instead, which is the one direction an article has to spare."""
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1}), "pl": _cell({"1": 1, "3": 4})},
        title="answer distribution by position",
        categories=["1", "2", "3"],
        horizontal=True,
    )

    axes = drawn.figure.axes[0]
    labels = [text.get_text() for text in axes.get_yticklabels()]
    assert labels == [row["label"] for row in drawn.rows]
    assert axes.get_xticklabels() != []
    assert {label.get_rotation() for label in axes.get_yticklabels()} == {0.0}


def test_a_horizontal_chart_reads_its_categories_down_from_the_top() -> None:
    """y climbs upward by default, which would put the first category last and
    reverse the table beside it."""
    axes = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        categories=["1", "2", "3"],
        horizontal=True,
    ).figure.axes[0]

    assert axes.get_ylim()[0] > axes.get_ylim()[1]


def test_a_horizontal_chart_writes_its_values_past_the_end_of_each_bar() -> None:
    """A bar running along x has no cap to sit over, and nothing crowds its side."""
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        categories=["1", "2", "3"],
        horizontal=True,
    )

    written = drawn.figure.axes[0].texts
    assert sorted(text.get_text() for text in written) == ["0%", "20%", "80%"]
    assert {text.get_rotation() for text in written} == {0.0}
    assert {text.get_horizontalalignment() for text in written} == {"left"}


def test_a_horizontal_chart_stands_its_reference_across_the_bars() -> None:
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        categories=["1", "2", "3"],
        reference=0.1,
        horizontal=True,
    )

    lines = drawn.figure.axes[0].get_lines()
    assert [list(line.get_xdata()) for line in lines] == [[0.1, 0.1]]


def test_a_chart_may_draw_the_line_it_is_read_against() -> None:
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        categories=["1", "2", "3"],
        reference=0.1,
    )

    lines = drawn.figure.axes[0].get_lines()
    assert [list(line.get_ydata()) for line in lines] == [[0.1, 0.1]]


def test_a_category_an_arm_never_picked_keeps_the_footprint_of_its_bar() -> None:
    """A column of no height is not a mark, so a zero is drawn as the dotted
    footprint of the bar it did not draw: the arm's own color, the width of its
    column, and no height at all, which reads as none rather than as nothing."""
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        categories=["1", "2", "3"],
    )

    axes = drawn.figure.axes[0]
    flat = [patch for patch in axes.patches if _tall(patch) == 0.0]
    assert sorted(text.get_text() for text in axes.texts) == ["0%", "20%", "80%"]
    assert len(flat) == 1

    mark = flat[0]
    column = next(patch for patch in axes.patches if _tall(patch) > 0.0)
    assert to_hex(mark.get_edgecolor()) == ARM_COLORS[0].lower()
    assert mark.get_facecolor()[3] == 0.0
    assert mark.get_linestyle() != "solid"
    assert _wide(mark) == pytest.approx(_wide(column))


def test_a_zero_mark_ends_on_a_whole_dot() -> None:
    """A fixed pattern divides an arbitrary bar width into whole dots and a
    remainder, and the remainder is drawn as a stub at the end of the row."""
    for width in (11.0, 17.3, 30.27, 44.5):
        _, (on, off) = _dots_across(width)
        dot, gap = on * _ZERO_MARK_PT, off * _ZERO_MARK_PT
        count = round((width + gap) / (dot + gap))

        assert count >= 2
        assert count * dot + (count - 1) * gap == pytest.approx(width)


def _tall(patch: Patch) -> float:
    vertices = patch.get_path().vertices
    return float(vertices[:, 1].max() - vertices[:, 1].min())


def _wide(patch: Patch) -> float:
    vertices = patch.get_path().vertices
    return float(vertices[:, 0].max() - vertices[:, 0].min())


def test_a_share_too_small_to_draw_is_still_given_a_visible_mark() -> None:
    """A bar a fraction of a pixel tall reads as the zero beside it, not as a few."""
    drawn = distribution(
        cells={"en": _cell({"apple": 997, "banana": 3})},
        title="answer distribution (en)",
    )

    axes = drawn.figure.axes[0]
    _, banana = axes.patches
    height = float(banana.get_path().vertices[:, 1].max())

    assert height > 0.003
    assert height / axes.get_ylim()[1] * axes.get_window_extent().height >= _MIN_MARK_PX
    assert "0.3%" in [text.get_text() for text in axes.texts]


def test_a_column_is_written_whole_where_a_decimal_would_say_nothing() -> None:
    """A decimal a column does not need is width taken from the column beside it,
    and width is the whole reason a value can be written flat at all."""
    drawn = distribution(
        cells={"en": _cell({"apple": 762, "banana": 191, "grape": 47})},
        title="answer distribution (en)",
    )

    written = drawn.figure.axes[0].texts
    assert sorted(text.get_text() for text in written) == ["19%", "5%", "76%"]


def test_a_share_is_written_to_the_decimal_its_reading_turns_on() -> None:
    """Written whole, 99.5% and 0.5% are a unanimous arm and a category never
    picked. Both ends round into a certainty the number does not have, so a share
    near either of them keeps the decimal that says which it is."""
    drawn = distribution(
        cells={"en": _cell({"apple": 995, "banana": 5})},
        title="answer distribution (en)",
    )

    written = drawn.figure.axes[0].texts
    assert sorted(text.get_text() for text in written) == ["0.5%", "99.5%"]


def test_a_share_too_fine_for_one_decimal_keeps_a_second() -> None:
    """One in two thousand is not the nothing that 0.0% would call it."""
    drawn = distribution(
        cells={"en": _cell({"apple": 1999, "banana": 1})},
        title="answer distribution (en)",
    )

    written = drawn.figure.axes[0].texts
    assert sorted(text.get_text() for text in written) == ["0.05%", "99.95%"]


def test_columns_too_close_to_tell_apart_whole_keep_their_decimals() -> None:
    """Written whole, 2.2% and 2.4% are one number twice, which is a chart saying
    two arms agreed when they did not. The column far from both stays whole: what
    earns a decimal is a neighbour it would otherwise be confused with."""
    drawn = distribution(
        cells={
            "en": _cell({"apple": 22, "banana": 750, "grape": 228}),
            "pl": _cell({"apple": 24, "banana": 750, "grape": 226}),
            "ja": _cell({"apple": 750, "banana": 22, "grape": 228}),
        },
        title="answer distribution by language",
    )

    written = [text.get_text() for text in drawn.figure.axes[0].texts]
    slots = len(drawn.rows)
    apple = [row["label"] for row in drawn.rows].index("apple")
    assert [written[arm * slots + apple] for arm in range(3)] == ["2.2%", "2.4%", "75%"]


def test_the_table_keeps_the_precision_the_chart_rounds_away() -> None:
    """The site is handed strings to print, not numbers to format, so both are
    written here. They are not the same string: a column is written to be read at
    a glance and the table is the record, which is where the decimal survives."""
    drawn = distribution(
        cells={"en": _cell({"apple": 984, "banana": 16})},
        title="answer distribution (en)",
    )

    drawn_labels = sorted(text.get_text() for text in drawn.figure.axes[0].texts)
    tabled = sorted(cell["written"] for row in drawn.rows for cell in row["cells"])
    assert drawn_labels == ["2%", "98%"]
    assert tabled == ["1.6%", "98.4%"]


def test_a_picked_category_is_never_tabled_as_the_zero_it_is_not() -> None:
    """The guard that keeps a drawn share off zero has to reach the table too."""
    drawn = distribution(
        cells={"en": _cell({"apple": 4999, "banana": 1})},
        title="answer distribution (en)",
    )

    banana = _cells(drawn, "banana")[0]
    assert banana["value"] > 0.0
    assert banana["written"] == "0.02%"


def test_a_tabled_estimate_is_written_in_the_unit_its_chart_plots() -> None:
    """A share and a count cannot share a formatter in the table either."""
    drawn = estimates(
        cells={"001a en": 1.0594},
        title="how many of the 10 fruits each arm was choosing between",
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300},
        intervals={"001a en": (1.0, 1.119)},
        unit=COUNT,
    )

    cell = drawn.rows[0]["cells"][0]
    assert cell["written"] == "1.06"
    assert cell["written_interval"] == "1–1.12"


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


def _packed() -> Drawn:
    """Three arms over all ten fruits, which is more columns than flat values fit."""
    fruits = ["apple", "banana", "grape", "lychee", "mango"]
    fruits += ["orange", "pineapple", "pomegranate", "strawberry", "watermelon"]

    return distribution(
        cells={
            arm: _cell(
                {fruit: 10 + index + place for place, fruit in enumerate(fruits)}
            )
            for index, arm in enumerate(("en", "pl", "ja"))
        },
        title="answer distribution by language",
    )


def test_every_column_carries_its_own_value() -> None:
    """A column a reader cannot read the number off is a column drawn twice: once
    in the chart and once in the table, with only the table saying anything."""
    drawn = distribution(
        cells={
            "en": _cell({"lychee": 6, "apple": 2, "grape": 1, "banana": 1}),
            "pl": _cell({"lychee": 5, "apple": 3, "grape": 1, "banana": 1}),
            "ja": _cell({"lychee": 4, "apple": 4, "grape": 1, "banana": 1}),
        },
        title="answer distribution by language",
    )

    written = drawn.figure.axes[0].texts
    assert len(written) == 12
    assert all(text.get_text().endswith("%") for text in written)


def test_values_are_turned_when_flat_ones_would_not_fit_their_columns() -> None:
    """The figure no longer widens to fit its values, so the other side of that
    same trade applies: a value that will not fit its column flat turns to fit."""
    figure = _packed().figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    written = figure.axes[0].texts
    assert {text.get_rotation() for text in written} == {90.0}
    boxes = [text.get_window_extent(renderer) for text in written]
    for first, second in combinations(boxes, 2):
        assert not first.overlaps(second)


def test_a_turned_value_stands_over_the_middle_of_its_own_column() -> None:
    """Turned, what is centred is the glyph band rather than the written word, so
    a value reads centred over its column instead of only measuring so."""
    figure = _packed().figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    axes = figure.axes[0]
    column = _wide(next(iter(axes.patches)))
    origin, edge = axes.transData.transform([(0.0, 0.0), (column, 0.0)])
    for text in axes.texts:
        anchor = axes.transData.transform((float(text.xy[0]), 0.0))[0]
        box = text.get_window_extent(renderer)
        assert abs((box.x0 + box.x1) / 2 - anchor) < _INLINE_SLACK_PX
        assert box.width < edge[0] - origin[0]


def test_a_turned_value_is_given_room_above_the_tallest_column() -> None:
    """Standing on end, a value claims the headroom a flat one never needed."""
    turned = _packed()
    flat = distribution(
        cells={"en": _cell({"lychee": 6, "apple": 2, "grape": 1, "banana": 1})},
        title="answer distribution (en)",
    )

    assert {text.get_rotation() for text in turned.figure.axes[0].texts} == {90.0}
    assert {text.get_rotation() for text in flat.figure.axes[0].texts} == {0.0}
    assert _headroom(turned) > _headroom(flat)


def _headroom(drawn: Drawn) -> float:
    axes = drawn.figure.axes[0]
    tallest = max(_tall(patch) for patch in axes.patches)
    return axes.get_ylim()[1] / tallest


def test_values_written_over_one_group_never_collide(languages: Aggregate) -> None:
    """Three arms picking the same fruit is the finding, not an edge case, so a
    value that cannot clear the one beside it turns rather than overlapping it."""
    figure = question_distribution(languages, _TITLE).figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    boxes = [text.get_window_extent(renderer) for text in figure.axes[0].texts]
    for first, second in combinations(boxes, 2):
        assert not first.overlaps(second)


def test_a_lone_series_writes_its_value_flat() -> None:
    """One column per category has the whole slot to itself, so turning the value
    would cost legibility to buy room that is not needed."""
    drawn = distribution(
        cells={"en": _cell({"lychee": 3, "apple": 1})},
        title="answer distribution (en)",
    )

    written = drawn.figure.axes[0].texts
    assert sorted(text.get_text() for text in written) == ["25%", "75%"]
    assert {text.get_rotation() for text in written} == {0.0}


def test_every_chart_is_drawn_at_the_width_the_page_gives_it() -> None:
    """The site renders a chart at the width of its column whatever the SVG says,
    so a figure drawn wider is a figure whose type the browser shrinks. One width
    for every form is what makes a declared point size the size read on the page,
    and what keeps two charts on one article the same size as each other."""
    drawn = [
        distribution(
            cells={"en": _cell({"apple": 3, "banana": 1})},
            title="answer distribution (en)",
        ),
        distribution(
            cells={"en": _cell({"1": 4, "3": 1})},
            title="answer distribution by position",
            categories=["1", "2", "3"],
            horizontal=True,
        ),
        estimates(
            cells={"001a en": 1.0594},
            title="how many of the 10 fruits each arm was choosing between",
            value_label="effective choices",
            row_label="arm",
            counts={"001a en": 300},
            intervals={"001a en": (1.0, 1.119)},
            unit=COUNT,
        ),
    ]

    widths = {figure.get_size_inches()[0] for figure in (one.figure for one in drawn)}
    assert widths == {_ARTICLE_WIDTH_IN}


def test_a_title_too_long_for_the_figure_is_broken_over_lines() -> None:
    """A figure used to widen until its title fit, which is the same knob that
    made a chart small on the page. At one width a long title has to wrap."""
    drawn = estimates(
        cells={"001a en": 1.0594},
        title="how many of the 10 fruits each arm was choosing between",
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300},
        intervals={"001a en": (1.0, 1.119)},
        unit=COUNT,
    )
    figure = drawn.figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()

    title = figure.axes[0].title
    assert "\n" in title.get_text()
    assert drawn.title == "how many of the 10 fruits each arm was choosing between"
    assert (
        title.get_window_extent(figure.canvas.get_renderer()).x1
        <= figure.get_size_inches()[0] * figure.dpi
    )


def test_a_value_axis_ends_just_above_the_data_rather_than_at_full_scale() -> None:
    """A 0-100% axis when nothing clears 35% spends most of the plot on white space."""
    drawn = distribution(
        cells={"en": _cell({"apple": 2, "banana": 3, "grape": 5})},
        title="answer distribution (en)",
    )

    assert drawn.figure.axes[0].get_ylim()[1] < 1.0


def test_short_categories_are_not_turned_when_they_already_fit() -> None:
    drawn = distribution(
        cells={"en": _cell({"1": 4, "2": 1})},
        title="answer distribution by position",
        categories=["1", "2"],
    )

    labels = drawn.figure.axes[0].get_xticklabels()
    assert {label.get_rotation() for label in labels} == {0.0}


def test_a_category_an_arm_never_picked_still_carries_its_uncertainty() -> None:
    """0 of 5 is 0-43%, not certainty, and only the table still says so."""
    drawn = distribution(
        cells={
            "en": _cell({"lychee": 5}),
            "pl": _cell({"apple": 5}),
        },
        title="answer distribution by language",
    )

    unpicked = _cells(drawn, "apple")[0]
    assert unpicked["value"] == 0.0
    assert unpicked["lo"] == 0.0
    assert unpicked["hi"] > 0.4
