"""Tests for the drawing itself: arm labels, shares, ordering, legend and panels."""

from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.image import imsave
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.text import Text

from conftest import SUPPORT, build_distribution
from llmango.aggregate import Aggregate, Distribution
from llmango.plot import (
    _ICON_GAP_PT,
    _MIN_MARK_PX,
    _ZERO_MARK_PT,
    COUNT,
    FRAME,
    NARROW,
    WIDE,
    Drawn,
    _dots_across,
    distribution,
    estimates,
    panels,
    question_distribution,
    styled,
    table,
)

_TITLE = "Chart 1.1: answer distribution by language"
_INLINE_SLACK_PX = 3.0
_TURNED_DEGREES = (315.0, 270.0)
_WHEEL = ("#0072B2", "#D55E00", "#009E73", "#000000")


def _turned(axes: Any) -> list[Text]:
    """The runs of category names that ran out of column and turned downward."""
    return [text for text in axes.texts if text.get_rotation() in _TURNED_DEGREES]


def _values(axes: Any) -> list[Text]:
    """The numbers written over the columns, which is every text that is not a name."""
    return [text for text in axes.texts if text.get_rotation() not in _TURNED_DEGREES]


def _palette() -> Callable[[str], str]:
    """A declared palette, the way an experiment owns one: a series name to a hex."""
    given: dict[str, str] = {}

    def color(series: str) -> str:
        return given.setdefault(series, _WHEEL[len(given) % len(_WHEEL)])

    return color


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
    drawn = question_distribution(languages, _TITLE, _palette())

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
        _palette(),
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
        _palette(),
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
        _palette(),
    )

    assert drawn.columns == ["en / en schema", "pl / en schema", "pl / pl schema"]


def test_shares_and_counts_come_from_the_aggregate(languages: Aggregate) -> None:
    english, polish = _cells(
        drawn := question_distribution(languages, _TITLE, _palette()), "apple"
    )

    assert (english["value"], english["count"], english["n"]) == (0.75, 3, 4)
    assert (polish["value"], polish["count"], polish["n"]) == (0.25, 1, 4)
    assert drawn.row_label == "category"


def test_every_plotted_share_carries_its_interval_into_the_table(
    languages: Aggregate,
) -> None:
    """The columns draw no caps, so the table is where the uncertainty is read."""
    english = _cells(question_distribution(languages, _TITLE, _palette()), "apple")[0]

    assert english["lo"] < english["value"] < english["hi"]


def test_no_column_is_drawn_with_a_cap_over_it(languages: Aggregate) -> None:
    """A grouped chart of ten fruits spends more ink on caps than on columns."""
    axes = question_distribution(languages, _TITLE, _palette()).figure.axes[0]

    assert axes.get_lines() == []


def test_unpicked_categories_are_dropped_and_other_sorts_last(
    languages: Aggregate,
) -> None:
    drawn = question_distribution(languages, _TITLE, _palette())

    labels = [row["label"] for row in drawn.rows]
    assert "lychee" not in labels
    assert labels == ["apple", "banana", "other"]


def test_rows_are_written_in_the_order_they_are_drawn(languages: Aggregate) -> None:
    """Columns run left to right along x and the table runs down in the same order,
    so a reader moving between the two never has to reverse anything."""
    drawn = question_distribution(languages, _TITLE, _palette())

    axes = drawn.figure.axes[0]
    labels = [text.get_text() for text in axes.get_xticklabels()]
    assert labels == [row["label"] for row in drawn.rows]
    assert axes.get_ylim()[0] < axes.get_ylim()[1]


def test_a_name_too_long_for_its_column_turns_down_at_the_edge_of_it() -> None:
    """A word long enough to collide with its neighbour is neither tilted whole
    nor shrunk: what fits is written flat and the rest turns the corner and reads
    down the side of the column, so the axis is still read left to right."""
    drawn = distribution(
        cells={"en": _cell({f"category number {slot}": 1 for slot in range(8)})},
        title=_TITLE,
        series_color=_palette(),
    )

    axes = drawn.figure.axes[0]
    assert {label.get_rotation() for label in axes.get_xticklabels()} == {0.0}
    assert _read_along(axes) == [row["label"] for row in drawn.rows]


def test_a_broken_name_bends_by_degrees_rather_than_snapping() -> None:
    """A name that snapped through a right angle read as two names set at right
    angles to each other. The letter it broke at takes half the turn instead, so
    the word bends around the edge of its column: flat, half way round, then
    straight down."""
    axes = distribution(
        cells={"en": _cell({f"category number {slot}": 1 for slot in range(8)})},
        title=_TITLE,
        series_color=_palette(),
    ).figure.axes[0]

    flat = axes.get_xticklabels()[0].get_text()
    half, down = _turned_at(axes, 0)
    assert (half.get_rotation(), down.get_rotation()) == (315.0, 270.0)
    assert len(half.get_text()) == 1
    assert flat + half.get_text() + down.get_text() == "category number 0"


def test_a_bending_name_carries_on_around_its_corner_without_a_gap() -> None:
    """The runs are one word, so each is laid against the one before it: the flat
    part, the letter half way round and the run straight down all touch, and the
    letters carry on rather than restarting somewhere past the bend."""
    figure = distribution(
        cells={"en": _cell({f"category number {slot}": 1 for slot in range(8)})},
        title=_TITLE,
        series_color=_palette(),
    ).figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    axes = figure.axes[0]
    for slot, label in enumerate(axes.get_xticklabels()):
        boxes = [label.get_window_extent(renderer)] + [
            text.get_window_extent(renderer) for text in _turned_at(axes, slot)
        ]
        for earlier, later in zip(boxes, boxes[1:], strict=False):
            assert earlier.overlaps(later)


def _turned_at(axes: Any, slot: int) -> list[Text]:
    """The runs one category's name turned into, in the order they are read."""
    return [text for text in _turned(axes) if int(text.xy[0]) == slot]


def _read_along(axes: Any) -> list[str]:
    """Read every category name back off the axis, run by run, the way a reader does."""
    return [
        label.get_text() + "".join(text.get_text() for text in _turned_at(axes, slot))
        for slot, label in enumerate(axes.get_xticklabels())
    ]


def test_a_name_that_fits_its_column_keeps_the_whole_of_it() -> None:
    """Only a name that has to turn owes the turn the width it will take, so a
    name that fits is never broken to reserve room it was never going to use."""
    drawn = distribution(
        cells={"en": _cell({"pomegranate": 3, "grape": 1, "watermelon": 1})},
        title=_TITLE,
        series_color=_palette(),
    )

    axes = drawn.figure.axes[0]
    assert [label.get_text() for label in axes.get_xticklabels()] == [
        row["label"] for row in drawn.rows
    ]
    assert _turned(axes) == []


def test_bare_categories_that_fit_beside_each_other_are_left_flat(
    languages: Aggregate,
) -> None:
    """Whether a name fits is measured, not assumed from its length: three short
    words over an article-wide figure sit side by side, and turning them would
    spend height on hanging words that had room to stand up."""
    labels = (
        question_distribution(languages, _TITLE, _palette())
        .figure.axes[0]
        .get_xticklabels()
    )

    assert {label.get_rotation() for label in labels} == {0.0}


def test_a_chart_may_write_its_categories_however_its_experiment_names_them(
    languages: Aggregate,
) -> None:
    """The emoji beside a fruit is experiment knowledge, so the toolkit takes it
    as a hook rather than knowing that a category is ever a fruit."""
    drawn = question_distribution(
        languages, _TITLE, _palette(), category_label=lambda name: f"{name} X"
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
        languages,
        _TITLE,
        _palette(),
        category_icon=lambda name: icon if name == "apple" else None,
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

    drawn = question_distribution(
        languages, _TITLE, _palette(), category_icon=lambda _: icon
    )
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


def test_an_estimate_is_drawn_as_a_dot_on_the_interval_it_carries() -> None:
    """Ten single numbers is the one chart whose finding is which differences
    survive their intervals, so here the interval is the mark rather than a cap
    on one. Each row is a line from low to high with the estimate sitting on it.

    A share and a count cannot share a formatter: 2.64 choices is not 264%."""
    drawn = estimates(
        cells={"001a en": 1.0594, "001d no schema": 2.6413},
        title="how many of the 10 fruits each arm was choosing between",
        series_color=_palette(),
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
        series_color=_palette(),
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
        series_color=_palette(),
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


def test_an_estimate_chart_is_keyed_by_what_its_colors_stand_for() -> None:
    """A dot's row names the arm, not the language its color encodes, so the key
    is handed in rather than read off the rows the way a series chart reads it."""
    drawn = estimates(
        cells={"001a en": 1.0594, "001a pl": 2.6413},
        title="how many of the 10 fruits each arm was choosing between",
        series_color=_palette(),
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300, "001a pl": 300},
        intervals={"001a en": (1.0, 1.119), "001a pl": (2.4628, 2.7936)},
        key={"en": "#0072B2", "pl": "#D55E00"},
        unit=COUNT,
        floor=1.0,
    )

    legend = drawn.figure.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["en", "pl"]
    assert [
        to_hex(cast(Patch, handle).get_facecolor()) for handle in legend.legend_handles
    ] == ["#0072b2", "#d55e00"]


def test_an_estimate_chart_given_no_key_is_drawn_without_one() -> None:
    drawn = estimates(
        cells={"001a en": 1.0594},
        title="how many of the 10 fruits each arm was choosing between",
        series_color=_palette(),
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300},
        intervals={"001a en": (1.0, 1.119)},
        unit=COUNT,
    )

    assert drawn.figure.axes[0].get_legend() is None


def test_a_distribution_is_always_written_as_a_share(languages: Aggregate) -> None:
    assert question_distribution(languages, _TITLE, _palette()).unit == "share"


def test_several_arms_are_keyed_by_a_legend(languages: Aggregate) -> None:
    legend = (
        question_distribution(languages, _TITLE, _palette()).figure.axes[0].get_legend()
    )

    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["en", "pl"]


def test_a_title_is_centred_on_the_canvas_and_its_legend_keyed_off_to_the_right(
    languages: Aggregate,
) -> None:
    """The two sit on their own rows above the plot, so a long title never has to
    negotiate width with the key that names its series. What the title is centred
    on is the canvas and not the plot, which the labels down the value axis push
    to the right: centred on the plot, a title long enough to wrap is drawn off
    the edge of the canvas and the writing past that edge is lost."""
    figure = question_distribution(languages, _TITLE, _palette()).figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    axes = figure.axes[0]
    legend = axes.get_legend()
    assert legend is not None
    titled = figure.texts[0].get_window_extent(renderer)
    keyed = legend.get_window_extent(renderer)
    assert titled.x0 >= 0.0 and titled.x1 <= figure.bbox.x1
    assert abs(titled.x0 + titled.x1 - figure.bbox.x1) < _INLINE_SLACK_PX
    assert abs(keyed.x1 - axes.get_window_extent().x1) < _INLINE_SLACK_PX
    assert axes.get_window_extent().y1 <= keyed.y0 and keyed.y1 <= titled.y0


def test_a_single_arm_needs_no_legend() -> None:
    drawn = question_distribution(
        _aggregate("001b", {"en": {"en": _cell({"apple": 2, "banana": 1})}}),
        _TITLE,
        _palette(),
    )

    assert drawn.figure.axes[0].get_legend() is None


def test_a_series_with_no_color_declared_is_refused_by_name() -> None:
    """The palette is an experiment's to declare, so the toolkit holds no hex of
    its own to fall back on: a series nobody gave a color is an authoring mistake,
    and what says so names the series rather than a cap it outgrew."""
    with pytest.raises(KeyError, match="ja"):
        question_distribution(
            _aggregate(
                "001f",
                {
                    "en": {
                        "en": _cell({"apple": 1}),
                        "ja": _cell({"apple": 1}),
                        "pl": _cell({"banana": 1}),
                    }
                },
            ),
            _TITLE,
            {"en": _WHEEL[0], "pl": _WHEEL[1]}.__getitem__,
        )


def _faceted() -> Drawn:
    """Three languages over two panels, one of them asked in two languages only."""
    return panels(
        cells={
            "en schema": {
                "en": _cell({"apple": 3, "banana": 1}),
                "pl": _cell({"apple": 1, "banana": 3}),
                "ja": _cell({"apple": 2, "banana": 2}),
            },
            "native schema": {
                "pl": _cell({"apple": 2, "banana": 2}),
                "ja": _cell({"banana": 4}),
            },
        },
        title="answer distribution by schema",
        series_color=_palette(),
    )


def test_a_panel_folds_into_the_column_name_the_table_is_keyed_by() -> None:
    """The table contract is unchanged by faceting: one row per category still,
    and a series that a panel never asked contributes no column to it."""
    drawn = _faceted()

    assert drawn.columns == [
        "en / en schema",
        "pl / en schema",
        "pl / native schema",
        "ja / en schema",
        "ja / native schema",
    ]
    assert [row["label"] for row in drawn.rows] == ["banana", "apple"]
    assert [cell["value"] for cell in _cells(drawn, "apple")] == [
        0.75,
        0.25,
        0.5,
        0.5,
        0.0,
    ]


def test_every_panel_is_read_against_one_shared_scale() -> None:
    """Panels sit over one category axis to be read down, so a value axis that
    differed between them would make every cross-panel comparison a lie."""
    figure = _faceted().figure

    assert len(figure.axes) == 2
    assert len({axes.get_ylim() for axes in figure.axes}) == 1


def test_every_panel_is_pictured_and_only_the_bottom_one_is_worded(
    tmp_path: Path,
) -> None:
    """A picture is what a category is found by, and a panel a reader has to trace
    down to the bottom of the stack to identify is a panel drawn without an axis.
    The word is what only needs saying once, so it is the word the panels above
    spend on their columns instead."""
    icon = tmp_path / "fruit.png"
    imsave(icon, np.zeros((8, 8, 4)))

    figure = panels(
        cells={
            "001a order": {"en": _cell({"apple": 3, "banana": 1})},
            "001b order": {"en": _cell({"apple": 1, "banana": 3})},
        },
        title="answer distribution by option order",
        series_color=_palette(),
        category_icon=lambda _: icon,
    ).figure

    top, bottom = figure.axes
    assert [label.get_text() for label in bottom.get_xticklabels()] == [
        "apple",
        "banana",
    ]
    assert not any(label.get_visible() for label in top.get_xticklabels())
    assert _pictured_count(bottom) == 2
    assert _pictured_count(top) == 2


def test_a_faceted_figure_is_keyed_once_for_every_panel_it_stacks() -> None:
    """One series means the same thing in every panel, so a key per panel would
    be the same key drawn n times, and its own title names each panel instead."""
    figure = _faceted().figure

    keyed = [axes for axes in figure.axes if axes.get_legend() is not None]
    assert len(keyed) == 1
    assert figure.legends == []
    legend = keyed[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["en", "pl", "ja"]
    assert [axes.get_title(loc="left") for axes in figure.axes] == [
        "en schema",
        "native schema",
    ]


def test_a_series_a_panel_never_asked_leaves_its_slot_empty() -> None:
    """A language keeps one place in every panel, which is what lets a category
    be read down the stack. The panel that never asked it draws nothing there
    rather than closing up, so the ragged design shows rather than misleads."""
    figure = _faceted().figure

    top, bottom = figure.axes
    apple = sorted({round(float(_middle(patch)), 6) for patch in _plotted(top)})
    below = sorted({round(float(_middle(patch)), 6) for patch in _plotted(bottom)})
    assert len(apple) == 6
    assert len(below) == 4
    assert set(below) < set(apple)


def test_every_other_category_is_banded_down_the_panels_it_is_read_over(
    tmp_path: Path,
) -> None:
    """A column of fruits several panels tall is read by holding one fruit and
    looking down it, so the shading belongs to the category rather than to any
    panel: one band behind the whole stack rather than one drawn inside each and
    broken between them. It runs from the top panel's ceiling to the foot of the
    figure, which is where the page's own caption picks the column back up."""
    icon = tmp_path / "fruit.png"
    imsave(icon, np.zeros((8, 8, 4)))

    figure = panels(
        cells={
            "001a order": {"en": _cell({"apple": 3, "banana": 1})},
            "001b order": {"en": _cell({"apple": 1, "banana": 3})},
        },
        title="answer distribution by option order",
        series_color=_palette(),
        category_icon=lambda _: icon,
    ).figure
    top, _ = figure.axes

    banded = [artist for artist in figure.artists if isinstance(artist, Rectangle)]
    assert len(banded) == 1
    assert to_hex(banded[0].get_facecolor()) == FRAME
    assert banded[0].get_x() == -0.5
    assert banded[0].get_width() == 1.0
    assert banded[0].get_y() == 0.0
    assert banded[0].get_y() + banded[0].get_height() == pytest.approx(
        top.get_position().y1
    )
    assert not any(
        isinstance(patch, Rectangle) for axes in figure.axes for patch in axes.patches
    )


def test_the_first_panel_is_named_on_the_same_line_as_its_key() -> None:
    """A panel's name and the key that reads its colors are one row of chrome and
    not two, since a key hung under the name spends a second row of height saying
    nothing the first row does not."""
    figure = _faceted().figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    keyed = figure.axes[0]
    legend = keyed.get_legend()
    assert legend is not None
    name = _named(keyed, "en schema")
    assert name.get_window_extent(renderer).y0 == pytest.approx(
        legend.get_texts()[0].get_window_extent(renderer).y0, abs=1.0
    )


def _named(axes: Any, name: str) -> Text:
    """The text one panel is named by, which is a title written off to its left."""
    return next(
        child
        for child in axes.get_children()
        if isinstance(child, Text) and child.get_text() == name
    )


def test_a_panel_is_named_in_a_weight_that_tells_it_from_the_axis_around_it() -> None:
    """A panel's name is the heading of a row of the stack, and it sits in the same
    size as the axis label beside it. Weight is what separates the two, so a reader
    scanning down the figure finds the headings without reading anything else."""
    figure = _faceted().figure

    named = [_named(axes, axes.get_title(loc="left")) for axes in figure.axes]
    assert [name.get_fontweight() for name in named] == ["semibold", "semibold"]


def test_every_panel_but_the_first_is_ruled_off_from_the_one_above_it() -> None:
    """The rule belongs to the panel under it, naming where one row of the stack
    ends and the next begins. The first needs none: the figure's own title and its
    key already open the stack."""
    figure = _faceted().figure

    top, bottom = figure.axes
    assert _rules(top) == []
    assert [line.get_color() for line in _rules(bottom)] == [FRAME]


def _middle(patch: Patch) -> float:
    vertices = patch.get_path().vertices
    return float((vertices[:, 0].max() + vertices[:, 0].min()) / 2)


def _plotted(axes: Any) -> list[PathPatch]:
    return [patch for patch in axes.patches if isinstance(patch, PathPatch)]


def _rules(axes: Any) -> list[Line2D]:
    return [child for child in axes.get_children() if isinstance(child, Line2D)]


def _pictured_count(axes: Any) -> int:
    return sum(1 for child in axes.get_children() if isinstance(child, AnnotationBbox))


def test_a_chart_may_label_its_own_arms_across_questions() -> None:
    """An order comparison names its own series, since the arms it draws are
    identical in schema and language and differ only by the question asked."""
    drawn = distribution(
        cells={
            "001a order": _cell({"apple": 3, "banana": 1}),
            "001b order": _cell({"apple": 1, "banana": 3}),
        },
        title="answer distribution by option order (en)",
        series_color=_palette(),
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
        series_color=_palette(),
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
        series_color=_palette(),
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
        series_color=_palette(),
        categories=["1", "2", "3"],
        horizontal=True,
    ).figure.axes[0]

    assert axes.get_ylim()[0] > axes.get_ylim()[1]


def test_a_horizontal_chart_writes_its_values_past_the_end_of_each_bar() -> None:
    """A bar running along x has no cap to sit over, and nothing crowds its side."""
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        series_color=_palette(),
        categories=["1", "2", "3"],
        horizontal=True,
    )

    written = drawn.figure.axes[0].texts
    assert sorted(text.get_text() for text in written) == ["20%", "80%"]
    assert {text.get_rotation() for text in written} == {0.0}
    assert {text.get_horizontalalignment() for text in written} == {"left"}


def test_a_horizontal_chart_stands_its_reference_across_the_bars() -> None:
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        series_color=_palette(),
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
        series_color=_palette(),
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
        series_color=_palette(),
        categories=["1", "2", "3"],
    )

    axes = drawn.figure.axes[0]
    flat = [patch for patch in axes.patches if _tall(patch) == 0.0]
    assert len(flat) == 1

    mark = flat[0]
    column = next(patch for patch in axes.patches if _tall(patch) > 0.0)
    assert to_hex(mark.get_edgecolor()) == _WHEEL[0].lower()
    assert mark.get_facecolor()[3] == 0.0
    assert mark.get_linestyle() != "solid"
    assert _wide(mark) == pytest.approx(_wide(column))


def test_a_category_an_arm_never_picked_is_left_to_its_mark_to_say_so() -> None:
    """The dotted footprint already says this arm picked none, in the arm's own
    color, so writing 0% over it says the same thing twice at the width of a
    number. The zero is still in the table the figure carries."""
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        series_color=_palette(),
        categories=["1", "2", "3"],
    )

    assert sorted(text.get_text() for text in drawn.figure.axes[0].texts) == [
        "20%",
        "80%",
    ]
    assert [cell["written"] for cell in _cells(drawn, "2")] == ["0.00%"]


def test_a_chart_may_write_out_the_zero_its_mark_stands_for() -> None:
    """One chart has to show a reader what the mark means before the rest can
    lean on it, so writing the zero out is the chart's own call to make."""
    drawn = distribution(
        cells={"en": _cell({"1": 4, "3": 1})},
        title="answer distribution by position",
        series_color=_palette(),
        categories=["1", "2", "3"],
        zeros_written=True,
    )

    assert sorted(text.get_text() for text in drawn.figure.axes[0].texts) == [
        "0%",
        "20%",
        "80%",
    ]


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
        series_color=_palette(),
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
        series_color=_palette(),
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
        series_color=_palette(),
    )

    written = drawn.figure.axes[0].texts
    assert sorted(text.get_text() for text in written) == ["0.5%", "99.5%"]


def test_a_share_too_fine_for_one_decimal_keeps_a_second() -> None:
    """One in two thousand is not the nothing that 0.0% would call it."""
    drawn = distribution(
        cells={"en": _cell({"apple": 1999, "banana": 1})},
        title="answer distribution (en)",
        series_color=_palette(),
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
        series_color=_palette(),
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
        series_color=_palette(),
    )

    drawn_labels = sorted(text.get_text() for text in drawn.figure.axes[0].texts)
    tabled = sorted(cell["written"] for row in drawn.rows for cell in row["cells"])
    assert drawn_labels == ["2%", "98%"]
    assert tabled == ["1.60%", "98.40%"]


def test_a_table_writes_every_share_to_the_same_decimals() -> None:
    """A column of numbers is read down, so each is written to the width of the
    last. What a drawn label spends on being read at a glance the table spends on
    lining up, and the count beside a share is what says a category was picked."""
    drawn = distribution(
        cells={"en": _cell({"apple": 4999, "banana": 1})},
        title="answer distribution (en)",
        series_color=_palette(),
    )

    apple, banana = _cells(drawn, "apple")[0], _cells(drawn, "banana")[0]
    assert apple["written"] == "99.98%"
    assert banana["written"] == "0.02%"


def test_a_tabled_estimate_is_written_in_the_unit_its_chart_plots() -> None:
    """A share and a count cannot share a formatter in the table either."""
    drawn = estimates(
        cells={"001a en": 1.0594},
        title="how many of the 10 fruits each arm was choosing between",
        series_color=_palette(),
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300},
        intervals={"001a en": (1.0, 1.119)},
        unit=COUNT,
    )

    cell = drawn.rows[0]["cells"][0]
    assert cell["written"] == "1.06"
    assert cell["written_interval"] == "1.00–1.12"


def _pooled(cells: dict[str, int], total: int = 34000) -> Any:
    """A pooled table the way an experiment asks for one, over counts it hands in."""
    return table(
        cells=cells,
        total=total,
        title="Table 1.1: how many times was each fruit picked",
        row_label="fruit",
        count_column="times picked",
        share_column="share of all answers",
    )


def test_a_pooled_count_is_tabled_beside_the_share_it_works_out_to() -> None:
    """A count answers how many and a share answers how much of the whole, and
    reading one against the other is what the two columns are for."""
    tabled = _pooled({"lychee": 18160, "orange": 0})

    assert tabled.unit == "count"
    assert tabled.columns == ["times picked", "share of all answers"]
    assert [row["label"] for row in tabled.rows] == ["lychee", "orange"]
    assert tabled.rows[0]["cells"] == [
        {"value": 18160, "count": 18160, "n": 34000, "written": "18160"},
        {"value": 0.534118, "count": 18160, "n": 34000, "written": "53.4%"},
    ]


def test_one_answer_in_a_pool_this_size_is_still_not_written_as_a_zero() -> None:
    """A share widens as far as it has to, so 1 in 34 000 never reads as none."""
    tabled = _pooled({"banana": 1, "orange": 0})

    assert [row["cells"][1]["written"] for row in tabled.rows] == ["0.003%", "0.0%"]


def test_a_pooled_share_is_tabled_without_an_interval_around_it() -> None:
    """Arms built to answer differently are not draws from one distribution."""
    tabled = _pooled({"lychee": 18160})

    assert not [cell for cell in tabled.rows[0]["cells"] if "written_interval" in cell]


def test_a_table_may_picture_a_row_the_way_its_experiment_illustrates_it() -> None:
    """A row points at the same file a chart draws that category with, and the
    stage that writes artifacts out is the one that puts it where the site reads."""
    lychee = Path("emoji_u1f330.png")
    tabled = table(
        cells={"lychee": 4, "other": 1},
        total=5,
        title="Table 1.1: how many times was each fruit picked",
        row_label="fruit",
        count_column="times picked",
        share_column="share of all answers",
        row_icon={"lychee": lychee}.get,
    )

    assert tabled.rows[0]["icon"] == lychee
    assert "icon" not in tabled.rows[1]


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
        series_color=_palette(),
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
        series_color=_palette(),
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

    written = _values(figure.axes[0])
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
    for text in _values(axes):
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
        series_color=_palette(),
    )

    assert {text.get_rotation() for text in _values(turned.figure.axes[0])} == {90.0}
    assert {text.get_rotation() for text in _values(flat.figure.axes[0])} == {0.0}
    assert _headroom(turned) > _headroom(flat)


def _headroom(drawn: Drawn) -> float:
    axes = drawn.figure.axes[0]
    tallest = max(_tall(patch) for patch in axes.patches)
    return axes.get_ylim()[1] / tallest


def test_values_written_over_one_group_never_collide(languages: Aggregate) -> None:
    """Three arms picking the same fruit is the finding, not an edge case, so a
    value that cannot clear the one beside it turns rather than overlapping it."""
    figure = question_distribution(languages, _TITLE, _palette()).figure
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
        series_color=_palette(),
    )

    written = drawn.figure.axes[0].texts
    assert sorted(text.get_text() for text in written) == ["25%", "75%"]
    assert {text.get_rotation() for text in written} == {0.0}


def _every_form() -> list[Drawn]:
    """One figure of every form a chart takes, since a width has to hold for all."""
    return [
        distribution(
            cells={"en": _cell({"apple": 3, "banana": 1})},
            title="answer distribution (en)",
            series_color=_palette(),
        ),
        distribution(
            cells={"en": _cell({"1": 4, "3": 1})},
            title="answer distribution by position",
            series_color=_palette(),
            categories=["1", "2", "3"],
            horizontal=True,
        ),
        estimates(
            cells={"001a en": 1.0594},
            title="how many of the 10 fruits each arm was choosing between",
            series_color=_palette(),
            value_label="effective choices",
            row_label="arm",
            counts={"001a en": 300},
            intervals={"001a en": (1.0, 1.119)},
            unit=COUNT,
        ),
    ]


def test_every_chart_is_drawn_at_the_width_the_page_gives_it() -> None:
    """The site renders a chart at the width of its column whatever the SVG says,
    so a figure drawn wider is a figure whose type the browser shrinks. One width
    per canvas is what makes a declared point size the size read on the page,
    and what keeps two charts on one article the same size as each other."""
    for canvas in (WIDE, NARROW):
        with styled(canvas):
            drawn = _every_form()

        widths = {one.figure.get_size_inches()[0] for one in drawn}
        assert widths == {canvas.width_in}


def test_the_narrow_canvas_lays_a_chart_out_again_rather_than_shrinking_it(
    tmp_path: Path,
) -> None:
    """A phone is served its own drawing, not the wide one scaled down, so the
    narrow canvas reaches its own layout decisions. A name that sits beside its
    picture at the width a laptop gives it has no room to at half of it, and drops
    under the picture there; the point sizes stay where they were declared."""
    icon = tmp_path / "fruit.png"
    imsave(icon, np.zeros((8, 8, 4)))
    cells = {"en": _cell({"pomegranate": 3, "strawberry": 1, "watermelon": 1})}

    def draw() -> Drawn:
        return distribution(
            cells=cells,
            title="answer distribution",
            series_color=_palette(),
            category_icon=lambda _: icon,
        )

    with styled(WIDE):
        wide = draw().figure
    with styled(NARROW):
        narrow = draw().figure

    assert _beside_its_picture(wide)
    assert not _beside_its_picture(narrow)
    assert {label.get_fontsize() for label in narrow.axes[0].get_xticklabels()} == {
        label.get_fontsize() for label in wide.axes[0].get_xticklabels()
    }


def _beside_its_picture(figure: Figure) -> bool:
    """Whether the first category's word shares a line with the picture naming it."""
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    word = figure.axes[0].get_xticklabels()[0].get_window_extent(renderer)
    shown = _pictures(figure)[0].get_window_extent(renderer)

    return abs((shown.y0 + shown.y1) - (word.y0 + word.y1)) < _INLINE_SLACK_PX


def test_a_title_too_long_for_the_figure_is_broken_over_lines() -> None:
    """A figure used to widen until its title fit, which is the same knob that
    made a chart small on the page. At a fixed width a long title has to wrap."""
    written = "Chart 1.7: how many of the 10 fruits each arm was choosing between"
    drawn = estimates(
        cells={"001a en": 1.0594},
        title=written,
        series_color=_palette(),
        value_label="effective choices",
        row_label="arm",
        counts={"001a en": 300},
        intervals={"001a en": (1.0, 1.119)},
        unit=COUNT,
    )
    figure = drawn.figure
    FigureCanvasAgg(figure)
    figure.canvas.draw()

    title = figure.texts[0]
    assert "\n" in title.get_text()
    assert drawn.title == written
    assert (
        title.get_window_extent(figure.canvas.get_renderer()).x1
        <= figure.get_size_inches()[0] * figure.dpi
    )


def test_a_value_axis_ends_just_above_the_data_rather_than_at_full_scale() -> None:
    """A 0-100% axis when nothing clears 35% spends most of the plot on white space."""
    drawn = distribution(
        cells={"en": _cell({"apple": 2, "banana": 3, "grape": 5})},
        title="answer distribution (en)",
        series_color=_palette(),
    )

    assert drawn.figure.axes[0].get_ylim()[1] < 1.0


def test_short_categories_are_not_turned_when_they_already_fit() -> None:
    drawn = distribution(
        cells={"en": _cell({"1": 4, "2": 1})},
        title="answer distribution by position",
        series_color=_palette(),
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
        series_color=_palette(),
    )

    unpicked = _cells(drawn, "apple")[0]
    assert unpicked["value"] == 0.0
    assert unpicked["lo"] == 0.0
    assert unpicked["hi"] > 0.4
