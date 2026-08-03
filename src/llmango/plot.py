"""Everything an experiment needs to draw a chart, and how a figure is written."""

from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as DrawPath
from matplotlib.textpath import text_to_path
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Affine2D, ScaledTranslation
from matplotlib.typing import RcKeyType

from llmango.aggregate import Aggregate, Distribution
from llmango.spec import FREE_TEXT, OTHER_CATEGORY
from llmango.stats import wilson_interval

ARM_COLORS = ("#0072B2", "#D55E00", "#009E73")
INK = "#767676"

LIGHT_SURFACE = "#f9f9f7"
DARK_SURFACE = "#0d0d0d"

TEXT_FAMILY = "DejaVu Sans"

_ARTICLE_WIDTH_IN = 6.58
_TITLE_WIDTH_IN = 4.9
_COLUMN_IN = 0.17
_BAR_IN = 0.14
_COLUMN_GAP_IN = 0.04
_GROUP_PAD_IN = 0.08
_MAX_HEIGHT_IN = 12.0
_PLOT_HEIGHT_IN = 2.5
_BAR_CHROME_IN = 1.1
_DOT_ROW_IN = 0.3
_LEGEND_IN = 0.34
_LABEL_CHAR_IN = 0.075
_MAX_COLUMN_IN = 0.42
_CORNER_PX = 4.0
_MIN_MARK_PX = 3.0
_ZERO_MARK_PT = 2.0
_ZERO_DOT_PT = 2.0
_ZERO_GAP_PT = 3.0
_HEADROOM = 1.09
_BAR_HEADROOM = 1.18
_PLOT_GUTTER_IN = 1.0
_PLOT_BAND_IN = 1.7
_LABEL_ROTATION = 45.0
_SHORT_LABEL = 3
_FINE_SHARE = 0.1
_WHOLE_SHARE = 1.0
_APART_POINTS = 1.0
_VALUE_PT = 9.0
_VALUE_GAP_PT = 2.5
_DOT_PT = 9.0
_INTERVAL_PT = 2.2
_TICK_PT = 11.5
_AXIS_LABEL_PT = 11.0
_LEGEND_PT = 11.0
_TITLE_PT = 13.0
_BODY_PT = 12.0
_TICK_PAD_PT = 4.5
_ICON_PT = 15.5
_ICON_GAP_PT = 2.75
_ICON_DROP_PT = 10.5
_ICON_SOURCE_PX = 128.0
_EXPORT_DPI = _ICON_SOURCE_PX * 72.0 / _ICON_PT

_STYLE: dict[RcKeyType, Any] = {
    "svg.hashsalt": "llmango",
    "svg.fonttype": "path",
    "font.family": TEXT_FAMILY,
    "font.size": _BODY_PT,
    "figure.dpi": 96,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": _TICK_PT,
    "ytick.labelsize": _TICK_PT,
    "axes.titlesize": _TITLE_PT,
}

Row = dict[str, Any]
CategoryLabel = Callable[[str], str]
CategoryIcon = Callable[[str], Path | None]
SchemaLabel = Callable[[str], str]


def _write_share(value: float) -> str:
    """Write a share to one decimal, never as the zero a picked category is not."""
    written = f"{value * 100:.1f}%"

    return f"{value * 100:.2f}%" if value > 0 and written == "0.0%" else written


def _write_column(value: float, group: list[float]) -> str:
    """Write a share over its own column, keeping only the decimals it needs.

    A column carries a number to be read at a glance, and a decimal it does not
    need is width taken from the column beside it. Two things still earn one: a
    share near enough an end that a whole number would round it into a certainty
    it does not have, and a neighbour in its own group close enough that both
    would otherwise be written the same. 99.5% is not a unanimous arm and 0.5%
    is not a category never picked, so neither is written as though it were.
    """
    percent = value * 100
    if percent <= 0:
        return "0%"
    if percent >= 100:
        return "100%"

    edge = min(percent, 100 - percent)
    if edge < _FINE_SHARE:
        return f"{percent:.2f}%"
    if edge < _WHOLE_SHARE or _crowded(value, group):
        return f"{percent:.1f}%"

    return f"{percent:.0f}%"


def _crowded(value: float, group: list[float]) -> bool:
    """Whether a column shares its group with one too near to tell apart whole."""
    near = sum(1 for other in group if abs(other - value) * 100 < _APART_POINTS)

    return near > 1


def _write_share_tick(value: float) -> str:
    """Write a share along an axis, whose round scale has no use for a decimal."""
    return f"{round(value * 100, 1):g}%"


def _write_count(value: float) -> str:
    """Write a count, dropping the decimals a whole number has no use for."""
    return f"{round(value, 2):g}"


def _write_count_column(value: float, _: list[float]) -> str:
    """Write a count over its column, which is a count written the one way there is."""
    return _write_count(value)


@dataclass(frozen=True)
class Unit:
    """What a chart's numbers are, since not every chart plots a share."""

    name: str
    write: Callable[[float], str]
    write_tick: Callable[[float], str]
    write_column: Callable[[float, list[float]], str]


SHARE = Unit("share", _write_share, _write_share_tick, _write_column)
COUNT = Unit("count", _write_count, _write_count, _write_count_column)


@dataclass(frozen=True, order=True)
class Arm:
    """One comparable series: a question under one schema and language."""

    schema: str
    lang: str


@dataclass(frozen=True)
class Series:
    """One drawn series: its column values and the labels written over them."""

    label: str
    color: str
    values: list[float]
    labels: list[str]


@dataclass(frozen=True)
class Estimate:
    """One named number and the interval around it, which is what a summary plots."""

    label: str
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class Drawn:
    """One finished figure, and the numbers behind it the site puts in a table."""

    figure: Figure
    title: str
    row_label: str
    unit: str
    columns: list[str]
    rows: list[Row]


@dataclass(frozen=True)
class ChartDef:
    """One chart an experiment declares: how it is cited, what it reads and draws."""

    name: str
    number: str
    title: str
    questions: tuple[str, ...]
    draw: Callable[[dict[str, Aggregate], str], "Drawn"]

    def numbered_title(self) -> str:
        """The title the figure carries, opening with the number a page cites it by."""
        return f"Chart {self.number}: {self.title}"


def distribution(
    cells: dict[str, Distribution],
    title: str,
    category_label: CategoryLabel | None = None,
    category_icon: CategoryIcon | None = None,
    row_label: str = "category",
    categories: list[str] | None = None,
    reference: float | None = None,
    horizontal: bool = False,
) -> Drawn:
    """Draw labeled arms' category shares, and return the numbers behind them."""
    _refuse_beyond_palette(cells, title)
    shown_categories = categories or _categories(cells.values())
    shares = [
        [_share(cell, category) for category in shown_categories]
        for cell in cells.values()
    ]
    series = [
        Series(label=label, color=ARM_COLORS[index], values=values, labels=labels)
        for index, (label, values, labels) in enumerate(
            zip(cells, shares, _written_columns(shares), strict=True)
        )
    ]
    written_categories = _shown(shown_categories, category_label)
    value_label = "share of valid answers"
    legend = len(cells) > 1
    figure = (
        bars(
            category_labels=written_categories,
            series=series,
            title=title,
            value_label=value_label,
            unit=SHARE,
            legend=legend,
            reference=reference,
        )
        if horizontal
        else columns(
            category_labels=written_categories,
            series=series,
            title=title,
            value_label=value_label,
            unit=SHARE,
            legend=legend,
            category_icons=_pictured(shown_categories, category_icon),
            reference=reference,
        )
    )

    return Drawn(
        figure=figure,
        title=title,
        row_label=row_label,
        unit=SHARE.name,
        columns=list(cells),
        rows=_rows(shown_categories, series, list(cells.values())),
    )


def question_distribution(
    aggregate: Aggregate,
    title: str,
    schema_label: SchemaLabel | None = None,
    category_label: CategoryLabel | None = None,
    category_icon: CategoryIcon | None = None,
    categories: list[str] | None = None,
) -> Drawn:
    """Draw one question's arms, labeled by whichever of its dimensions varies."""
    arms = _arms(aggregate["distributions"])
    labels = _labels(list(arms), schema_label)

    return distribution(
        cells=dict(zip(labels, arms.values(), strict=True)),
        title=title,
        category_label=category_label,
        category_icon=category_icon,
        categories=categories,
    )


def summary(
    cells: dict[str, float],
    title: str,
    value_label: str,
    row_label: str,
    counts: dict[str, int],
    intervals: dict[str, tuple[float, float]],
    unit: Unit = SHARE,
    reference: float | None = None,
) -> Drawn:
    """Draw one number per named thing, which is what a cross-question chart has."""
    plotted = _with_intervals(cells, intervals)
    values = [estimate.value for estimate in plotted]
    series = [
        Series(
            label=value_label,
            color=ARM_COLORS[0],
            values=values,
            labels=[unit.write_column(value, values) for value in values],
        )
    ]
    figure = columns(
        category_labels=[estimate.label for estimate in plotted],
        series=series,
        title=title,
        value_label=value_label,
        unit=unit,
        legend=False,
        reference=reference,
    )

    return Drawn(
        figure=figure,
        title=title,
        row_label=row_label,
        unit=unit.name,
        columns=[value_label],
        rows=_estimate_rows(plotted, counts, unit),
    )


def estimates(
    cells: dict[str, float],
    title: str,
    value_label: str,
    row_label: str,
    counts: dict[str, int],
    intervals: dict[str, tuple[float, float]],
    unit: Unit = SHARE,
    floor: float = 0.0,
) -> Drawn:
    """Draw one number per named thing as a dot on the interval around it.

    A column asks to be read from its base, so it can only stand on the zero its
    statistic may not have. A dot stands nowhere, which lets the axis start at the
    floor the statistic actually has and spend its whole length on the range the
    numbers occupy. What that buys is the overlap between two intervals, which is
    what says whether two arms differ at all.
    """
    plotted = _with_intervals(cells, intervals)

    return Drawn(
        figure=dots(plotted, title, value_label, unit, floor),
        title=title,
        row_label=row_label,
        unit=unit.name,
        columns=[value_label],
        rows=_estimate_rows(plotted, counts, unit),
    )


def _with_intervals(
    cells: dict[str, float], intervals: dict[str, tuple[float, float]]
) -> list[Estimate]:
    """Pair every named number with its interval, in the order they were given."""
    return [Estimate(name, value, *intervals[name]) for name, value in cells.items()]


def _estimate_rows(
    plotted: list[Estimate], counts: dict[str, int], unit: Unit
) -> list[Row]:
    """Describe every plotted estimate, one row per name, for the table view."""
    return [
        {
            "label": estimate.label,
            "cells": [
                {
                    "value": estimate.value,
                    "n": counts[estimate.label],
                    "lo": estimate.low,
                    "hi": estimate.high,
                    **_written(unit, estimate.value, estimate.low, estimate.high),
                }
            ],
        }
        for estimate in plotted
    ]


def _written(unit: Unit, value: float, low: float, high: float) -> dict[str, str]:
    """Write a cell's number and its interval, so the site prints and never formats."""
    return {
        "written": unit.write(value),
        "written_interval": f"{unit.write(low)}–{unit.write(high)}",
    }


def styled() -> AbstractContextManager[None]:
    """Apply the chart style every chart is drawn and saved under."""
    return matplotlib.rc_context(_STYLE)


def save(figure: Figure, path: Path) -> Path:
    """Save one figure as the transparent, reproducible SVG the site embeds."""
    figure.savefig(
        path,
        format="svg",
        dpi=_EXPORT_DPI,
        transparent=True,
        metadata={"Date": None},
    )

    return path


def columns(
    category_labels: list[str],
    series: list[Series],
    title: str,
    value_label: str,
    unit: Unit,
    legend: bool,
    category_icons: list[Path | None] | None = None,
    reference: float | None = None,
) -> Figure:
    """Draw one vertical bar chart, grouped when it carries several series."""
    count = len(series)
    pitch = count * _COLUMN_IN + (count - 1) * _COLUMN_GAP_IN + _GROUP_PAD_IN
    icons: list[Path | None] = category_icons or [None] * len(category_labels)
    pictured = _any_icon(icons)
    turned = not pictured and _needs_turning(category_labels)
    slots = max(len(category_labels), 1)
    turned_values = not _values_fit(series, pitch, slots)
    figure = Figure(
        figsize=(
            _ARTICLE_WIDTH_IN,
            _height(category_labels, legend, turned, pictured),
        ),
        layout="constrained",
    )
    axes = figure.add_subplot()
    offsets = _offsets(count, _COLUMN_IN, pitch)

    _frame(
        axes,
        category_labels,
        title,
        value_label,
        unit,
        legend,
        turned,
        _top(series, reference, _headroom(series, turned_values)),
    )
    if reference is not None:
        _reference(axes, reference, horizontal=False)
    _icons(figure, axes, category_labels, icons)
    _annotate_above(axes, series, offsets, turned_values)
    if legend:
        _legend(axes, series)

    _settle(figure)
    thickness = _thickness(axes, figure, _COLUMN_IN / pitch, horizontal=False)
    thickness_pt = _in_points(axes, figure, thickness, horizontal=False)
    radius_x, radius_y = _in_data_units(axes, _CORNER_PX)
    _, shortest = _in_data_units(axes, _MIN_MARK_PX)
    for entry, offset in zip(series, offsets, strict=True):
        for slot, value in enumerate(entry.values):
            if value > 0:
                axes.add_patch(
                    PathPatch(
                        _bar_path(
                            max(value, shortest),
                            slot + offset,
                            thickness,
                            radius_x,
                            radius_y,
                        ),
                        facecolor=entry.color,
                        edgecolor="none",
                    )
                )
            else:
                axes.add_patch(
                    _zero_mark(
                        slot + offset,
                        thickness,
                        thickness_pt,
                        entry.color,
                        horizontal=False,
                    )
                )

    return figure


def bars(
    category_labels: list[str],
    series: list[Series],
    title: str,
    value_label: str,
    unit: Unit,
    legend: bool,
    reference: float | None = None,
) -> Figure:
    """Draw one horizontal bar chart, for categories too many to stand along x."""
    count = len(series)
    pitch = count * _BAR_IN + (count - 1) * _COLUMN_GAP_IN + _GROUP_PAD_IN
    slots = max(len(category_labels), 1)
    figure = Figure(
        figsize=(_ARTICLE_WIDTH_IN, _bar_height(slots, pitch, legend)),
        layout="constrained",
    )
    axes = figure.add_subplot()
    offsets = _offsets(count, _BAR_IN, pitch)

    _horizontal_frame(
        axes,
        category_labels,
        title,
        value_label,
        unit,
        0.0,
        _top(series, reference, _BAR_HEADROOM),
        legend,
    )
    if reference is not None:
        _reference(axes, reference, horizontal=True)
    _annotate_beside(axes, series, offsets)
    if legend:
        _legend(axes, series)

    _settle(figure)
    thickness = _thickness(axes, figure, _BAR_IN / pitch, horizontal=True)
    thickness_pt = _in_points(axes, figure, thickness, horizontal=True)
    radius_x, radius_y = _in_data_units(axes, _CORNER_PX)
    shortest, _ = _in_data_units(axes, _MIN_MARK_PX)
    for entry, offset in zip(series, offsets, strict=True):
        for slot, value in enumerate(entry.values):
            if value > 0:
                axes.add_patch(
                    PathPatch(
                        _laid_on_its_side(
                            _bar_path(
                                max(value, shortest),
                                slot + offset,
                                thickness,
                                radius_y,
                                radius_x,
                            )
                        ),
                        facecolor=entry.color,
                        edgecolor="none",
                    )
                )
            else:
                axes.add_patch(
                    _zero_mark(
                        slot + offset,
                        thickness,
                        thickness_pt,
                        entry.color,
                        horizontal=True,
                    )
                )

    return figure


def dots(
    plotted: list[Estimate], title: str, value_label: str, unit: Unit, floor: float
) -> Figure:
    """Draw every estimate as a dot on its own row, its interval the line under it."""
    slots = max(len(plotted), 1)
    end = _end(plotted, floor)
    labels = [estimate.label for estimate in plotted]
    figure = Figure(
        figsize=(_ARTICLE_WIDTH_IN, _bar_height(slots, _DOT_ROW_IN, legend=False)),
        layout="constrained",
    )
    axes = figure.add_subplot()

    _horizontal_frame(
        axes,
        labels,
        title,
        value_label,
        unit,
        floor,
        end,
        legend=False,
    )
    for slot, estimate in enumerate(plotted):
        axes.plot(
            [estimate.low, estimate.high],
            [slot, slot],
            color=ARM_COLORS[0],
            linewidth=_INTERVAL_PT,
            solid_capstyle="round",
        )
        axes.plot(
            [estimate.value],
            [slot],
            marker="o",
            markersize=_DOT_PT,
            color=ARM_COLORS[0],
            linestyle="none",
        )
        axes.annotate(
            unit.write(estimate.value),
            (estimate.high, slot),
            xytext=(_DOT_PT / 2.0 + 4.0, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=_VALUE_PT,
            color=INK,
        )

    return figure


def _end(plotted: list[Estimate], floor: float) -> float:
    """End the value axis past the widest interval, leaving room for its label."""
    span = max([floor] + [estimate.high for estimate in plotted]) - floor

    return floor + span * _BAR_HEADROOM if span > 0 else floor + 1.0


def _offsets(count: int, thickness_in: float, pitch: float) -> list[float]:
    """Where each series sits inside a category's slot, the group centred on it."""
    step = (thickness_in + _COLUMN_GAP_IN) / pitch

    return [(index - (count - 1) / 2) * step for index in range(count)]


def _bar_height(slots: int, pitch: float, legend: bool) -> float:
    """Grow a horizontal chart down the page with every bar it stacks, then cap it."""
    wanted = _BAR_CHROME_IN + slots * pitch + (_LEGEND_IN if legend else 0.0)

    return min(wanted, _MAX_HEIGHT_IN)


def _written_width(label: str, size: float) -> float:
    """How wide a word sits in points, at the size and family it is written in."""
    width, _, _ = text_to_path.get_text_width_height_descent(
        label, FontProperties(family=TEXT_FAMILY, size=size), ismath=False
    )

    return float(width)


def _values_fit(series: list[Series], pitch: float, slots: int) -> bool:
    """Whether every value written flat clears the value written next along the axis."""
    category = (_ARTICLE_WIDTH_IN - _PLOT_GUTTER_IN) / max(slots, 1)
    written = sorted(
        ((slot + offset) * category, _written_width(text, _VALUE_PT) / 72.0)
        for entry, offset in zip(
            series, _offsets(len(series), _COLUMN_IN, pitch), strict=True
        )
        for slot, text in enumerate(entry.labels)
    )

    return all(
        later - earlier >= (earlier_width + later_width) / 2
        for (earlier, earlier_width), (later, later_width) in zip(
            written, written[1:], strict=False
        )
    )


def _headroom(series: list[Series], turned: bool) -> float:
    """Leave the tallest column room for the value standing above it."""
    if not turned:
        return _HEADROOM

    return _HEADROOM + _widest_value(series) / 72.0 / _PLOT_BAND_IN


def _widest_value(series: list[Series]) -> float:
    """How wide in points the longest value any series writes sits."""
    return max(
        (_written_width(text, _VALUE_PT) for entry in series for text in entry.labels),
        default=0.0,
    )


def _top(series: list[Series], reference: float | None, headroom: float) -> float:
    """End the value axis past the longest bar, leaving room for its label."""
    peak = max(
        [reference or 0.0] + [value for entry in series for value in entry.values]
    )

    return peak * headroom if peak > 0 else 1.0


def _thickness(axes: Axes, figure: Figure, wanted: float, horizontal: bool) -> float:
    """Hold a bar to a readable thickness, so few categories do not draw slabs."""
    limits = axes.get_ylim() if horizontal else axes.get_xlim()
    extent = axes.get_window_extent()
    inches = (extent.height if horizontal else extent.width) / figure.dpi
    if inches <= 0:
        return wanted

    return min(wanted, _MAX_COLUMN_IN * abs(limits[1] - limits[0]) / inches)


def _height(
    category_labels: list[str], legend: bool, turned: bool, pictured: bool
) -> float:
    """Leave a figure room for the labels under its axis, turned or written flat."""
    longest = max((len(label) for label in category_labels), default=0)
    hanging = longest * _LABEL_CHAR_IN * 0.71 if turned else 0.0
    band = _icon_band() / 72.0 if pictured else 0.0

    return _PLOT_HEIGHT_IN + hanging + band + (_LEGEND_IN if legend else 0.0)


def _shown(categories: list[str], category_label: CategoryLabel | None) -> list[str]:
    """Name each category the way its experiment writes it on an axis."""
    if category_label is None:
        return list(categories)

    return [category_label(category) for category in categories]


def _pictured(
    categories: list[str], category_icon: CategoryIcon | None
) -> list[Path | None]:
    """Find each category's picture, the way its experiment illustrates one."""
    if category_icon is None:
        return [None] * len(categories)

    return [category_icon(category) for category in categories]


def _refuse_beyond_palette(cells: dict[str, Distribution], title: str) -> None:
    """Refuse a comparison the palette cannot color, since wrapping it would lie."""
    if len(cells) > len(ARM_COLORS):
        raise ValueError(
            f"{title} has {len(cells)} arms but the palette holds {len(ARM_COLORS)}. "
            f"The cap is a property of a transparent export, not of the palette: no "
            f"fourth color clears both page surfaces. Fold the tail into one series, "
            f"split the comparison across charts, or plot a summary instead."
        )


def _arms(distributions: dict[str, dict[str, Distribution]]) -> dict[Arm, Distribution]:
    """Read a question's aggregate as arm -> numbers, in a stable order."""
    return {
        Arm(schema=schema, lang=lang): cell
        for schema, langs in sorted(distributions.items())
        for lang, cell in sorted(langs.items())
    }


def _varies(arms: list[Arm]) -> tuple[bool, bool]:
    """Whether the schema and the language differ across a question's arms."""
    return (
        len({arm.schema for arm in arms}) > 1,
        len({arm.lang for arm in arms}) > 1,
    )


def _labels(arms: list[Arm], schema_label: SchemaLabel | None) -> list[str]:
    """Label each arm by whichever of schema and language varies."""
    written = schema_label or _schema_label
    many_schemas, many_langs = _varies(arms)
    labels: list[str] = []
    for arm in arms:
        label = written(arm.schema)
        if many_schemas and many_langs:
            labels.append(f"{arm.lang} / {label}")
        elif many_schemas:
            labels.append(label)
        else:
            labels.append(arm.lang)

    return labels


def _schema_label(schema: str) -> str:
    """Name a schema arm the way a legend should read it."""
    if schema == FREE_TEXT:
        return "no schema"

    return f"{schema} schema"


def _categories(cells: Iterable[Distribution]) -> list[str]:
    """The categories some arm actually picked, most picked first, 'other' last."""
    totals: Counter[str] = Counter()
    for cell in cells:
        totals.update(cell["counts"])

    return sorted(
        (name for name, total in totals.items() if total > 0),
        key=lambda name: (name == OTHER_CATEGORY, -totals[name], name),
    )


def _share(cell: Distribution, category: str) -> float:
    """One category's share of an arm's valid answers, 0.0 when it picked none."""
    total = cell["n"]
    if not total:
        return 0.0

    return round(cell["counts"].get(category, 0) / total, 4)


def _written_columns(shares: list[list[float]]) -> list[list[str]]:
    """Write every column's share at the precision its own group turns out to need."""
    groups = [list(group) for group in zip(*shares, strict=True)]

    return [
        [_write_column(value, groups[slot]) for slot, value in enumerate(values)]
        for values in shares
    ]


def _rows(
    categories: list[str], series: list[Series], cells: list[Distribution]
) -> list[Row]:
    """Describe every plotted number, one row per category, for the table view."""
    return [
        {
            "label": category,
            "cells": [
                _table_cell(entry, cell, category, index)
                for entry, cell in zip(series, cells, strict=True)
            ],
        }
        for index, category in enumerate(categories)
    ]


def _table_cell(
    entry: Series, cell: Distribution, category: str, index: int
) -> dict[str, Any]:
    """One tabled number: the share drawn, the counts under it and its bounds."""
    count = cell["counts"].get(category, 0)
    value = entry.values[index]
    low, high = wilson_interval(count, cell["n"])

    return {
        "value": value,
        "count": count,
        "n": cell["n"],
        "lo": low,
        "hi": high,
        **_written(SHARE, value, low, high),
    }


def _settle(figure: Figure) -> None:
    """Resolve the layout so the data transform is final."""
    engine = figure.get_layout_engine()
    if engine is not None:
        engine.execute(figure)


def _frame(
    axes: Axes,
    category_labels: list[str],
    title: str,
    value_label: str,
    unit: Unit,
    legend: bool,
    turned: bool,
    top: float,
) -> None:
    """Set up the plot frame: a recessive grid, its own unit and room for labels."""
    axes.set_ylim(0.0, top)
    axes.set_xlim(-0.5, max(len(category_labels), 1) - 0.5)
    axes.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    axes.yaxis.set_major_formatter(_tick_format(unit))
    axes.set_xticks(range(len(category_labels)), labels=category_labels)
    axes.set_ylabel(value_label, fontsize=_AXIS_LABEL_PT)
    axes.grid(axis="y", color=INK, alpha=0.25, linewidth=0.6)
    _bare(axes, title, legend)
    if turned:
        for label in axes.get_xticklabels():
            label.set_rotation(_LABEL_ROTATION)
            label.set_rotation_mode("anchor")
            label.set_horizontalalignment("right")
            label.set_verticalalignment("top")


def _horizontal_frame(
    axes: Axes,
    category_labels: list[str],
    title: str,
    value_label: str,
    unit: Unit,
    start: float,
    end: float,
    legend: bool,
) -> None:
    """Set up a horizontal chart's frame, its categories reading down from the top."""
    axes.set_xlim(start, end)
    axes.set_ylim(-0.5, max(len(category_labels), 1) - 0.5)
    axes.invert_yaxis()
    axes.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    axes.xaxis.set_major_formatter(_tick_format(unit))
    axes.set_yticks(range(len(category_labels)), labels=category_labels)
    axes.set_xlabel(value_label, fontsize=_AXIS_LABEL_PT)
    axes.grid(axis="x", color=INK, alpha=0.25, linewidth=0.6)
    _bare(axes, title, legend)


def _bare(axes: Axes, title: str, legend: bool) -> None:
    """Strip a plot to its centred title and its grid, which both forms share."""
    axes.set_title(_wrapped(title), loc="center", pad=30 if legend else 10)
    axes.set_axisbelow(True)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)


def _wrapped(title: str) -> str:
    """Break a title over lines, since a figure no longer widens to fit one."""
    written = _written_width(title, _TITLE_PT) / 72.0
    if written <= _TITLE_WIDTH_IN:
        return title

    return "\n".join(wrap(title, ceil(len(title) * _TITLE_WIDTH_IN / written)))


def _tick_format(unit: Unit) -> FuncFormatter:
    """Write a value axis's own ticks in the unit that axis plots."""

    def write_tick(value: float, _: int) -> str:
        return unit.write_tick(value)

    return FuncFormatter(write_tick)


def _needs_turning(category_labels: list[str]) -> bool:
    """Turn category names only when they are too long to sit side by side."""
    return max((len(label) for label in category_labels), default=0) > _SHORT_LABEL


def _reference(axes: Axes, value: float, horizontal: bool) -> None:
    """Draw the line a chart is read against, such as a fair die's even spread."""
    chrome: dict[str, Any] = {
        "color": INK,
        "linewidth": 0.9,
        "linestyle": (0, (4, 3)),
        "zorder": 1,
    }
    if horizontal:
        axes.axvline(value, **chrome)
    else:
        axes.axhline(value, **chrome)


def _icons(
    figure: Figure, axes: Axes, category_labels: list[str], icons: list[Path | None]
) -> None:
    """Set each category's picture into its label, immediately before its word."""
    if not _any_icon(icons):
        return

    axes.tick_params(axis="x", pad=_TICK_PAD_PT)
    for written in axes.get_xticklabels():
        written.set_transform(written.get_transform() + _word_shift(figure))
    for slot, icon in enumerate(icons):
        if icon is not None:
            image = imread(icon)
            beside = (
                -(_written_width(category_labels[slot], _TICK_PT) + _ICON_GAP_PT) / 2.0
            )
            axes.add_artist(
                AnnotationBbox(
                    OffsetImage(image, zoom=_ICON_PT / image.shape[0]),
                    (slot, 0.0),
                    xybox=(beside, -_ICON_DROP_PT),
                    xycoords=("data", "axes fraction"),
                    boxcoords="offset points",
                    box_alignment=(0.5, 0.5),
                    pad=0.0,
                    frameon=False,
                    annotation_clip=False,
                )
            )


def _word_shift(figure: Figure) -> ScaledTranslation:
    """Move a word clear of its picture, so picture and word centre together."""
    return ScaledTranslation(
        (_ICON_PT + _ICON_GAP_PT) / 2.0 / 72.0, 0.0, figure.dpi_scale_trans
    )


def _any_icon(icons: list[Path | None]) -> bool:
    """Whether any category on this axis brought a picture to sit beside its word."""
    return any(icon is not None for icon in icons)


def _icon_band() -> float:
    """How much room in points a row of pictures claims under the axis."""
    return _ICON_DROP_PT + _ICON_PT / 2.0


def _annotate_above(
    axes: Axes, series: list[Series], offsets: list[float], turned: bool
) -> None:
    """Write every column's value above its cap, centred on the column it belongs to.

    Turned, a value claims only its own height across the axis, which is what lets
    every column carry one at a width the figure no longer grows to fit. Centring
    then falls on the glyph band rather than on the written word, so a value reads
    centred over its column instead of merely measuring so.
    """
    for entry, offset in zip(series, offsets, strict=True):
        for slot, text in enumerate(entry.labels):
            axes.annotate(
                text,
                (slot + offset, entry.values[slot]),
                xytext=(0, _VALUE_GAP_PT),
                textcoords="offset points",
                rotation=90.0 if turned else 0.0,
                rotation_mode="anchor",
                va="center" if turned else "bottom",
                ha="left" if turned else "center",
                fontsize=_VALUE_PT,
                color=INK,
            )


def _annotate_beside(axes: Axes, series: list[Series], offsets: list[float]) -> None:
    """Write every bar's value past its end, on the line the bar runs along."""
    for entry, offset in zip(series, offsets, strict=True):
        for slot, text in enumerate(entry.labels):
            axes.annotate(
                text,
                (entry.values[slot], slot + offset),
                xytext=(_VALUE_GAP_PT, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=_VALUE_PT,
                color=INK,
            )


def _legend(axes: Axes, series: list[Series]) -> None:
    """Key the series by swatch, so identity never rests on color alone."""
    axes.legend(
        handles=[Patch(facecolor=entry.color, label=entry.label) for entry in series],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        borderaxespad=0.0,
        ncols=len(series),
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        fontsize=_LEGEND_PT,
    )


def _in_data_units(axes: Axes, pixels: float) -> tuple[float, float]:
    """Convert a length in pixels into the x and y data units of one plot."""
    inverse = axes.transData.inverted()
    origin = inverse.transform((0.0, 0.0))
    corner = inverse.transform((pixels, pixels))

    return abs(corner[0] - origin[0]), abs(corner[1] - origin[1])


def _laid_on_its_side(path: DrawPath) -> DrawPath:
    """Swap a column's coordinates, which is that same bar run along x instead."""
    swap = Affine2D(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

    return path.transformed(swap)


def _zero_mark(
    center: float,
    thickness: float,
    thickness_pt: float,
    color: str,
    horizontal: bool,
) -> PathPatch:
    """Mark an arm that picked none with the dotted footprint of the bar it did not
    draw, so a zero reads as this arm choosing none rather than as nothing plotted."""
    footprint = DrawPath(
        [(center - thickness / 2, 0.0), (center + thickness / 2, 0.0)],
        [DrawPath.MOVETO, DrawPath.LINETO],
    )

    return PathPatch(
        _laid_on_its_side(footprint) if horizontal else footprint,
        facecolor="none",
        edgecolor=color,
        linewidth=_ZERO_MARK_PT,
        linestyle=_dots_across(thickness_pt),
        clip_on=False,
    )


def _dots_across(points: float) -> tuple[float, tuple[float, float]]:
    """Fit whole dots across a footprint, so its row never ends on a cut-off one."""
    ratio = _ZERO_GAP_PT / _ZERO_DOT_PT
    count = max(round((points + _ZERO_GAP_PT) / (_ZERO_DOT_PT + _ZERO_GAP_PT)), 2)
    dot = points / (count + (count - 1) * ratio)

    return (0.0, (dot / _ZERO_MARK_PT, dot * ratio / _ZERO_MARK_PT))


def _in_points(axes: Axes, figure: Figure, span: float, horizontal: bool) -> float:
    """Write a span of data units in points, the unit a dash pattern is set in."""
    origin = axes.transData.transform((0.0, 0.0))
    end = axes.transData.transform((0.0, span) if horizontal else (span, 0.0))
    reach = abs(end[1] - origin[1]) if horizontal else abs(end[0] - origin[0])

    return reach * 72.0 / figure.dpi


def _bar_path(
    value: float, center: float, thickness: float, rx: float, ry: float
) -> DrawPath:
    """A column from zero to value, square on the baseline and rounded at its top."""
    ry = min(ry, value)
    rx = min(rx, thickness / 2)
    low, high = center - thickness / 2, center + thickness / 2

    return DrawPath(
        [
            (low, 0.0),
            (low, value - ry),
            (low, value),
            (low + rx, value),
            (high - rx, value),
            (high, value),
            (high, value - ry),
            (high, 0.0),
            (low, 0.0),
        ],
        [
            DrawPath.MOVETO,
            DrawPath.LINETO,
            DrawPath.CURVE3,
            DrawPath.CURVE3,
            DrawPath.LINETO,
            DrawPath.CURVE3,
            DrawPath.CURVE3,
            DrawPath.LINETO,
            DrawPath.CLOSEPOLY,
        ],
    )
