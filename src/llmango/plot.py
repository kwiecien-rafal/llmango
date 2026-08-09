"""How a chart is drawn, how its numbers are tabled, and how a figure is written."""

import re
from base64 import b64encode
from collections import Counter
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import ceil, cos, radians, sin
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as DrawPath
from matplotlib.textpath import text_to_path
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import (
    Affine2D,
    ScaledTranslation,
    blended_transform_factory,
)
from matplotlib.typing import RcKeyType

from llmango.aggregate import Aggregate, Distribution
from llmango.spec import FREE_TEXT, OTHER_CATEGORY
from llmango.stats import wilson_interval

INK = "#767676"
FRAME = "#8b6f4e"

TEXT_FAMILY = "DejaVu Sans"

_COLUMN_IN = 0.17
_BAR_IN = 0.14
_COLUMN_GAP_IN = 0.04
_GROUP_PAD_IN = 0.08
_COLUMN_STEP = (_COLUMN_IN + _COLUMN_GAP_IN) / _COLUMN_IN
_BAR_STEP = (_BAR_IN + _COLUMN_GAP_IN) / _BAR_IN
_MAX_HEIGHT_IN = 12.0
_PLOT_HEIGHT_IN = 2.5
_PANEL_IN = 1.9
_ROW_RULE_IN = 0.18
_BAR_CHROME_IN = 1.1
_DOT_ROW_IN = 0.3
_LEGEND_IN = 0.34
_TITLE_LINE_IN = 0.24
_MAX_COLUMN_IN = 0.42
_CORNER_PX = 4.0
_MIN_MARK_PX = 3.0
_ZERO_MARK_PT = 2.0
_ZERO_DOT_PT = 2.0
_ZERO_GAP_PT = 3.0
_HEADROOM = 1.09
_BAR_HEADROOM = 1.18
_ROW_SPACE = 0.12
_BAND_ALPHA = 0.1
_PLOT_GUTTER_IN = 1.0
_PLOT_BAND_IN = 1.7
_TURN = (-45.0, -90.0)
_PANEL_TITLE_WEIGHT = "semibold"
_FINE_SHARE = 0.1
_FINEST_PLACES = 6
_CELL_PLACES = 2
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
_TITLE_PAD_PT = 10.0
_OVER_LEGEND_PT = 30.0
_KEY_DROP_PT = 8.1
_ROW_RULE_PT = 1.0
_ROW_RULE_GAP_PT = 7.0
_ICON_PT = 15.5
_ICON_GAP_PT = 2.75
_ICON_DROP_PT = 10.5
_ICON_SOURCE_PX = 128.0
_EXPORT_DPI = _ICON_SOURCE_PX * 72.0 / _ICON_PT
_ICON_GID = "icon-"
_HASHED_ID = re.compile(r'id="([A-Za-z]+[0-9a-f]{10})"')
_EMBEDDED_ICON = re.compile(
    rf'(<g id="{_ICON_GID}(\w+?)-\d+">\s*)<image xlink:href="'
    r'data:image/png;base64,[^"]*"(?: id="[^"]*")?'
    r' transform="scale\(1 -1\) translate\(0 -([\d.]+)\)"'
    r' x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="\3"/>'
)

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


@dataclass(frozen=True)
class Canvas:
    """One width a chart is drawn to be read at, since the page reads it at two."""

    width_in: float
    title_width_in: float


WIDE = Canvas(width_in=8.3125, title_width_in=6.19)
NARROW = Canvas(width_in=4.375, title_width_in=3.26)

_canvas: ContextVar[Canvas] = ContextVar("canvas", default=WIDE)
_drawn_icons: ContextVar[list[Path] | None] = ContextVar("drawn_icons", default=None)


def _width() -> float:
    """How wide the figure being drawn is, which the page decides and not the chart."""
    return _canvas.get().width_in


Row = dict[str, Any]
Key = dict[str, str]
CategoryLabel = Callable[[str], str]
CategoryIcon = Callable[[str], Path | None]
SchemaLabel = Callable[[str], str]
SeriesColor = Callable[[str], str]

_SHARE_LABEL = "share of valid answers"


def _write_share(value: float) -> str:
    """Write a share to one decimal, never as the zero a picked category is not."""
    percent = value * 100
    places = next(
        (place for place in range(1, _FINEST_PLACES + 1) if round(percent, place) > 0),
        1,
    )

    return f"{percent:.{places}f}%"


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


def _write_share_cell(value: float) -> str:
    """Write a share down a table column, every number carrying the same decimals."""
    return f"{value * 100:.{_CELL_PLACES}f}%"


def _write_count(value: float) -> str:
    """Write a count, dropping the decimals a whole number has no use for."""
    return f"{round(value, 2):g}"


def _write_count_cell(value: float) -> str:
    """Write a count down a table column, every number carrying the same decimals."""
    return f"{value:.{_CELL_PLACES}f}"


@dataclass(frozen=True)
class Unit:
    """What a chart's numbers are, since not every chart plots a share."""

    name: str
    write: Callable[[float], str]
    write_tick: Callable[[float], str]
    write_cell: Callable[[float], str]


SHARE = Unit("share", _write_share, _write_share_tick, _write_share_cell)
COUNT = Unit("count", _write_count, _write_count, _write_count_cell)


@dataclass(frozen=True, order=True)
class Arm:
    """One comparable series: a question under one schema and language."""

    schema: str
    lang: str


@dataclass(frozen=True)
class Written:
    """One category's name under its column: what fits flat, and what turns down."""

    flat: str
    turned: str


@dataclass(frozen=True)
class Piece:
    """One run of a name that turned: what it says, its angle and where it hangs."""

    text: str
    angle: float
    x: float
    y: float


@dataclass(frozen=True)
class Series:
    """One drawn series: its column values and the labels written over them."""

    label: str
    color: str
    values: list[float]
    labels: list[str]


@dataclass(frozen=True)
class Estimate:
    """One named number and the interval around it, which is what a dot plots."""

    label: str
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class Tabled:
    """The numbers the site prints, whether or not a figure was drawn from them."""

    title: str
    row_label: str
    unit: str
    columns: list[str]
    rows: list[Row]


@dataclass(frozen=True)
class Drawn(Tabled):
    """One finished figure, and the numbers behind it the site puts in a table."""

    figure: Figure


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


@dataclass(frozen=True)
class TableDef:
    """One table an experiment declares: how it is cited, what it reads and writes."""

    name: str
    number: str
    title: str
    questions: tuple[str, ...]
    build: Callable[[dict[str, Aggregate], str], "Tabled"]

    def numbered_title(self) -> str:
        """The title the table carries, opening with the number a page cites it by."""
        return f"Table {self.number}: {self.title}"


def distribution(
    cells: dict[str, Distribution],
    title: str,
    series_color: SeriesColor,
    category_label: CategoryLabel | None = None,
    category_icon: CategoryIcon | None = None,
    row_label: str = "category",
    categories: list[str] | None = None,
    reference: float | None = None,
    horizontal: bool = False,
    zeros_written: bool = False,
    ceiling: float | None = None,
) -> Drawn:
    """Draw labeled arms' category shares, and return the numbers behind them."""
    shown_categories = categories or _categories(cells.values())
    series = list(
        _series(cells, shown_categories, series_color, zeros_written).values()
    )
    written_categories = _shown(shown_categories, category_label)
    legend = len(cells) > 1
    figure = (
        bars(
            category_labels=written_categories,
            series=series,
            title=title,
            value_label=_SHARE_LABEL,
            unit=SHARE,
            legend=legend,
            reference=reference,
            ceiling=ceiling,
        )
        if horizontal
        else columns(
            category_labels=written_categories,
            series=series,
            title=title,
            value_label=_SHARE_LABEL,
            unit=SHARE,
            legend=legend,
            category_icons=_pictured(shown_categories, category_icon),
            reference=reference,
            ceiling=ceiling,
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
    series_color: SeriesColor,
    schema_label: SchemaLabel | None = None,
    category_label: CategoryLabel | None = None,
    category_icon: CategoryIcon | None = None,
    categories: list[str] | None = None,
    zeros_written: bool = False,
    ceiling: float | None = None,
) -> Drawn:
    """Draw one question's arms, labeled by whichever of its dimensions varies."""
    arms = _arms(aggregate["distributions"])
    labels = _labels(list(arms), schema_label)

    return distribution(
        cells=dict(zip(labels, arms.values(), strict=True)),
        title=title,
        series_color=series_color,
        category_label=category_label,
        category_icon=category_icon,
        categories=categories,
        zeros_written=zeros_written,
        ceiling=ceiling,
    )


def panels(
    cells: dict[str, dict[str, Distribution]],
    title: str,
    series_color: SeriesColor,
    category_label: CategoryLabel | None = None,
    category_icon: CategoryIcon | None = None,
    row_label: str = "category",
    categories: list[str] | None = None,
    zeros_written: bool = False,
) -> Drawn:
    """Draw one panel per thing a question varies, its series colored throughout."""
    shown_categories = categories or _categories(
        cell for panel in cells.values() for cell in panel.values()
    )
    drawn = {
        name: _series(panel, shown_categories, series_color, zeros_written)
        for name, panel in cells.items()
    }
    order = list(dict.fromkeys(label for panel in cells.values() for label in panel))
    figure = panel_columns(
        panels=[(name, list(series.values())) for name, series in drawn.items()],
        order=order,
        category_labels=_shown(shown_categories, category_label),
        title=title,
        value_label=_SHARE_LABEL,
        unit=SHARE,
        category_icons=_pictured(shown_categories, category_icon),
    )
    placed = [
        (f"{label} / {name}", drawn[name][label], panel[label])
        for label in order
        for name, panel in cells.items()
        if label in panel
    ]

    return Drawn(
        figure=figure,
        title=title,
        row_label=row_label,
        unit=SHARE.name,
        columns=[column for column, _, _ in placed],
        rows=_rows(
            shown_categories,
            [entry for _, entry, _ in placed],
            [cell for _, _, cell in placed],
        ),
    )


def estimates(
    cells: dict[str, float],
    title: str,
    value_label: str,
    row_label: str,
    counts: dict[str, int],
    intervals: dict[str, tuple[float, float]],
    series_color: SeriesColor,
    key: Key | None = None,
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
    plotted = [Estimate(name, value, *intervals[name]) for name, value in cells.items()]

    return Drawn(
        figure=dots(
            plotted,
            [series_color(estimate.label) for estimate in plotted],
            title,
            value_label,
            unit,
            floor,
            key,
        ),
        title=title,
        row_label=row_label,
        unit=unit.name,
        columns=[value_label],
        rows=_estimate_rows(plotted, counts, unit),
    )


def table(
    cells: dict[str, int],
    total: int,
    title: str,
    row_label: str,
    count_column: str,
    share_column: str,
    row_icon: CategoryIcon | None = None,
) -> Tabled:
    """Write what each name was counted and its share, with no figure drawn from it."""
    return Tabled(
        title=title,
        row_label=row_label,
        unit=COUNT.name,
        columns=[count_column, share_column],
        rows=[
            _pooled_row(label, count, total, row_icon) for label, count in cells.items()
        ],
    )


def _pooled_row(
    label: str, count: int, total: int, row_icon: CategoryIcon | None
) -> Row:
    """One name's row: what it was counted, what share that is, and where its
    picture is, which analyze writes out the way it writes out a figure."""
    icon = row_icon(label) if row_icon is not None else None
    row: Row = {
        "label": label,
        "cells": [_counted_cell(count, total), _pooled_cell(count, total)],
    }
    if icon is not None:
        row["icon"] = icon

    return row


def _counted_cell(count: int, total: int) -> dict[str, Any]:
    """One pooled count, written as the plain number of answers behind it."""
    return {
        "value": count,
        "count": count,
        "n": total,
        "written": COUNT.write(count),
    }


def _pooled_cell(count: int, total: int) -> dict[str, Any]:
    """One pooled number's share of the whole, and the counts it was taken from."""
    value = round(count / total, _FINEST_PLACES) if total else 0.0

    return {
        "value": value,
        "count": count,
        "n": total,
        "written": SHARE.write(value),
    }


def _series(
    cells: dict[str, Distribution],
    categories: list[str],
    series_color: SeriesColor,
    zeros_written: bool,
) -> dict[str, Series]:
    """Build one drawn series per arm, each column written at the precision it needs."""
    shares = [
        [_share(cell, category) for category in categories] for cell in cells.values()
    ]

    return {
        label: Series(
            label=label,
            color=series_color(label),
            values=values,
            labels=written if zeros_written else _only_picked(values, written),
        )
        for label, values, written in zip(
            cells, shares, _written_columns(shares), strict=True
        )
    }


def _only_picked(values: list[float], written: list[str]) -> list[str]:
    """Leave a category this arm never picked its dotted footprint and no number.

    The mark under an empty slot says the arm picked none of that category, and it
    says it in the arm's own color. A chart that has already shown a reader what
    one looks like spends nothing further on writing every one of them out, and
    the number is in the table under the figure either way.
    """
    return [
        text if value > 0 else "" for value, text in zip(values, written, strict=True)
    ]


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
        "written": unit.write_cell(value),
        "written_interval": f"{unit.write_cell(low)}–{unit.write_cell(high)}",
    }


@contextmanager
def styled(canvas: Canvas = WIDE) -> Generator[None]:
    """Scope one chart: the style and width it is drawn at, and the icons it drew.

    A chart is saved inside this rather than after it, because what the icons of
    the chart being written are is only known while that chart is the one drawn.
    """
    canvas_token = _canvas.set(canvas)
    icons_token = _drawn_icons.set([])
    try:
        with matplotlib.rc_context(_STYLE):
            yield
    finally:
        _drawn_icons.reset(icons_token)
        _canvas.reset(canvas_token)


def save(figure: Figure, path: Path) -> Path:
    """Save one figure as the transparent, reproducible SVG the site embeds."""
    icons = {icon.stem: icon for icon in _drawn_icons.get() or []}
    figure.savefig(
        path, format="svg", dpi=_EXPORT_DPI, transparent=True, metadata={"Date": None}
    )

    document = _sourced_icons(path.read_text(encoding="utf-8"), icons)
    path.write_text(_settled_ids(document), encoding="utf-8")

    return path


def _settled_ids(document: str) -> str:
    """Number the ids matplotlib hashes, since a clip path's hashes its address."""
    settled: dict[str, str] = {}
    for found in _HASHED_ID.findall(document):
        settled.setdefault(found, f"id{len(settled)}")

    for found, numbered in settled.items():
        document = document.replace(found, numbered)

    return document


def _sourced_icons(document: str, icons: dict[str, Path]) -> str:
    """Carry each icon's own file, rather than the copy matplotlib resampled.

    Matplotlib redraws an image into the SVG at the size and subpixel offset that
    placement landed on, so ten fruits reach the file as dozens of slightly soft,
    slightly different rasters. The source file is what is carried instead, once
    per placement and untouched. It stays carried rather than linked because an
    SVG a page loads through <img> is not allowed to fetch anything.

    Matplotlib's own raster runs bottom row first and is stood back up by a flip.
    A file does not, so the flip is undone: mirroring the box about y = 0 is what
    the flip amounted to. The id it hashed for the raster goes with it.
    """

    def sourced(found: re.Match[str]) -> str:
        opening, stem, _, x, y, side = found.groups()
        top = y[1:] if y.startswith("-") else f"-{y}"
        encoded = b64encode(icons[stem].read_bytes()).decode("ascii")

        return (
            f'{opening}<image xlink:href="data:image/png;base64,{encoded}" '
            f'x="{x}" y="{top}" width="{side}" height="{side}"/>'
        )

    return _EMBEDDED_ICON.sub(sourced, document)


def columns(
    category_labels: list[str],
    series: list[Series],
    title: str,
    value_label: str,
    unit: Unit,
    legend: bool,
    category_icons: list[Path | None] | None = None,
    reference: float | None = None,
    ceiling: float | None = None,
) -> Figure:
    """Draw one vertical bar chart, grouped when it carries several series."""
    icons: list[Path | None] = category_icons or [None] * len(category_labels)
    pictured = _any_icon(icons)
    slots = max(len(category_labels), 1)
    stacked = _stacks(category_labels, icons, slots)
    written = _written_categories(category_labels, stacked, slots)
    turned_values = not _values_fit(series, _placed(slots, len(series)), slots)
    figure = Figure(
        figsize=(
            _width(),
            _height(written, title, legend, stacked, pictured),
        ),
        layout="constrained",
    )
    axes = figure.add_subplot()

    _frame(
        axes,
        written,
        title,
        value_label,
        unit,
        legend,
        ceiling or _top(series, reference, _headroom(series, turned_values)),
    )
    if reference is not None:
        _reference(axes, reference, horizontal=False)
    _named(figure, axes, written, icons, stacked, worded=True)
    if legend:
        _legend(figure, axes, _series_key(series))

    _settle(figure)
    _draw_columns(
        axes, figure, series, list(range(len(series))), len(series), turned_values
    )

    return figure


def panel_columns(
    panels: list[tuple[str, list[Series]]],
    order: list[str],
    category_labels: list[str],
    title: str,
    value_label: str,
    unit: Unit,
    category_icons: list[Path | None] | None = None,
) -> Figure:
    """Draw one column chart per panel, stacked over one shared category axis."""
    places = [[order.index(entry.label) for entry in series] for _, series in panels]
    drawn = [entry for _, series in panels for entry in series]
    icons: list[Path | None] = category_icons or [None] * len(category_labels)
    pictured = _any_icon(icons)
    slots = max(len(category_labels), 1)
    stacked = pictured or _stacks(category_labels, icons, slots)
    written = _written_categories(category_labels, stacked, slots)
    offsets = _placed(slots, len(order))
    turned_values = not all(
        _values_fit(series, [offsets[place] for place in at], slots)
        for (_, series), at in zip(panels, places, strict=True)
    )
    top = _top(drawn, None, _headroom(drawn, turned_values))
    figure = Figure(
        figsize=(
            _width(),
            _panel_height(len(panels), written, title, stacked, pictured),
        ),
        layout=ConstrainedLayoutEngine(hspace=_ROW_SPACE),
    )
    plots = [
        figure.add_subplot(len(panels), 1, index + 1) for index in range(len(panels))
    ]

    figure.suptitle(_wrapped(title), fontsize=_TITLE_PT)
    figure.supylabel(value_label, fontsize=_AXIS_LABEL_PT)
    for axes, (name, _) in zip(plots, panels, strict=True):
        keyed, bottom = axes is plots[0], axes is plots[-1]
        _panel_frame(axes, written, name, unit, top, keyed)
        if keyed:
            _legend(figure, axes, _keyed(order, drawn), above=_TITLE_PAD_PT)
        else:
            _row_rule(figure, axes)
        _named(figure, axes, written, icons, stacked, worded=bottom)

    _settle(figure)
    _banded(figure, plots, slots)
    for axes, (_, series), at in zip(plots, panels, places, strict=True):
        _draw_columns(axes, figure, series, at, len(order), turned_values)

    return figure


def _draw_columns(
    axes: Axes,
    figure: Figure,
    series: list[Series],
    places: list[int],
    count: int,
    turned_values: bool,
) -> None:
    """Draw one plot's columns and their values, once the layout under them is final."""
    thickness = _thickness(axes, figure, _COLUMN_IN / _pitch(count), horizontal=False)
    thickness_pt = _in_points(axes, figure, thickness, horizontal=False)
    radius_x, radius_y = _in_data_units(axes, _CORNER_PX)
    _, shortest = _in_data_units(axes, _MIN_MARK_PX)
    grouped = _offsets(count, thickness, _COLUMN_STEP)
    offsets = [grouped[place] for place in places]

    _annotate_above(axes, series, offsets, turned_values)
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


def bars(
    category_labels: list[str],
    series: list[Series],
    title: str,
    value_label: str,
    unit: Unit,
    legend: bool,
    reference: float | None = None,
    ceiling: float | None = None,
) -> Figure:
    """Draw one horizontal bar chart, for categories too many to stand along x."""
    count = len(series)
    pitch = count * _BAR_IN + (count - 1) * _COLUMN_GAP_IN + _GROUP_PAD_IN
    slots = max(len(category_labels), 1)
    figure = Figure(
        figsize=(_width(), _bar_height(slots, pitch, legend)),
        layout="constrained",
    )
    axes = figure.add_subplot()
    offsets = _offsets(count, _BAR_IN / pitch, _BAR_STEP)

    _horizontal_frame(
        axes,
        category_labels,
        title,
        value_label,
        unit,
        0.0,
        ceiling or _top(series, reference, _BAR_HEADROOM),
        legend,
    )
    if reference is not None:
        _reference(axes, reference, horizontal=True)
    _annotate_beside(axes, series, offsets)
    if legend:
        _legend(figure, axes, _series_key(series))

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
    plotted: list[Estimate],
    colors: list[str],
    title: str,
    value_label: str,
    unit: Unit,
    floor: float,
    key: Key | None = None,
) -> Figure:
    """Draw every estimate as a dot on its own row, its interval the line under it."""
    slots = max(len(plotted), 1)
    end = _end(plotted, floor)
    labels = [estimate.label for estimate in plotted]
    legend = bool(key)
    figure = Figure(
        figsize=(_width(), _bar_height(slots, _DOT_ROW_IN, legend)),
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
        legend,
    )
    axes.tick_params(axis="y", pad=_DOT_PT / 2.0 + _TICK_PAD_PT)
    if key:
        _legend(figure, axes, key)
    for slot, (estimate, color) in enumerate(zip(plotted, colors, strict=True)):
        axes.plot(
            [estimate.low, estimate.high],
            [slot, slot],
            color=color,
            linewidth=_INTERVAL_PT,
            solid_capstyle="round",
            clip_on=False,
        )
        axes.plot(
            [estimate.value],
            [slot],
            marker="o",
            markersize=_DOT_PT,
            color=color,
            linestyle="none",
            clip_on=False,
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


def _pitch(count: int) -> float:
    """How wide a group of columns sits, which is what a category's slot holds."""
    return count * _COLUMN_IN + (count - 1) * _COLUMN_GAP_IN + _GROUP_PAD_IN


def _offsets(count: int, thickness: float, step: float) -> list[float]:
    """Where each series sits inside a category's slot, the group centred on it."""
    return [(index - (count - 1) / 2) * thickness * step for index in range(count)]


def _placed(slots: int, count: int) -> list[float]:
    """Where a group's columns will land, close enough to ask whether values fit."""
    room = (_width() - _PLOT_GUTTER_IN) / max(slots, 1)
    thickness = min(_COLUMN_IN / _pitch(count), _MAX_COLUMN_IN / room)

    return _offsets(count, thickness, _COLUMN_STEP)


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


def _values_fit(series: list[Series], offsets: list[float], slots: int) -> bool:
    """Whether every value written flat clears the value written next along the axis."""
    category = (_width() - _PLOT_GUTTER_IN) / max(slots, 1)
    written = sorted(
        ((slot + offset) * category, _written_width(text, _VALUE_PT) / 72.0)
        for entry, offset in zip(series, offsets, strict=True)
        for slot, text in enumerate(entry.labels)
        if text
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
        (
            _written_width(text, _VALUE_PT)
            for entry in series
            for text in entry.labels
            if text
        ),
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
    written: list[Written],
    title: str,
    legend: bool,
    stacked: bool,
    pictured: bool,
) -> float:
    """Leave a figure room for the labels under its axis, hung under or beside.

    A title wraps to whatever the canvas it is read at leaves it, and every line
    past the first is height the plot under it does not get. Narrow, that is what
    stands between an axis label and the bottom of the figure.
    """
    strip = _label_strip(written, stacked, pictured)
    wrapped = _wrapped(title).count("\n") * _TITLE_LINE_IN

    return _PLOT_HEIGHT_IN + strip + wrapped + (_LEGEND_IN if legend else 0.0)


def _label_strip(written: list[Written], stacked: bool, pictured: bool) -> float:
    """How deep in inches the categories sit under their axis, pictured and turned."""
    if not stacked:
        return (_icon_band() if pictured else 0.0) / 72.0

    hanging = max((_reach(entry) for entry in written), default=0.0)

    return (_word_drop(pictured) + max(hanging, _line_height())) / 72.0


def _panel_height(
    count: int, written: list[Written], title: str, stacked: bool, pictured: bool
) -> float:
    """Grow a faceted figure down the page, since its width is the article's."""
    one = _height(written, title, legend=True, stacked=stacked, pictured=pictured)

    return one + (count - 1) * (_PANEL_IN + _ROW_RULE_IN)


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


def _written_columns(plotted: list[list[float]]) -> list[list[str]]:
    """Write every column's number at the precision its own group turns out to need."""
    groups = [list(group) for group in zip(*plotted, strict=True)]

    return [
        [_write_column(value, groups[slot]) for slot, value in enumerate(values)]
        for values in plotted
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
    written: list[Written],
    title: str,
    value_label: str,
    unit: Unit,
    legend: bool,
    top: float,
) -> None:
    """Set up the plot frame: a recessive grid, its own unit and room for labels."""
    _value_axis(axes, written, unit, top)
    axes.set_ylabel(value_label, fontsize=_AXIS_LABEL_PT)
    _bare(axes, title, legend)


def _panel_frame(
    axes: Axes,
    written: list[Written],
    name: str,
    unit: Unit,
    top: float,
    keyed: bool,
) -> None:
    """Set up one panel's frame, named in the weight that tells a row from a title."""
    _value_axis(axes, written, unit, top)
    axes.set_title(
        name,
        loc="left",
        fontsize=_AXIS_LABEL_PT,
        fontweight=_PANEL_TITLE_WEIGHT,
        pad=_TITLE_PAD_PT + _KEY_DROP_PT if keyed else _TITLE_PAD_PT,
    )
    _stripped(axes)


def _row_rule(figure: Figure, axes: Axes) -> None:
    """Rule one panel off from the one above it, just over the name it is given."""
    above = _TITLE_PAD_PT + _AXIS_LABEL_PT + _ROW_RULE_GAP_PT
    axes.add_artist(
        Line2D(
            (0.0, 1.0),
            (1.0, 1.0),
            transform=axes.transAxes
            + ScaledTranslation(0.0, above / 72.0, figure.dpi_scale_trans),
            color=FRAME,
            linewidth=_ROW_RULE_PT,
            clip_on=False,
        )
    )


def _banded(figure: Figure, plots: list[Axes], slots: int) -> None:
    """Shade every other category the whole height of the stack it is read down.

    The shading belongs to the category, not to any one panel, so it is one band
    behind the panels rather than a band drawn inside each: it passes behind the
    rules between them and behind every row of pictures, and a fruit is one
    unbroken column from the top panel's ceiling to the foot of the figure.
    """
    top = plots[0].get_position().y1
    span = blended_transform_factory(plots[0].transData, figure.transFigure)
    for slot in range(0, slots, 2):
        figure.add_artist(
            Rectangle(
                (slot - 0.5, 0.0),
                1.0,
                top,
                transform=span,
                facecolor=FRAME,
                edgecolor="none",
                alpha=_BAND_ALPHA,
                zorder=-1,
            )
        )


def _value_axis(axes: Axes, written: list[Written], unit: Unit, top: float) -> None:
    """Scale a column chart: its categories along x, its unit and grid up y."""
    axes.set_ylim(0.0, top)
    axes.set_xlim(-0.5, max(len(written), 1) - 0.5)
    axes.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    axes.yaxis.set_major_formatter(_tick_format(unit))
    axes.set_xticks(range(len(written)), labels=[entry.flat for entry in written])
    axes.grid(axis="y", color=INK, alpha=0.25, linewidth=0.6)


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
    axes.set_title(
        _wrapped(title), loc="center", pad=_OVER_LEGEND_PT if legend else _TITLE_PAD_PT
    )
    _stripped(axes)


def _stripped(axes: Axes) -> None:
    """Take a plot down to its grid, which every form a chart takes shares."""
    axes.set_axisbelow(True)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)


def _wrapped(title: str) -> str:
    """Break a title over lines, since a figure no longer widens to fit one."""
    room = _canvas.get().title_width_in
    written = _written_width(title, _TITLE_PT) / 72.0
    if written <= room:
        return title

    return "\n".join(wrap(title, ceil(len(title) * room / written)))


def _tick_format(unit: Unit) -> FuncFormatter:
    """Write a value axis's own ticks in the unit that axis plots."""

    def write_tick(value: float, _: int) -> str:
        return unit.write_tick(value)

    return FuncFormatter(write_tick)


def _stacks(category_labels: list[str], icons: list[Path | None], slots: int) -> bool:
    """Whether a category's word is too wide to sit beside its picture on one line."""
    return any(
        _label_width(label, icon) > _room(slots)
        for label, icon in zip(category_labels, icons, strict=True)
    )


def _room(slots: int) -> float:
    """How wide in inches one category's own column is, which is what it may fill."""
    return (_width() - _PLOT_GUTTER_IN) / max(slots, 1)


def _label_width(label: str, icon: Path | None) -> float:
    """How wide in inches one category's label sits, its picture beside its word."""
    beside = _ICON_PT + _ICON_GAP_PT if icon is not None else 0.0

    return (_written_width(label, _TICK_PT) + beside) / 72.0


def _written_categories(
    category_labels: list[str], stacked: bool, slots: int
) -> list[Written]:
    """Lay every category's name out flat, turning down what runs past its column."""
    if not stacked:
        return [Written(flat=label, turned="") for label in category_labels]

    return [_broken(label, _room(slots)) for label in category_labels]


def _broken(label: str, room: float) -> Written:
    """Break a name at the last letter that fits its column, the rest turning down.

    A name too long for its column used to be turned whole, which spent height on
    every name to fit the longest and asked the reader to tilt for all of them.
    Broken instead, a name reads flat until it reaches the column beside it and
    then turns the corner: still one word, and still one word read left to right.

    A name that fits keeps the whole column, since only a name that has to turn
    owes the turn the width it will take. One letter always stays flat, so that
    however narrow the column gets, the name still starts where the eye is.
    """
    if _written_width(label, _TICK_PT) / 72.0 <= room:
        return Written(flat=label, turned="")

    kept = room - _line_height() / 72.0
    for cut in range(len(label) - 1, 1, -1):
        if _written_width(label[:cut], _TICK_PT) / 72.0 <= kept:
            return Written(flat=label[:cut], turned=label[cut:])

    return Written(flat=label[:1], turned=label[1:])


def _line_height() -> float:
    """How deep in points one written line of category names sits.

    Matplotlib lays every single-line label out to at least the extent of "lp",
    so that pair is what a row of names occupies whatever letters it happens to
    carry, and what a name turned on its side claims across its column.
    """
    _, ascent, descent = text_to_path.get_text_width_height_descent(
        "lp", FontProperties(family=TEXT_FAMILY, size=_TICK_PT), ismath=False
    )

    return float(ascent + descent)


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


def _named(
    figure: Figure,
    axes: Axes,
    written: list[Written],
    icons: list[Path | None],
    stacked: bool,
    worded: bool,
) -> None:
    """Name every category under one plot: its picture, and the word that follows.

    A panel that is not the last of a stack is given the pictures alone. The words
    are written once, under the bottom panel, and a row of fruit is what lets a
    column be found in the panels above without reading down to them.
    """
    axes.tick_params(labelbottom=worded)
    pictured = _any_icon(icons)
    if not pictured and not stacked:
        return

    axes.tick_params(axis="x", pad=_TICK_PAD_PT)
    if pictured:
        _pictured_row(axes, written, icons, stacked)
    if not worded:
        return

    drop = _word_drop(pictured)
    for label, entry in zip(axes.get_xticklabels(), written, strict=True):
        shift = _word_shift(figure, stacked, entry, drop)
        label.set_transform(label.get_transform() + shift)
    if stacked:
        _hung(axes, written, pictured)


def _recorded(icon: Path) -> str:
    """Note one placement of an icon, under the name its link will carry.

    A chart draws the same fruit once per panel, so the placement is numbered
    across the whole figure rather than within the panel it sits in.
    """
    placements: list[Path] = _drawn_icons.get() or []
    _drawn_icons.set(placements)
    placements.append(icon)

    return f"{icon.stem}-{len(placements) - 1}"


def _pictured_row(
    axes: Axes, written: list[Written], icons: list[Path | None], stacked: bool
) -> None:
    """Set each category's picture under its tick, above its word or ahead of it."""
    for slot, icon in enumerate(icons):
        if icon is None:
            continue

        image = imread(icon)
        beside = (
            0.0
            if stacked
            else -(_written_width(written[slot].flat, _TICK_PT) + _ICON_GAP_PT) / 2.0
        )
        drawn = AnnotationBbox(
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
        drawn.set_gid(f"{_ICON_GID}{_recorded(icon)}")
        axes.add_artist(drawn)


def _hung(axes: Axes, written: list[Written], pictured: bool) -> None:
    """Hang the rest of a broken name off the end of the part that stayed flat."""
    top = -(_TICK_PAD_PT + _word_drop(pictured))
    for slot, entry in enumerate(written):
        for piece in _bent(entry, top):
            axes.annotate(
                piece.text,
                (slot, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(piece.x, piece.y),
                textcoords="offset points",
                rotation=piece.angle,
                rotation_mode="anchor",
                va="center",
                ha="left",
                fontsize=_TICK_PT,
                color=INK,
                annotation_clip=False,
            )


def _bent(entry: Written, top: float) -> list[Piece]:
    """Walk a broken name round its corner, one run per angle it is written at.

    The turn is taken in two steps rather than one. A name that snapped through a
    right angle read as two names set at right angles to each other; the letter it
    broke at takes half the turn instead, and the word bends around the edge of
    its column. The runs sit on one line that turns under them, each beginning
    where the one before it ended, so the letters keep the spacing they would have
    had on a straight line and the word simply follows the bend.
    """
    x = (_written_width(entry.flat, _TICK_PT) - _turn_band(entry)) / 2.0
    y = top - _line_height() / 2.0
    walked: list[Piece] = []
    for text, angle in zip(_runs(entry.turned), _TURN, strict=True):
        walked.append(Piece(text=text, angle=angle, x=x, y=y))
        x, y = _onward(x, y, angle, _written_width(text, _TICK_PT))

    return [piece for piece in walked if piece.text]


def _runs(turned: str) -> tuple[str, str]:
    """Split what turns into the letter that takes half the turn and the rest."""
    return turned[:1], turned[1:]


def _onward(x: float, y: float, angle: float, run: float) -> tuple[float, float]:
    """Move along a run's own line, which is where the run after it is measured from."""
    return x + run * cos(radians(angle)), y + run * sin(radians(angle))


def _reach(entry: Written) -> float:
    """How far in points under its own line a broken name reaches, corner and all."""
    height = _line_height()
    fallen = [piece.y + _dips(piece, height) for piece in _bent(entry, 0.0)]

    return -min(fallen) if fallen else 0.0


def _dips(piece: Piece, height: float) -> float:
    """How far one run's letters fall below the point that run is hung from."""
    angle = radians(piece.angle)

    return _written_width(piece.text, _TICK_PT) * sin(angle) - height / 2.0 * cos(angle)


def _word_shift(
    figure: Figure, stacked: bool, entry: Written, drop: float
) -> ScaledTranslation:
    """Move a word clear of its picture: beside it on one line, or under it.

    Under it, the word gives up half the room its own turn will take, so that what
    is centred on the category is the whole name rather than the part of it that
    happened to stay flat.
    """
    if stacked:
        return ScaledTranslation(
            -_turn_band(entry) / 2.0 / 72.0, -drop / 72.0, figure.dpi_scale_trans
        )

    return ScaledTranslation(
        (_ICON_PT + _ICON_GAP_PT) / 2.0 / 72.0, 0.0, figure.dpi_scale_trans
    )


def _turn_band(entry: Written) -> float:
    """How much of a column a name's turn claims across it, none when it stays flat."""
    return _line_height() if entry.turned else 0.0


def _word_drop(pictured: bool) -> float:
    """How far in points a stacked word sits under the row of pictures over it."""
    return _icon_band() + _ICON_GAP_PT if pictured else 0.0


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
            if not text:
                continue
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
            if not text:
                continue
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


def _legend(figure: Figure, axes: Axes, key: Key, above: float = 0.0) -> None:
    """Key the colors by swatch, so identity never rests on color alone."""
    axes.legend(
        handles=_swatches(key),
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        bbox_transform=axes.transAxes
        + ScaledTranslation(0.0, above / 72.0, figure.dpi_scale_trans),
        borderaxespad=0.0,
        ncols=len(key),
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        fontsize=_LEGEND_PT,
    )


def _series_key(series: list[Series]) -> Key:
    """What every color a chart of series draws stands for, which is the series."""
    return {entry.label: entry.color for entry in series}


def _keyed(order: list[str], drawn: list[Series]) -> Key:
    """One entry per label a faceted figure draws, since its key names each once."""
    found = _series_key(drawn)

    return {label: found[label] for label in order}


def _swatches(key: Key) -> list[Patch]:
    """One labeled swatch per name a key carries, which is what a key is made of."""
    return [Patch(facecolor=color, label=label) for label, color in key.items()]


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
