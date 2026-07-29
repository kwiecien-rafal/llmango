"""Everything an experiment needs to draw a chart, and how a figure is written.

matplotlib draws the finished artwork here, so a chart is defined exactly once
and the file opened locally is the file the site ships. The background is
transparent and every color reads against both a light and a dark page, so one
export serves both themes and the site never restyles a chart.

Because the glyphs are outlined paths, no text in an SVG is selectable or
reachable by a screen reader. Every drawing therefore returns the numbers behind
it alongside the figure, so the site can pair each chart with a table. That table
is the accessible twin of the image, not an optional extra.

Nothing here imports llmango.experiments, which is what lets an experiment's
charts module import this one without the two cycling back on each other.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as DrawPath
from matplotlib.ticker import FixedLocator, PercentFormatter
from matplotlib.typing import RcKeyType

from llmango.aggregate import Aggregate, Distribution
from llmango.spec import FREE_TEXT, OTHER_CATEGORY

ARM_COLORS = ("#3987e5", "#d95926", "#199e70")
INK = "#7d7b76"

_FIG_WIDTH_IN = 5.2
_BAR_IN = 0.16
_BAR_GAP_IN = 0.021
_GROUP_PAD_IN = 0.13
_CHROME_IN = 1.05
_LEGEND_IN = 0.34
_CORNER_PX = 4.0
_HEADROOM = 1.28

_STYLE: dict[RcKeyType, Any] = {
    "svg.hashsalt": "llmango",
    "svg.fonttype": "path",
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 96,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "axes.titlesize": 12,
}

Row = dict[str, Any]


@dataclass(frozen=True, order=True)
class Arm:
    """One comparable series: a question under one schema and language."""

    schema: str
    lang: str


@dataclass(frozen=True)
class Series:
    """One drawn series: its bar values by row, and which rows carry a label.

    Labels are sparse on purpose. A value beside every bar goes unread, so only
    the rows worth pointing at carry one and the table view carries the rest.
    """

    label: str
    color: str
    values: list[float]
    labels: list[str | None]


@dataclass(frozen=True)
class Drawn:
    """One finished figure, and the numbers behind it the site puts in a table."""

    figure: Figure
    title: str
    row_label: str
    columns: list[str]
    rows: list[Row]


@dataclass(frozen=True)
class ChartDef:
    """One chart an experiment declares: its name, what it reads, and how it draws.

    The questions are the declaration analyze skips on: a chart is drawn only
    once every question it names has an aggregate, and draw receives those and
    nothing else.
    """

    name: str
    questions: tuple[str, ...]
    draw: Callable[[dict[str, Aggregate]], "Drawn"]


def distribution(cells: dict[str, Distribution], title: str) -> Drawn:
    """Draw labeled arms' category shares, and return the numbers behind them.

    Shares rather than counts, so arms of different size stay comparable; the
    counts they came from ride along in the table the site draws beside it.
    """
    if len(cells) > len(ARM_COLORS):
        raise ValueError(
            f"{title} has {len(cells)} arms but the palette holds "
            f"{len(ARM_COLORS)}. Extend and revalidate the palette against both "
            f"page surfaces, or split the comparison across charts."
        )
    categories = _categories(cells.values())
    series = [
        _series(cell, label, ARM_COLORS[index], categories)
        for index, (label, cell) in enumerate(cells.items())
    ]
    figure = bars(
        row_labels=categories,
        series=series,
        title=title,
        value_label="share of valid answers",
        legend=len(cells) > 1,
    )
    return Drawn(
        figure=figure,
        title=title,
        row_label="category",
        columns=list(cells),
        rows=_rows(categories, series, list(cells.values())),
    )


def question_distribution(aggregate: Aggregate) -> Drawn:
    """Draw one question's arms, labeled and titled by whatever varies within it."""
    arms = _arms(aggregate)
    labels = _labels(list(arms))
    return distribution(
        cells=dict(zip(labels, arms.values(), strict=True)),
        title=_title(aggregate["question_id"], list(arms), labels),
    )


def styled() -> AbstractContextManager[None]:
    """Apply the chart style, which both drawing and saving read.

    One context has to span the pair: the fonts and colors are baked in as a
    figure is built, while the pinned hash salt and the outlined glyphs are read
    by the SVG writer, and reproducibility needs both.
    """
    return matplotlib.rc_context(_STYLE)


def save(figure: Figure, path: Path) -> Path:
    """Save one figure as the transparent, reproducible SVG the site embeds.

    The date matplotlib would otherwise stamp into the metadata is dropped, so
    redrawing unchanged aggregates rewrites an identical file instead of a diff.
    """
    figure.savefig(path, format="svg", transparent=True, metadata={"Date": None})
    return path


def bars(
    row_labels: list[str],
    series: list[Series],
    title: str,
    value_label: str,
    legend: bool,
) -> Figure:
    """Draw one horizontal bar chart, grouped when it carries several series.

    Row pitch is set in inches rather than data units, so bar thickness tracks
    the number of bars sharing a row instead of the length of the chart. The y
    axis runs downwards so rows read top down, which is why the first series
    takes the most negative offset and so sits at the top of its group, matching
    the order the legend lists.

    The layout is solved before the bars are built: rounding a corner by a fixed
    number of pixels needs the finished data transform, and that only exists
    once constrained layout has settled the axes box.
    """
    count = len(series)
    pitch = count * _BAR_IN + (count - 1) * _BAR_GAP_IN + _GROUP_PAD_IN
    rows = max(len(row_labels), 1)
    height = _CHROME_IN + (_LEGEND_IN if legend else 0.0) + rows * pitch
    figure = Figure(figsize=(_FIG_WIDTH_IN, height), layout="constrained")
    axes = figure.add_subplot()

    thickness = _BAR_IN / pitch
    step = (_BAR_IN + _BAR_GAP_IN) / pitch
    offsets = [(index - (count - 1) / 2) * step for index in range(count)]

    _frame(axes, row_labels, title, value_label, legend)
    _annotate(axes, series, offsets)
    if legend:
        _legend(axes, series)

    _settle(figure)
    radius_x, radius_y = _corner_radii(axes)
    for entry, offset in zip(series, offsets, strict=True):
        for row, value in enumerate(entry.values):
            if value > 0:
                axes.add_patch(
                    PathPatch(
                        _bar_path(value, row + offset, thickness, radius_x, radius_y),
                        facecolor=entry.color,
                        edgecolor="none",
                    )
                )
    return figure


def _arms(aggregate: Aggregate) -> dict[Arm, Distribution]:
    """Read a question's aggregate as arm -> numbers, in a stable order."""
    return {
        Arm(schema=schema, lang=lang): cell
        for schema, langs in sorted(aggregate["distributions"].items())
        for lang, cell in sorted(langs.items())
    }


def _varies(arms: list[Arm]) -> tuple[bool, bool]:
    """Whether the schema and the language differ across a question's arms."""
    return (
        len({arm.schema for arm in arms}) > 1,
        len({arm.lang for arm in arms}) > 1,
    )


def _labels(arms: list[Arm]) -> list[str]:
    """Label each arm by whichever of schema and language varies.

    Returned in the order the arms were given, so a caller can zip the labels
    straight onto the cells they belong to.
    """
    many_schemas, many_langs = _varies(arms)
    labels: list[str] = []
    for arm in arms:
        label = _schema_label(arm.schema)
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


def _dimension(arms: list[Arm]) -> str:
    """Name what a question's arms differ in, for the legend title."""
    many_schemas, many_langs = _varies(arms)
    if many_schemas and many_langs:
        return "arm"
    return "schema" if many_schemas else "language"


def _title(question_id: str, arms: list[Arm], labels: list[str]) -> str:
    """Title one question's chart, naming whatever its legend cannot.

    Several arms are keyed by a legend, so the title only has to say what
    separates them. A lone arm gets no legend at all, so the title names it.
    """
    if len(arms) > 1:
        return f"{question_id}: answer distribution by {_dimension(arms)}"
    return f"{question_id}: answer distribution ({labels[0]})"


def _categories(cells: Iterable[Distribution]) -> list[str]:
    """The categories some arm actually picked, most picked first, 'other' last.

    A question's canonical set can hold dozens of values while a run picks a
    handful, so unpicked categories are dropped rather than drawn as empty rows.
    """
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


def _series(
    cell: Distribution, label: str, color: str, categories: list[str]
) -> Series:
    """Turn one arm's counts into the bars and sparse labels that draw it."""
    values = [_share(cell, category) for category in categories]
    texts = [
        f"{value:.0%}  {cell['counts'].get(category, 0)}/{cell['n']}"
        for category, value in zip(categories, values, strict=True)
    ]
    return Series(
        label=label, color=color, values=values, labels=_peak_labels(values, texts)
    )


def _peak_labels(values: list[float], texts: list[str]) -> list[str | None]:
    """Label only a series' largest bar, so direct labels stay worth reading."""
    if not values or max(values) <= 0:
        return [None] * len(values)
    peak = max(range(len(values)), key=lambda index: values[index])
    return [texts[index] if index == peak else None for index in range(len(values))]


def _rows(
    categories: list[str], series: list[Series], cells: list[Distribution]
) -> list[Row]:
    """Describe every plotted number, one row per category, for the table view."""
    return [
        {
            "label": category,
            "cells": [
                {
                    "value": entry.values[index],
                    "count": cell["counts"].get(category, 0),
                    "n": cell["n"],
                }
                for entry, cell in zip(series, cells, strict=True)
            ],
        }
        for index, category in enumerate(categories)
    ]


def _settle(figure: Figure) -> None:
    """Resolve the layout so the data transform is final.

    Only the constrained-layout solve is needed here, and running it alone skips
    a full render of every artist that would otherwise be thrown away.
    """
    engine = figure.get_layout_engine()
    if engine is not None:
        engine.execute(figure)


def _frame(
    axes: Axes, row_labels: list[str], title: str, value_label: str, legend: bool
) -> None:
    """Set up the plot frame: a recessive grid, percent ticks and room for labels.

    The x limit runs past 100% so a direct label always sits clear of its bar
    tip rather than being clipped by it, while the ticks still stop at 100%.
    Rows read top down, which is why the y limit is inverted rather than sorted
    backwards: the drawing order stays the same as the data's.

    A chart with a legend pads its title far enough to clear the legend row,
    which sits between the title and the plot.
    """
    axes.set_xlim(0.0, _HEADROOM)
    axes.set_ylim(max(len(row_labels), 1) - 0.5, -0.5)
    axes.xaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))
    axes.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes.set_yticks(range(len(row_labels)), labels=row_labels)
    axes.set_xlabel(value_label, fontsize=10)
    axes.set_title(title, loc="left", pad=30 if legend else 10)
    axes.grid(axis="x", color=INK, alpha=0.25, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)


def _annotate(axes: Axes, series: list[Series], offsets: list[float]) -> None:
    """Write each series' direct labels just past the bar tips they belong to."""
    for entry, offset in zip(series, offsets, strict=True):
        for row, text in enumerate(entry.labels):
            if text:
                axes.annotate(
                    text,
                    (entry.values[row], row + offset),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=9.5,
                    color=INK,
                )


def _legend(axes: Axes, series: list[Series]) -> None:
    """Key the series by swatch, so identity never rests on color alone.

    The legend carries no title of its own. What separates the arms is already
    named in the chart title, and a second heading here only crowds it.
    """
    axes.legend(
        handles=[Patch(facecolor=entry.color, label=entry.label) for entry in series],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncols=len(series),
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        fontsize=10,
    )


def _corner_radii(axes: Axes) -> tuple[float, float]:
    """Convert the corner radius from pixels into x and y data units.

    A bar is far wider than it is tall in data terms, so one shared radius would
    render as a stretched ellipse. Each axis converts separately and the corner
    comes out round.
    """
    inverse = axes.transData.inverted()
    origin = inverse.transform((0.0, 0.0))
    corner = inverse.transform((_CORNER_PX, _CORNER_PX))
    return abs(corner[0] - origin[0]), abs(corner[1] - origin[1])


def _bar_path(
    value: float, center: float, thickness: float, rx: float, ry: float
) -> DrawPath:
    """A bar from zero to value, square on the baseline and rounded at its tip."""
    rx = min(rx, value)
    ry = min(ry, thickness / 2)
    low, high = center - thickness / 2, center + thickness / 2
    return DrawPath(
        [
            (0.0, low),
            (value - rx, low),
            (value, low),
            (value, low + ry),
            (value, high - ry),
            (value, high),
            (value - rx, high),
            (0.0, high),
            (0.0, low),
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
