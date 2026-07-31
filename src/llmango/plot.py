"""Everything an experiment needs to draw a chart, and how a figure is written."""

from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as DrawPath
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.typing import RcKeyType

from llmango.aggregate import Aggregate, Distribution
from llmango.spec import FREE_TEXT, OTHER_CATEGORY
from llmango.stats import wilson_interval

ARM_COLORS = ("#0072B2", "#D55E00", "#009E73")
INK = "#767676"

LIGHT_SURFACE = "#f9f9f7"
DARK_SURFACE = "#0d0d0d"

EMOJI_FAMILY = "Noto Emoji"
TEXT_FAMILY = "DejaVu Sans"
FONTS_DIR = Path(__file__).parent / "fonts"

_COLUMN_IN = 0.17
_COLUMN_GAP_IN = 0.021
_GROUP_PAD_IN = 0.16
_SIDE_CHROME_IN = 1.15
_MIN_WIDTH_IN = 3.6
_MAX_WIDTH_IN = 8.4
_PLOT_HEIGHT_IN = 2.5
_LEGEND_IN = 0.34
_LABEL_CHAR_IN = 0.075
_LEGEND_CHAR_IN = 0.068
_TITLE_CHAR_IN = 0.077
_SWATCH_IN = 0.36
_MAX_COLUMN_IN = 0.42
_CORNER_PX = 4.0
_HEADROOM = 1.07
_STACK_HEADROOM = 0.09
_LABEL_ROTATION = 45.0
_SHORT_LABEL = 3
_STACKED_LABEL_PT = 13.0

_STYLE: dict[RcKeyType, Any] = {
    "svg.hashsalt": "llmango",
    "svg.fonttype": "path",
    "font.family": [TEXT_FAMILY, EMOJI_FAMILY],
    "font.size": 11,
    "figure.dpi": 96,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.titlesize": 12,
}

Row = dict[str, Any]
CategoryLabel = Callable[[str], str]


def _register_fonts() -> None:
    """Register the vendored fonts, so an emoji renders the same on any machine."""
    for font_file in sorted(FONTS_DIR.glob("*.ttf")):
        fontManager.addfont(str(font_file))


_register_fonts()


@dataclass(frozen=True, order=True)
class Arm:
    """One comparable series: a question under one schema and language."""

    schema: str
    lang: str


@dataclass(frozen=True)
class Series:
    """One drawn series: its column values, its labels, and their uncertainty."""

    label: str
    color: str
    values: list[float]
    labels: list[str | None]
    intervals: list[tuple[float, float]] = field(
        default_factory=list[tuple[float, float]]
    )


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
    """One chart an experiment declares: its name, what it reads, and how it draws."""

    name: str
    questions: tuple[str, ...]
    draw: Callable[[dict[str, Aggregate]], "Drawn"]


def distribution(
    cells: dict[str, Distribution],
    title: str,
    category_label: CategoryLabel | None = None,
    row_label: str = "category",
    categories: list[str] | None = None,
    reference: float | None = None,
) -> Drawn:
    """Draw labeled arms' category shares, and return the numbers behind them."""
    _refuse_beyond_palette(cells, title)
    shown_categories = categories or _categories(cells.values())
    series = [
        _series(cell, label, ARM_COLORS[index], shown_categories)
        for index, (label, cell) in enumerate(cells.items())
    ]
    figure = columns(
        category_labels=_shown(shown_categories, category_label),
        series=series,
        title=title,
        value_label="share of valid answers",
        legend=len(cells) > 1,
        reference=reference,
    )

    return Drawn(
        figure=figure,
        title=title,
        row_label=row_label,
        columns=list(cells),
        rows=_rows(shown_categories, series, list(cells.values())),
    )


def question_distribution(
    aggregate: Aggregate, category_label: CategoryLabel | None = None
) -> Drawn:
    """Draw one question's arms, labeled and titled by whatever varies within it."""
    arms = _arms(aggregate["distributions"])
    labels = _labels(list(arms))

    return distribution(
        cells=dict(zip(labels, arms.values(), strict=True)),
        title=_title(aggregate["question_id"], list(arms), labels),
        category_label=category_label,
    )


def summary(
    cells: dict[str, float],
    title: str,
    value_label: str,
    row_label: str,
    reference: float | None = None,
    counts: dict[str, int] | None = None,
) -> Drawn:
    """Draw one number per named thing, which is what a cross-question chart has."""
    names = list(cells)
    values = [cells[name] for name in names]
    texts: list[str | None] = [f"{value:.0%}" for value in values]
    series = [
        Series(label=value_label, color=ARM_COLORS[0], values=values, labels=texts)
    ]
    figure = columns(
        category_labels=names,
        series=series,
        title=title,
        value_label=value_label,
        legend=False,
        reference=reference,
    )

    return Drawn(
        figure=figure,
        title=title,
        row_label=row_label,
        columns=[value_label],
        rows=[
            {
                "label": name,
                "cells": [{"value": value, "n": (counts or {}).get(name, 0)}],
            }
            for name, value in zip(names, values, strict=True)
        ],
    )


def styled() -> AbstractContextManager[None]:
    """Apply the chart style every chart is drawn and saved under."""
    return matplotlib.rc_context(_STYLE)


def save(figure: Figure, path: Path) -> Path:
    """Save one figure as the transparent, reproducible SVG the site embeds."""
    figure.savefig(path, format="svg", transparent=True, metadata={"Date": None})

    return path


def columns(
    category_labels: list[str],
    series: list[Series],
    title: str,
    value_label: str,
    legend: bool,
    reference: float | None = None,
) -> Figure:
    """Draw one vertical bar chart, grouped when it carries several series."""
    count = len(series)
    pitch = count * _COLUMN_IN + (count - 1) * _COLUMN_GAP_IN + _GROUP_PAD_IN
    slots = max(len(category_labels), 1)
    figure = Figure(
        figsize=(
            _width(slots, pitch, title, series if legend else []),
            _height(category_labels, legend),
        ),
        layout="constrained",
    )
    axes = figure.add_subplot()

    step = (_COLUMN_IN + _COLUMN_GAP_IN) / pitch
    offsets = [(index - (count - 1) / 2) * step for index in range(count)]
    ranks = _ranks(series)

    _frame(
        axes,
        category_labels,
        title,
        value_label,
        legend,
        _top(series, reference, ranks),
    )
    if reference is not None:
        _reference(axes, reference, slots)
    _annotate(axes, series, offsets, ranks)
    if legend:
        _legend(axes, series)

    _settle(figure)
    thickness = _thickness(axes, figure, _COLUMN_IN / pitch)
    radius_x, radius_y = _corner_radii(axes)
    for entry, offset in zip(series, offsets, strict=True):
        for slot, value in enumerate(entry.values):
            if value > 0:
                axes.add_patch(
                    PathPatch(
                        _bar_path(value, slot + offset, thickness, radius_x, radius_y),
                        facecolor=entry.color,
                        edgecolor="none",
                    )
                )
        _intervals(axes, entry, offset)

    return figure


def _width(slots: int, pitch: float, title: str, legend: list[Series]) -> float:
    """Widen a figure with what it carries, title and legend included, then cap it."""
    keyed = _SIDE_CHROME_IN + sum(
        len(entry.label) * _LEGEND_CHAR_IN + _SWATCH_IN for entry in legend
    )
    titled = _SIDE_CHROME_IN + len(title) * _TITLE_CHAR_IN
    wanted = max(_SIDE_CHROME_IN + slots * pitch, keyed, titled)

    return min(max(wanted, _MIN_WIDTH_IN), _MAX_WIDTH_IN)


def _top(series: list[Series], reference: float | None, ranks: list[int]) -> float:
    """End the value axis above the tallest cap, leaving room for its label stack."""
    reach = [
        max(entry.values + [bounds[1] for bounds in entry.intervals], default=0.0)
        for entry in series
    ]
    peak = max(max(reach, default=0.0), reference or 0.0)
    headroom = _HEADROOM + (max(ranks, default=0) + 1) * _STACK_HEADROOM

    return peak * headroom if peak > 0 else 1.0


def _thickness(axes: Axes, figure: Figure, wanted: float) -> float:
    """Hold a column to a readable width, so few categories do not draw slabs."""
    span = axes.get_xlim()[1] - axes.get_xlim()[0]
    inches = axes.get_window_extent().width / figure.dpi
    if inches <= 0:
        return wanted

    return min(wanted, _MAX_COLUMN_IN * span / inches)


def _height(category_labels: list[str], legend: bool) -> float:
    """Leave a figure room for its rotated labels, which hang below the axis."""
    longest = max((len(label) for label in category_labels), default=0)
    hanging = longest * _LABEL_CHAR_IN * 0.71

    return _PLOT_HEIGHT_IN + hanging + (_LEGEND_IN if legend else 0.0)


def _shown(categories: list[str], category_label: CategoryLabel | None) -> list[str]:
    """Name each category the way its experiment writes it on an axis."""
    if category_label is None:
        return list(categories)

    return [category_label(category) for category in categories]


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


def _labels(arms: list[Arm]) -> list[str]:
    """Label each arm by whichever of schema and language varies."""
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
    """Title one question's chart, naming whatever its legend cannot."""
    if len(arms) > 1:
        return f"{question_id}: answer distribution by {_dimension(arms)}"

    return f"{question_id}: answer distribution ({labels[0]})"


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


def _series(
    cell: Distribution, label: str, color: str, categories: list[str]
) -> Series:
    """Turn one arm's counts into the columns, sparse labels and caps that draw it."""
    values = [_share(cell, category) for category in categories]
    texts = [f"{value:.0%}" for value in values]
    intervals = [_interval(cell, category) for category in categories]

    return Series(
        label=label,
        color=color,
        values=values,
        labels=_peak_labels(values, texts),
        intervals=intervals,
    )


def _interval(cell: Distribution, category: str) -> tuple[float, float]:
    """The Wilson bounds around one category's share, derived from the counts.

    A category this arm never picked still has an interval, and a wide one:
    0 of 5 is 0-43%, not certainty. Reading the bounds off a stored map would
    give those categories a flat cap and claim precisely what is least known.
    """
    return wilson_interval(cell["counts"].get(category, 0), cell["n"])


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
                    "lo": entry.intervals[index][0],
                    "hi": entry.intervals[index][1],
                }
                for entry, cell in zip(series, cells, strict=True)
            ],
        }
        for index, category in enumerate(categories)
    ]


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
    legend: bool,
    top: float,
) -> None:
    """Set up the plot frame: a recessive grid, percent ticks and room for labels."""
    axes.set_ylim(0.0, top)
    axes.set_xlim(-0.5, max(len(category_labels), 1) - 0.5)
    axes.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes.set_xticks(range(len(category_labels)), labels=category_labels)
    axes.set_ylabel(value_label, fontsize=10)
    axes.set_title(title, loc="left", pad=30 if legend else 10)
    axes.grid(axis="y", color=INK, alpha=0.25, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    if _needs_turning(category_labels):
        for label in axes.get_xticklabels():
            label.set_rotation(_LABEL_ROTATION)
            label.set_rotation_mode("anchor")
            label.set_horizontalalignment("right")
            label.set_verticalalignment("top")


def _needs_turning(category_labels: list[str]) -> bool:
    """Turn category names only when they are too long to sit side by side."""
    return max((len(label) for label in category_labels), default=0) > _SHORT_LABEL


def _reference(axes: Axes, value: float, slots: int) -> None:
    """Draw the line a chart is read against, such as a fair die's even spread."""
    axes.plot(
        [-0.5, slots - 0.5],
        [value, value],
        color=INK,
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        zorder=1,
    )


def _annotate(
    axes: Axes, series: list[Series], offsets: list[float], ranks: list[int]
) -> None:
    """Write each series' direct labels just above the column tips they belong to."""
    bases = _label_bases(series)
    for entry, offset, rank in zip(series, offsets, ranks, strict=True):
        for slot, text in enumerate(entry.labels):
            if text:
                axes.annotate(
                    text,
                    (slot + offset, bases[slot]),
                    xytext=(0, 5 + rank * _STACKED_LABEL_PT),
                    textcoords="offset points",
                    va="bottom",
                    ha="center",
                    fontsize=9.5,
                    color=INK,
                )


def _label_bases(series: list[Series]) -> dict[int, float]:
    """The height every label on one category lifts from, so a stack comes out even."""
    bases: dict[int, float] = {}
    for entry in series:
        for slot, text in enumerate(entry.labels):
            if text:
                bases[slot] = max(bases.get(slot, 0.0), _tip(entry, slot))

    return bases


def _ranks(series: list[Series]) -> list[int]:
    """Stack the labels of series peaking on one category, which is the usual case.

    Every language picking the same fruit is the finding, not an edge case, so
    their three labels land on one slot and would otherwise be drawn over
    each other. Each still sits above its own column; only the height differs.
    """
    peaks = [_peak_slot(entry) for entry in series]
    ranks: list[int] = []
    for index, peak in enumerate(peaks):
        shared = [
            other for other in peaks[:index] if other == peak and other is not None
        ]
        ranks.append(len(shared))

    return ranks


def _peak_slot(entry: Series) -> int | None:
    """Which category a series labeled, or None when it labeled nothing."""
    for slot, text in enumerate(entry.labels):
        if text:
            return slot

    return None


def _tip(entry: Series, slot: int) -> float:
    """Where a column's label sits: above its cap when it has one, its top if not."""
    if slot < len(entry.intervals):
        return max(entry.values[slot], entry.intervals[slot][1])

    return entry.values[slot]


def _intervals(axes: Axes, entry: Series, offset: float) -> None:
    """Draw each column's Wilson bounds, so a share is read with its uncertainty."""
    for slot, bounds in enumerate(entry.intervals):
        if entry.values[slot] > 0 or bounds[1] > 0:
            axes.plot(
                [slot + offset, slot + offset],
                [bounds[0], bounds[1]],
                color=INK,
                linewidth=0.8,
                solid_capstyle="butt",
                zorder=3,
            )


def _legend(axes: Axes, series: list[Series]) -> None:
    """Key the series by swatch, so identity never rests on color alone."""
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
    """Convert the corner radius from pixels into x and y data units."""
    inverse = axes.transData.inverted()
    origin = inverse.transform((0.0, 0.0))
    corner = inverse.transform((_CORNER_PX, _CORNER_PX))

    return abs(corner[0] - origin[0]), abs(corner[1] - origin[1])


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
