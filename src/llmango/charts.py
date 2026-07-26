"""Draw an experiment's aggregates as the SVG charts the site embeds.

Reads the committed JSON under data/aggregated/<experiment_id>/ and writes one
SVG per chart into site/public/charts/<experiment_id>/, which Astro serves
verbatim, alongside an index carrying the numbers behind every chart so the site
can render a table beside each image.

matplotlib draws the finished artwork here, so a chart is defined exactly once
and the file opened locally is the file the site ships. The background is
transparent and every color reads against both a light and a dark page, so one
export serves both themes and the site never restyles a chart.

Because the glyphs are outlined paths, no text in an SVG is selectable or
reachable by a screen reader. The index therefore carries every plotted number
and the site pairs each chart with a table view. That table is the accessible
twin of the image, not an optional extra.

One series is one arm: a question asked under one schema variant in one language.
Arms are labeled by whichever of the two actually varies within their question,
which is what puts 001d's three schema arms in a single chart rather than three
one-series charts.
"""

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as DrawPath
from matplotlib.ticker import FixedLocator, PercentFormatter
from matplotlib.typing import RcKeyType

from llmango.config import AGG_DIR, CHARTS_DIR
from llmango.registry import FREE_TEXT_VARIANT, OTHER_CATEGORY, resolve_experiment_id

_DISTRIBUTIONS = "distributions.json"
_INDEX = "index.json"

_ARM_COLORS = ("#3987e5", "#d95926", "#199e70")
_INK = "#7d7b76"

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
    "text.color": _INK,
    "axes.labelcolor": _INK,
    "xtick.color": _INK,
    "ytick.color": _INK,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "axes.titlesize": 12,
}

Cell = Mapping[str, Any]
Row = dict[str, Any]


@dataclass(frozen=True, order=True)
class Arm:
    """One comparable series: a question under one schema variant and language."""

    schema_variant: str
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
class Chart:
    """One written chart, and the numbers behind it the site puts in a table."""

    metric: str
    question_id: str | None
    file: str
    title: str
    row_label: str
    arms: list[str]
    columns: list[str]
    rows: list[Row]


@dataclass(frozen=True)
class RateChart:
    """How to draw one of the per-experiment rate metrics as a single bar chart."""

    metric: str
    source: str
    file: str
    title: str
    counts: tuple[str, str, str]


@dataclass(frozen=True)
class AnalyzeOutcome:
    """The charts and the index one analysis run wrote."""

    experiment_id: str
    charts: list[Chart]
    index_path: Path


_RATE_CHARTS = (
    RateChart(
        metric="language_match",
        source="language_match.json",
        file="language_match.svg",
        title="in-language rate",
        counts=("matched", "total", "undetermined"),
    ),
)


def analyze_experiment(experiment_id: str) -> AnalyzeOutcome:
    """Draw an experiment's charts from its aggregates into site/public/charts.

    The distribution chart is per question, since each question asks its own
    thing. The rate charts are per experiment, so a question with a single arm
    contributes a bar to a comparison rather than becoming a one-bar chart.
    """
    experiment_id = resolve_experiment_id(experiment_id)
    distributions = _load(experiment_id, _DISTRIBUTIONS)
    if distributions is None:
        raise FileNotFoundError(
            f"No aggregates for {experiment_id}. Run 'llmango aggregate' first."
        )

    charts = [
        _write_distribution(experiment_id, question_id, arms)
        for question_id, arms in distributions.items()
    ]
    for rate in _RATE_CHARTS:
        metric = _load(experiment_id, rate.source)
        if metric is not None:
            charts.append(_write_rate(experiment_id, rate, metric))

    return AnalyzeOutcome(
        experiment_id=experiment_id,
        charts=charts,
        index_path=_write_index(experiment_id, charts),
    )


def _load(experiment_id: str, name: str) -> dict[str, dict[Arm, Cell]] | None:
    """Read one aggregate file as question -> arm -> numbers, or None if absent."""
    path = AGG_DIR / experiment_id / name
    if not path.is_file():
        return None
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    questions = cast(dict[str, dict[str, dict[str, Cell]]], payload["questions"])
    return {
        question_id: {
            Arm(schema_variant=schema_variant, lang=lang): cell
            for schema_variant, langs in sorted(variants.items())
            for lang, cell in sorted(langs.items())
        }
        for question_id, variants in sorted(questions.items())
    }


def _varies(arms: list[Arm]) -> tuple[bool, bool]:
    """Whether the schema variant and the language differ across a question's arms."""
    return (
        len({arm.schema_variant for arm in arms}) > 1,
        len({arm.lang for arm in arms}) > 1,
    )


def _labels(arms: list[Arm]) -> list[str]:
    """Label each arm by whichever of schema variant and language varies.

    Returned in the order the arms were given, so a caller can zip the labels
    straight onto the cells they belong to.
    """
    many_variants, many_langs = _varies(arms)
    labels: list[str] = []
    for arm in arms:
        schema = _schema_label(arm.schema_variant)
        if many_variants and many_langs:
            labels.append(f"{arm.lang} / {schema}")
        elif many_variants:
            labels.append(schema)
        else:
            labels.append(arm.lang)
    return labels


def _schema_label(schema_variant: str) -> str:
    """Name a schema arm the way a legend should read it."""
    if schema_variant == FREE_TEXT_VARIANT:
        return "no schema"
    return f"{schema_variant} schema"


def _dimension(arms: list[Arm]) -> str:
    """Name what a question's arms differ in, for the legend title."""
    many_variants, many_langs = _varies(arms)
    if many_variants and many_langs:
        return "arm"
    return "schema" if many_variants else "language"


def _distribution_title(question_id: str, arms: list[Arm], labels: list[str]) -> str:
    """Title one question's chart, naming whatever its legend cannot.

    Several arms are keyed by a legend, so the title only has to say what
    separates them. A lone arm gets no legend at all, so the title names it.
    """
    if len(arms) > 1:
        return f"{question_id}: answer distribution by {_dimension(arms)}"
    return f"{question_id}: answer distribution ({labels[0]})"


def _categories(arms: dict[Arm, Cell]) -> list[str]:
    """The categories some arm actually picked, most picked first, 'other' last.

    A question's canonical set can hold dozens of values while a run picks a
    handful, so unpicked categories are dropped rather than drawn as empty rows.
    """
    totals: Counter[str] = Counter()
    for cell in arms.values():
        totals.update(cast(Mapping[str, int], cell["counts"]))
    return sorted(
        (name for name, total in totals.items() if total > 0),
        key=lambda name: (name == OTHER_CATEGORY, -totals[name], name),
    )


def _share(cell: Cell, category: str) -> float:
    """One category's share of an arm's valid answers, 0.0 when it picked none."""
    total = int(cell["n"])
    if not total:
        return 0.0
    return round(int(cell["counts"].get(category, 0)) / total, 4)


def _peak_labels(values: list[float], texts: list[str]) -> list[str | None]:
    """Label only a series' largest bar, so direct labels stay worth reading."""
    if not values or max(values) <= 0:
        return [None] * len(values)
    peak = max(range(len(values)), key=lambda index: values[index])
    return [texts[index] if index == peak else None for index in range(len(values))]


def _draw(
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
    axes.grid(axis="x", color=_INK, alpha=0.25, linewidth=0.6)
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
                    color=_INK,
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


def _distribution_figure(
    arms: dict[Arm, Cell], labels: list[str], title: str
) -> tuple[Figure, list[Row]]:
    """Draw one question's category shares, and return the numbers behind them.

    Shares rather than counts, so arms of different size stay comparable; the
    counts they came from ride along in the table the site draws beside it.
    """
    if len(arms) > len(_ARM_COLORS):
        raise ValueError(
            f"{title} has {len(arms)} arms but the palette holds "
            f"{len(_ARM_COLORS)}. Extend and revalidate the palette against both "
            f"page surfaces, or split the question across charts."
        )
    categories = _categories(arms)
    series: list[Series] = []
    for index, (cell, label) in enumerate(zip(arms.values(), labels, strict=True)):
        values = [_share(cell, category) for category in categories]
        texts = [
            f"{value:.0%}  {int(cell['counts'].get(category, 0))}/{int(cell['n'])}"
            for category, value in zip(categories, values, strict=True)
        ]
        series.append(
            Series(
                label=label,
                color=_ARM_COLORS[index],
                values=values,
                labels=_peak_labels(values, texts),
            )
        )

    figure = _draw(
        row_labels=categories,
        series=series,
        title=title,
        value_label="share of valid answers",
        legend=len(arms) > 1,
    )
    rows = [
        {
            "label": category,
            "cells": [
                {
                    "value": entry.values[index],
                    "count": int(cell["counts"].get(category, 0)),
                    "n": int(cell["n"]),
                }
                for entry, cell in zip(series, arms.values(), strict=True)
            ],
        }
        for index, category in enumerate(categories)
    ]
    return figure, rows


def _rate_figure(
    rate: RateChart, metric: dict[str, dict[Arm, Cell]], names: list[str], title: str
) -> tuple[Figure, list[Row]]:
    """Draw one rate across every arm of every question as a single bar chart.

    Every bar is labeled here, unlike the distribution chart: one series over
    few bars leaves the labels room, and the rate is the point of the chart.
    """
    cells = [cell for arms in metric.values() for cell in arms.values()]
    values = [float(cell["rate"]) for cell in cells]
    part, whole, aside = rate.counts
    texts = [
        f"{value:.0%}  {int(cell[part])} of {int(cell[whole])}, "
        f"{int(cell[aside])} {aside}"
        for value, cell in zip(values, cells, strict=True)
    ]
    series = [
        Series(
            label=rate.title,
            color=_ARM_COLORS[0],
            values=values,
            labels=list(texts),
        )
    ]
    figure = _draw(
        row_labels=names,
        series=series,
        title=title,
        value_label=rate.title,
        legend=False,
    )
    rows = [
        {
            "label": name,
            "cells": [
                {
                    "value": value,
                    "count": int(cell[part]),
                    "n": int(cell[whole]),
                    aside: int(cell[aside]),
                }
            ],
        }
        for name, value, cell in zip(names, values, cells, strict=True)
    ]
    return figure, rows


def _arm_names(metric: dict[str, dict[Arm, Cell]]) -> list[str]:
    """Name every arm of every question, so one chart can hold them all."""
    return [
        f"{question_id} {label}"
        for question_id, arms in metric.items()
        for label in _labels(list(arms))
    ]


def _write_distribution(
    experiment_id: str, question_id: str, arms: dict[Arm, Cell]
) -> Chart:
    """Draw one question's distribution and describe it for the index."""
    labels = _labels(list(arms))
    title = _distribution_title(question_id, list(arms), labels)
    file = f"{question_id}__distribution.svg"
    with matplotlib.rc_context(_STYLE):
        figure, rows = _distribution_figure(arms, labels, title)
        _write_svg(experiment_id, file, figure)
    return Chart(
        metric="distribution",
        question_id=question_id,
        file=file,
        title=title,
        row_label="category",
        arms=labels,
        columns=labels,
        rows=rows,
    )


def _write_rate(
    experiment_id: str, rate: RateChart, metric: dict[str, dict[Arm, Cell]]
) -> Chart:
    """Draw one per-experiment rate and describe it for the index."""
    names = _arm_names(metric)
    title = f"{experiment_id}: {rate.title} by arm"
    with matplotlib.rc_context(_STYLE):
        figure, rows = _rate_figure(rate, metric, names, title)
        _write_svg(experiment_id, rate.file, figure)
    return Chart(
        metric=rate.metric,
        question_id=None,
        file=rate.file,
        title=title,
        row_label="arm",
        arms=names,
        columns=[rate.title],
        rows=rows,
    )


def _write_index(experiment_id: str, charts: list[Chart]) -> Path:
    """Write the index the site reads for its chart list and its table views.

    The Chart dataclass is serialized wholesale, so a field added to it reaches
    the site instead of being silently dropped from the index.
    """
    body = {
        "experiment_id": experiment_id,
        "charts": [asdict(chart) for chart in charts],
    }
    path = _chart_dir(experiment_id) / _INDEX
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _write_svg(experiment_id: str, name: str, figure: Figure) -> Path:
    """Save one figure as the transparent, reproducible SVG the site embeds.

    The date matplotlib would otherwise stamp into the metadata is dropped, so
    redrawing unchanged aggregates rewrites an identical file instead of a diff.
    """
    path = _chart_dir(experiment_id) / name
    figure.savefig(path, format="svg", transparent=True, metadata={"Date": None})
    return path


def _chart_dir(experiment_id: str) -> Path:
    """The served directory an experiment's charts are written into."""
    directory = CHARTS_DIR / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory
