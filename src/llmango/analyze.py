"""Write every experiment's named charts as SVGs the site embeds, and its tables."""

import importlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llmango.aggregate import Aggregate
from llmango.config import get_aggregate_path, get_charts_dir
from llmango.experiments import EXPERIMENTS
from llmango.plot import (
    NARROW,
    WIDE,
    Canvas,
    ChartDef,
    Drawn,
    Row,
    TableDef,
    save,
    styled,
)
from llmango.spec import ExperimentSpec

_INDEX = "index.json"
_ICONS = "icons"


@dataclass(frozen=True)
class Chart:
    """One written chart, and the numbers behind it the site puts in a table."""

    name: str
    number: str
    file: str
    narrow_file: str
    questions: list[str]
    title: str
    row_label: str
    unit: str
    columns: list[str]
    rows: list[Row]


@dataclass(frozen=True)
class Table:
    """One written table, which the site prints with no figure drawn beside it."""

    name: str
    number: str
    questions: list[str]
    title: str
    row_label: str
    unit: str
    columns: list[str]
    rows: list[Row]


@dataclass(frozen=True)
class AnalyzeOutcome:
    """One experiment's charts and tables, the ones it could not make, and its index."""

    experiment: str
    charts: list[Chart]
    tables: list[Table]
    skipped: list[str]
    index_path: Path | None


def analyze_all() -> list[AnalyzeOutcome]:
    """Draw every experiment's charts from whatever aggregates each one has."""
    outcomes = [_analyze(experiment) for experiment in EXPERIMENTS]

    if not any(outcome.charts or outcome.tables for outcome in outcomes):
        raise FileNotFoundError(
            "No aggregates to analyze. Run 'llmango aggregate <question_id>' first."
        )

    return outcomes


def _analyze(experiment: ExperimentSpec) -> AnalyzeOutcome:
    """Make one experiment's declared charts and tables, skipping what lacks data."""
    definitions = _definitions(experiment.folder)
    table_definitions = _table_definitions(experiment.folder)
    aggregates = _load(experiment.folder, experiment.questions)

    if not aggregates:
        return AnalyzeOutcome(
            experiment=experiment.folder,
            charts=[],
            tables=[],
            skipped=[
                definition.name for definition in (*definitions, *table_definitions)
            ],
            index_path=None,
        )

    directory = get_charts_dir(experiment.folder)
    directory.mkdir(parents=True, exist_ok=True)
    drawn: list[Chart] = []
    built: list[Table] = []
    skipped: list[str] = []
    for definition in definitions:
        if _readable(definition.questions, aggregates):
            drawn.append(_draw(definition, aggregates, directory))
        else:
            skipped.append(definition.name)
    for table_definition in table_definitions:
        if _readable(table_definition.questions, aggregates):
            built.append(_build(table_definition, aggregates, directory))
        else:
            skipped.append(table_definition.name)

    return AnalyzeOutcome(
        experiment=experiment.folder,
        charts=drawn,
        tables=built,
        skipped=skipped,
        index_path=_write_index(experiment.folder, drawn, built, directory),
    )


def _readable(questions: tuple[str, ...], aggregates: dict[str, Aggregate]) -> bool:
    """Whether every question a chart or table reads has been aggregated."""
    return all(question in aggregates for question in questions)


def _definitions(folder: str) -> tuple[ChartDef, ...]:
    """Read an experiment's declared charts, importing its module only now."""
    module = importlib.import_module(f"llmango.experiments.{folder}.charts")
    return cast(tuple[ChartDef, ...], module.CHARTS)


def _table_definitions(folder: str) -> tuple[TableDef, ...]:
    """Read an experiment's declared tables, from the module its charts live in."""
    module = importlib.import_module(f"llmango.experiments.{folder}.charts")
    return cast(tuple[TableDef, ...], module.TABLES)


def _load(folder: str, question_ids: tuple[str, ...]) -> dict[str, Aggregate]:
    """Read the aggregates an experiment's questions have, skipping those without."""
    found: dict[str, Aggregate] = {}
    for question_id in question_ids:
        aggregate_file = get_aggregate_path(folder, question_id)
        if aggregate_file.is_file():
            found[question_id] = cast(
                Aggregate, json.loads(aggregate_file.read_text(encoding="utf-8"))
            )

    return found


def _draw(
    definition: ChartDef, aggregates: dict[str, Aggregate], directory: Path
) -> Chart:
    """Draw one declared chart at both widths the page reads it at, and describe it."""
    read = {question: aggregates[question] for question in definition.questions}
    title = definition.numbered_title()
    file = f"{definition.name}.svg"
    narrow_file = f"{definition.name}--narrow.svg"

    drawn = _render(definition, read, title, WIDE)
    save(drawn.figure, directory / file)
    save(_render(definition, read, title, NARROW).figure, directory / narrow_file)

    return Chart(
        name=definition.name,
        number=definition.number,
        file=file,
        narrow_file=narrow_file,
        questions=list(definition.questions),
        title=drawn.title,
        row_label=drawn.row_label,
        unit=drawn.unit,
        columns=drawn.columns,
        rows=drawn.rows,
    )


def _render(
    definition: ChartDef, aggregates: dict[str, Aggregate], title: str, canvas: Canvas
) -> Drawn:
    """Draw one chart at one width, under the style every chart is drawn in."""
    with styled(canvas):
        return definition.draw(aggregates, title)


def _build(
    definition: TableDef, aggregates: dict[str, Aggregate], directory: Path
) -> Table:
    """Build one declared table from the questions it reads, drawing nothing."""
    read = {question: aggregates[question] for question in definition.questions}
    built = definition.build(read, definition.numbered_title())

    return Table(
        name=definition.name,
        number=definition.number,
        questions=list(definition.questions),
        title=built.title,
        row_label=built.row_label,
        unit=built.unit,
        columns=built.columns,
        rows=[_pictured(row, directory) for row in built.rows],
    )


def _pictured(row: Row, directory: Path) -> Row:
    """Write a row's picture out beside the charts, named the way the site reads one.

    A chart carries its pictures inside the SVG, because a drawing has to be one
    file that renders the same anywhere. A page is not one file, so a row points
    at the same vendored image instead of writing it out ten times over.
    """
    icon = row.get("icon")
    if not isinstance(icon, Path):
        return row

    into = directory / _ICONS
    into.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(icon, into / icon.name)

    return row | {"icon": f"{_ICONS}/{icon.name}"}


def _write_index(
    folder: str, charts: list[Chart], tables: list[Table], directory: Path
) -> Path:
    """Write the index the site reads for its chart lookup and its table views."""
    body = {
        "experiment": folder,
        "charts": [asdict(chart) for chart in charts],
        "tables": [asdict(table) for table in tables],
    }
    path = directory / _INDEX
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return path
