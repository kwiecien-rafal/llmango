"""Draw every experiment's named charts as the SVGs the site embeds."""

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llmango.aggregate import Aggregate
from llmango.config import AGG_DIR, CHARTS_DIR
from llmango.experiments import EXPERIMENTS
from llmango.plot import ChartDef, Row, save, styled
from llmango.spec import ExperimentSpec

_INDEX = "index.json"


@dataclass(frozen=True)
class Chart:
    """One written chart, and the numbers behind it the site puts in a table."""

    name: str
    number: str
    file: str
    questions: list[str]
    title: str
    row_label: str
    unit: str
    columns: list[str]
    rows: list[Row]


@dataclass(frozen=True)
class AnalyzeOutcome:
    """One experiment's charts, the ones it could not draw, and its index."""

    experiment: str
    charts: list[Chart]
    skipped: list[str]
    index_path: Path | None


def analyze_all() -> list[AnalyzeOutcome]:
    """Draw every experiment's charts from whatever aggregates each one has."""
    with styled():
        outcomes = [_analyze(experiment) for experiment in EXPERIMENTS]

    if not any(outcome.charts for outcome in outcomes):
        raise FileNotFoundError(
            "No aggregates to analyze. Run 'llmango aggregate <question_id>' first."
        )

    return outcomes


def _analyze(experiment: ExperimentSpec) -> AnalyzeOutcome:
    """Draw one experiment's declared charts, skipping those missing an aggregate."""
    definitions = _definitions(experiment.folder)
    aggregates = _load(experiment.questions)

    if not aggregates:
        return AnalyzeOutcome(
            experiment=experiment.folder,
            charts=[],
            skipped=[definition.name for definition in definitions],
            index_path=None,
        )

    directory = _chart_dir(experiment.folder)
    drawn: list[Chart] = []
    skipped: list[str] = []
    for definition in definitions:
        if all(question in aggregates for question in definition.questions):
            drawn.append(_draw(definition, aggregates, directory))
        else:
            skipped.append(definition.name)

    return AnalyzeOutcome(
        experiment=experiment.folder,
        charts=drawn,
        skipped=skipped,
        index_path=_write_index(experiment.folder, drawn, directory),
    )


def _definitions(folder: str) -> tuple[ChartDef, ...]:
    """Read an experiment's declared charts, importing its module only now."""
    module = importlib.import_module(f"llmango.experiments.{folder}.charts")
    return cast(tuple[ChartDef, ...], module.CHARTS)


def _load(question_ids: tuple[str, ...]) -> dict[str, Aggregate]:
    """Read the aggregates an experiment's questions have, skipping those without."""
    found: dict[str, Aggregate] = {}
    for question_id in question_ids:
        path = AGG_DIR / f"{question_id}.json"
        if path.is_file():
            found[question_id] = cast(
                Aggregate, json.loads(path.read_text(encoding="utf-8"))
            )

    return found


def _draw(
    definition: ChartDef, aggregates: dict[str, Aggregate], directory: Path
) -> Chart:
    """Draw one declared chart, save it, and describe it for the index."""
    drawn = definition.draw(
        {question: aggregates[question] for question in definition.questions},
        definition.numbered_title(),
    )
    file = f"{definition.name}.svg"
    save(drawn.figure, directory / file)

    return Chart(
        name=definition.name,
        number=definition.number,
        file=file,
        questions=list(definition.questions),
        title=drawn.title,
        row_label=drawn.row_label,
        unit=drawn.unit,
        columns=drawn.columns,
        rows=drawn.rows,
    )


def _write_index(folder: str, charts: list[Chart], directory: Path) -> Path:
    """Write the index the site reads for its chart lookup and its table views."""
    body = {"experiment": folder, "charts": [asdict(chart) for chart in charts]}
    path = directory / _INDEX
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return path


def _chart_dir(folder: str) -> Path:
    """The served directory an experiment's charts are written into."""
    directory = CHARTS_DIR / folder
    directory.mkdir(parents=True, exist_ok=True)

    return directory
