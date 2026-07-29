"""Draw one experiment's named charts as the SVGs the site embeds.

Takes a question id, resolves the experiment owning it, reads the committed
aggregates of every question that experiment declares, and writes one SVG per
chart into site/public/charts/<experiment>/, which Astro serves verbatim,
alongside an index carrying the numbers behind every chart so the site can
render a table beside each image.

A chart is a named artifact the experiment declares, not one per question, so it
may read several questions at once: that is what lets 001b be plotted against
the 001a it exists to be read against. A chart whose questions are not all
aggregated yet is reported as skipped rather than drawn short.

The experiment's charts module is imported only here, which keeps matplotlib off
the path of every other command.
"""

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llmango.aggregate import Aggregate
from llmango.config import AGG_DIR, CHARTS_DIR
from llmango.experiments import spec_for
from llmango.plot import ChartDef, Row, save, styled

_INDEX = "index.json"


@dataclass(frozen=True)
class Chart:
    """One written chart, and the numbers behind it the site puts in a table."""

    name: str
    file: str
    questions: list[str]
    title: str
    row_label: str
    columns: list[str]
    rows: list[Row]


@dataclass(frozen=True)
class AnalyzeOutcome:
    """The charts one analysis run drew, the ones it could not, and its index."""

    experiment: str
    charts: list[Chart]
    skipped: list[str]
    index_path: Path


def analyze_question(question_id: str) -> AnalyzeOutcome:
    """Draw the charts of the experiment owning a question, from its aggregates."""
    spec = spec_for(question_id)
    aggregates = _load(spec.questions)
    if not aggregates:
        raise FileNotFoundError(
            f"No data for question {question_id} to analyze. "
            f"Run 'llmango aggregate {question_id}' first."
        )
    definitions = _definitions(spec.folder)
    directory = _chart_dir(spec.folder)

    drawn: list[Chart] = []
    skipped: list[str] = []
    with styled():
        for definition in definitions:
            if all(question in aggregates for question in definition.questions):
                drawn.append(_draw(definition, aggregates, directory))
            else:
                skipped.append(definition.name)

    return AnalyzeOutcome(
        experiment=spec.folder,
        charts=drawn,
        skipped=skipped,
        index_path=_write_index(spec.folder, drawn, directory),
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
    """Draw one declared chart, save it, and describe it for the index.

    The hook is handed the questions it declared and nothing else, so what a
    chart reads is what analyze checked was there.
    """
    drawn = definition.draw(
        {question: aggregates[question] for question in definition.questions}
    )
    file = f"{definition.name}.svg"
    save(drawn.figure, directory / file)
    return Chart(
        name=definition.name,
        file=file,
        questions=list(definition.questions),
        title=drawn.title,
        row_label=drawn.row_label,
        columns=drawn.columns,
        rows=drawn.rows,
    )


def _write_index(folder: str, charts: list[Chart], directory: Path) -> Path:
    """Write the index the site reads for its chart lookup and its table views.

    The Chart dataclass is serialized wholesale, so a field added to it reaches
    the site instead of being silently dropped from the index.
    """
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
