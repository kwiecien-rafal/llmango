"""Command line entry points for the llmango pipeline."""

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from llmango.analyze import AnalyzeOutcome
    from llmango.normalize import NormalizeOutcome
    from llmango.runner import RunOutcome, RunPlan

app = typer.Typer(help="Probe how LLM behavior shifts across languages.")

QuestionArgument = Annotated[str, typer.Argument(help="Question id (001a, 001b, ...).")]

_PIPELINE_ERRORS = (OSError, RuntimeError, ValueError)


def _reports_pipeline_errors[**Params](
    command: Callable[Params, None],
) -> Callable[Params, None]:
    """Report a pipeline failure as its type and message, and a non-zero exit."""

    @wraps(command)
    def reporting(*args: Params.args, **kwargs: Params.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except _PIPELINE_ERRORS as error:
            typer.echo(f"{type(error).__name__}: {error}")
            raise typer.Exit(code=1) from error

    return reporting


@app.command()
@_reports_pipeline_errors
def run(
    question: QuestionArgument,
    samples: Annotated[
        int, typer.Option("--samples", "-n", min=1, help="Samples per arm.")
    ] = 1,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan without generating.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Allow a large paid run.")
    ] = False,
) -> None:
    """Run question across language-schema arms, appending each result as it lands."""
    from llmango import runner

    planned = runner.plan(question, samples_per_arm=samples)
    _report_plan(planned)

    if dry_run:
        return

    _report_outcome(runner.run(planned, force=force))


@app.command()
@_reports_pipeline_errors
def normalize(
    question: QuestionArgument,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report LLM usage without calling it.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Allow a large paid normalization run.")
    ] = False,
) -> None:
    """Map raw answers to canonical categories and write a normalized Parquet file."""
    from llmango.normalize import normalize_question

    _report_normalize(normalize_question(question, force=force, dry_run=dry_run))


@app.command()
@_reports_pipeline_errors
def aggregate(question: QuestionArgument) -> None:
    """Aggregate one question's normalized answers into the JSON the charts read."""
    from llmango.aggregate import aggregate_question

    typer.echo(f"Aggregate: {aggregate_question(question)}")


@app.command()
@_reports_pipeline_errors
def analyze(question: QuestionArgument) -> None:
    """Draw the charts the site embeds from a question's experiment's aggregates."""
    from llmango.analyze import analyze_question

    _report_analyze(analyze_question(question))


def _report_plan(plan: "RunPlan") -> None:
    """Report what a run would send, what it would cost and which arms it covers."""
    question = plan.question
    price = plan.price
    typer.echo(f"Plan for {question.question_id} via {question.provider}:")
    typer.echo(f"  model:       {question.model}")
    typer.echo(f"  temperature: {question.temperature}")
    typer.echo(f"  inputs:      {', '.join(sorted(question.inputs)) or 'none'}")
    typer.echo(
        f"  samples:     {plan.samples_total} total, {plan.samples_per_arm} per arm"
    )
    if price is not None:
        typer.echo(
            f"  price:       ${price.input}/1M in, ${price.output}/1M out "
            f"(updated {price.last_updated})"
        )
    else:
        typer.echo(
            f"  price:       no entry for {question.model}; add it to "
            f"src/llmango/pricing.json before running."
        )
    typer.echo(f"  arms:        {len(question.arms)}")
    for arm in question.arms:
        typer.echo(f"    {arm.label}  {arm.lang}")


def _report_outcome(outcome: "RunOutcome") -> None:
    """Report what a run wrote, what it used and where both landed."""
    manifest = outcome.manifest
    usage = manifest.usage
    if outcome.finished:
        typer.echo(f"Run {outcome.run_id}: wrote {outcome.rows_written} rows.")
    else:
        typer.echo(
            f"Run {outcome.run_id}: stopped after {manifest.samples_written} of "
            f"{manifest.samples_total} rows. Rerun to add the rest."
        )
    typer.echo(f"Usage:    {usage.total_tokens} tokens, ${usage.total_cost_usd:.6f}")
    typer.echo(f"Results:  {outcome.results_path}")
    typer.echo(f"Manifest: {outcome.manifest_path}")


def _report_normalize(outcome: "NormalizeOutcome") -> None:
    """Report how many answers were mapped, and how many needed the LLM."""
    written = outcome.parquet_path is not None
    resolved = "resolved by the LLM" if written else "would be resolved by the LLM"
    typer.echo(
        f"{outcome.rows} rows, {outcome.distinct} distinct answers, "
        f"{outcome.llm_calls} {resolved}."
    )
    if written:
        typer.echo(f"Parquet: {outcome.parquet_path}")


def _report_analyze(outcome: "AnalyzeOutcome") -> None:
    """Report every chart drawn for an experiment, those skipped, and its index."""
    typer.echo(f"Drew {len(outcome.charts)} charts for {outcome.experiment}:")
    for chart in outcome.charts:
        typer.echo(
            f"  {chart.file}  {', '.join(chart.questions)}, {len(chart.columns)} arms"
        )
    if outcome.skipped:
        typer.echo(
            f"Skipped for want of aggregates: {', '.join(outcome.skipped)}. "
            f"Aggregate the questions they read first."
        )
    typer.echo(f"Index: {outcome.index_path}")


if __name__ == "__main__":
    app()
