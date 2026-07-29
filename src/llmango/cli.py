"""Command line entry points for the llmango pipeline."""

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Annotated

import typer

from llmango import runner
from llmango.aggregate import AggregateOutcome, aggregate_question
from llmango.experiments import spec_for
from llmango.normalize import NormalizeOutcome, normalize_question
from llmango.pricing import guard_cost

if TYPE_CHECKING:
    from llmango.charts import AnalyzeOutcome

app = typer.Typer(help="Probe how LLM behavior shifts across languages.")

QuestionArgument = Annotated[str, typer.Argument(help="Question id (001a, 001b, ...).")]

_PIPELINE_ERRORS = (OSError, RuntimeError, ValueError)


def _reports_pipeline_errors[**Params](
    command: Callable[Params, None],
) -> Callable[Params, None]:
    """Report a pipeline failure as a message and a non-zero exit."""

    @wraps(command)
    def reporting(*args: Params.args, **kwargs: Params.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except _PIPELINE_ERRORS as error:
            typer.echo(str(error))
            raise typer.Exit(code=1) from error

    return reporting


@app.command()
@_reports_pipeline_errors
def run(
    question: QuestionArgument,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the question's model.")
    ] = None,
    samples: Annotated[
        int, typer.Option("--samples", "-n", min=1, help="Samples per arm.")
    ] = 1,
    lang: Annotated[
        list[str] | None, typer.Option("--lang", help="Restrict to these languages.")
    ] = None,
    batch: Annotated[
        bool, typer.Option("--batch", help="Submit via the OpenAI Batch API.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan without generating.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Allow a large paid run.")
    ] = False,
) -> None:
    """Run question across language-schema arms and persist raw results to Parquet."""
    planned = runner.plan(
        question, samples_per_arm=samples, model=model, languages=lang
    )
    _report_plan(planned)
    if dry_run:
        return
    guard_cost(planned.manifest.samples_total, force)
    _report_outcome(runner.run(planned, batch=batch))


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
    _report_normalize(normalize_question(question, force=force, dry_run=dry_run))


@app.command()
@_reports_pipeline_errors
def aggregate(question: QuestionArgument) -> None:
    """Aggregate one question's normalized answers into the JSON the charts read."""
    spec_for(question)
    _report_aggregate(aggregate_question(question))


@app.command()
@_reports_pipeline_errors
def analyze(question: QuestionArgument) -> None:
    """Draw the charts the site embeds from one question's aggregates."""
    from llmango.charts import analyze_question

    spec_for(question)
    _report_analyze(analyze_question(question))


@app.command(name="batch-fetch")
@_reports_pipeline_errors
def batch_fetch(
    run_id: Annotated[str, typer.Argument(help="Run id of a submitted batch.")],
) -> None:
    """Fetch a previously submitted batch and persist its results to Parquet."""
    outcome = runner.fetch_batch(run_id)
    typer.echo(f"Run {outcome.run_id}: wrote {outcome.rows_written} rows.")
    typer.echo(f"Parquet: {outcome.parquet_path}")


def _report_plan(plan: runner.RunPlan) -> None:
    """Report what a run would send, what it would cost and which arms it covers."""
    manifest = plan.manifest
    typer.echo(f"Plan for {manifest.question_id} via {manifest.provider}:")
    typer.echo(f"  model:       {manifest.model}")
    typer.echo(f"  temperature: {manifest.temperature}")
    typer.echo(f"  inputs:      {', '.join(sorted(manifest.inputs)) or 'none'}")
    typer.echo(
        f"  samples:     {manifest.samples_total} total, "
        f"{manifest.samples_per_arm} per arm"
    )
    if manifest.pricing is not None:
        price = manifest.pricing
        typer.echo(
            f"  price:       ${price.input}/1M in, ${price.output}/1M out "
            f"(updated {price.last_updated})"
        )
    else:
        typer.echo(
            f"  price:       no entry for {manifest.model}; add it to "
            f"data/pricing.json before running."
        )
    typer.echo(f"  arms:        {len(manifest.arms)}")
    for arm in manifest.arms:
        typer.echo(f"    {arm.label}  {arm.lang}")


def _report_outcome(outcome: runner.RunOutcome) -> None:
    """Report a run, which either submitted a batch or wrote its rows."""
    if outcome.batch_id is not None:
        typer.echo(f"Run {outcome.run_id}: submitted batch {outcome.batch_id}.")
        typer.echo(f"Fetch results with: llmango batch-fetch {outcome.run_id}")
        return
    typer.echo(f"Run {outcome.run_id}: wrote {outcome.rows_written} rows.")
    usage = outcome.manifest.usage
    if usage is not None:
        typer.echo(
            f"Usage:    {usage.total_tokens} tokens, ${usage.total_cost_usd:.6f}"
        )
    typer.echo(f"Parquet:  {outcome.parquet_path}")
    typer.echo(f"Manifest: {outcome.manifest_path}")


def _report_normalize(outcome: NormalizeOutcome) -> None:
    """Report how many answers were mapped, and how many needed the LLM."""
    written = outcome.parquet_path is not None
    resolved = "resolved by the LLM" if written else "would be resolved by the LLM"
    typer.echo(
        f"{outcome.rows} rows, {outcome.distinct} distinct answers, "
        f"{outcome.llm_calls} {resolved}."
    )
    if written:
        typer.echo(f"Parquet: {outcome.parquet_path}")


def _report_aggregate(outcome: AggregateOutcome) -> None:
    """Report where a question's aggregates landed."""
    typer.echo(f"Aggregate: {outcome.path}")


def _report_analyze(outcome: "AnalyzeOutcome") -> None:
    """Report every chart drawn for a question, and its index."""
    typer.echo(f"Drew {len(outcome.charts)} charts for {outcome.question_id}:")
    for chart in outcome.charts:
        typer.echo(f"  {chart.file}  {chart.metric}, {len(chart.arms)} arms")
    typer.echo(f"Index: {outcome.index_path}")


if __name__ == "__main__":
    app()
