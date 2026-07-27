"""Command line entry points for the llmango pipeline.

llmango.charts is imported inside the analyze command rather than here, because
importing it pulls in matplotlib, which costs roughly half a second of startup
that every other command would pay without ever drawing anything.
"""

from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer

from llmango.aggregate import AggregateOutcome, aggregate_question
from llmango.backends.openai_backend import OpenAIBackend
from llmango.backends.openai_batch import OpenAIBatchBackend
from llmango.experiments import spec_for
from llmango.normalize import NormalizeOutcome, normalize_question
from llmango.questions import load_question
from llmango.runner import RunOutcome, RunPlan, fetch_batch, plan_run, submit_batch
from llmango.runner import run as run_experiment

if TYPE_CHECKING:
    from llmango.charts import AnalyzeOutcome

app = typer.Typer(help="Probe how LLM behavior shifts across languages.")

SMOKE_SAMPLES = 5
SMOKE_SAMPLE_LIMIT = 25

QuestionArgument = Annotated[str, typer.Argument(help="Question id (001a, 001b, ...).")]

_PIPELINE_ERRORS = (OSError, RuntimeError, ValueError, KeyError)


@app.callback()
def main() -> None:
    """Probe how LLM behavior shifts across languages."""


@app.command()
def run(
    question: QuestionArgument = "001a",
    model: Annotated[
        str | None, typer.Option("--model", help="Override the experiment model.")
    ] = None,
    samples: Annotated[
        int | None, typer.Option("--samples", "-n", help="Samples per language.")
    ] = None,
    lang: Annotated[
        list[str] | None, typer.Option("--lang", help="Restrict to these languages.")
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for per-sample prompt inputs.")
    ] = None,
    batch: Annotated[
        bool, typer.Option("--batch", help="Submit via the OpenAI Batch API.")
    ] = False,
    smoke: Annotated[
        bool, typer.Option("--smoke", help=f"Tiny {SMOKE_SAMPLES}-sample smoke run.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan without generating.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Allow a large paid run.")
    ] = False,
) -> None:
    """Run one question across languages and persist raw results to Parquet.

    A question declaring several schema variants (e.g. 001d) runs once per
    variant, each tagged with its schema language.
    """
    count = _resolve_samples(samples, smoke, dry_run, force)
    try:
        variants = load_question(question).schema_variants
        for schema_variant in variants:
            _run_variant(
                question, schema_variant, count, model, lang, seed, batch, dry_run
            )
    except _PIPELINE_ERRORS as error:
        _die(str(error))


def _run_variant(
    question: str,
    schema_variant: str,
    samples: int,
    model: str | None,
    lang: list[str] | None,
    seed: int | None,
    batch: bool,
    dry_run: bool,
) -> None:
    """Run or preview one question for a single schema variant."""
    opts: dict[str, Any] = {
        "model": model,
        "samples": samples,
        "languages": lang,
        "seed": seed,
        "schema_variant": schema_variant,
    }
    if dry_run:
        backend = OpenAIBatchBackend if batch else OpenAIBackend
        _report_plan(plan_run(question, backend.backend_id, **opts))
    elif batch:
        _report_submit(submit_batch(question, OpenAIBatchBackend(), **opts))
    else:
        _report_run(run_experiment(question, OpenAIBackend(), **opts))


@app.command()
def normalize(
    question: QuestionArgument = "001a",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override the normalization model."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report LLM usage without calling it.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Allow a large paid normalization run.")
    ] = False,
) -> None:
    """Map raw answers to canonical categories and write a normalized Parquet file."""
    try:
        outcome = normalize_question(
            question,
            make_backend=OpenAIBackend,
            model=model,
            max_llm_calls=None if force else SMOKE_SAMPLE_LIMIT,
            dry_run=dry_run,
        )
    except _PIPELINE_ERRORS as error:
        _die(str(error))
    _report_normalize(outcome)


@app.command()
def aggregate(question: QuestionArgument = "001a") -> None:
    """Aggregate one question's normalized answers into the JSON the charts read."""
    _check_question(question)
    try:
        outcome = aggregate_question(question)
    except _PIPELINE_ERRORS as error:
        _die(str(error))
    _report_aggregate(outcome)


@app.command()
def analyze(question: QuestionArgument = "001a") -> None:
    """Draw the charts the site embeds from one question's aggregates."""
    from llmango.charts import analyze_question

    _check_question(question)
    try:
        outcome = analyze_question(question)
    except _PIPELINE_ERRORS as error:
        _die(str(error))
    _report_analyze(outcome)


@app.command(name="batch-fetch")
def batch_fetch(
    run_id: Annotated[str, typer.Argument(help="Run id of a submitted batch.")],
) -> None:
    """Fetch a previously submitted batch and persist its results to Parquet."""
    try:
        outcome = fetch_batch(run_id, OpenAIBatchBackend())
    except _PIPELINE_ERRORS as error:
        _die(str(error))
    typer.echo(f"Run {outcome.run_id}: wrote {outcome.rows_written} rows.")
    typer.echo(f"Parquet: {outcome.parquet_path}")


def _resolve_samples(
    samples: int | None, smoke: bool, dry_run: bool, force: bool
) -> int:
    """Resolve the sample count, applying the smoke preset and the cost guardrail."""
    if smoke and samples is not None:
        _die("Pass either --smoke or --samples, not both.")
    if smoke:
        return SMOKE_SAMPLES
    count = samples if samples is not None else 1
    if not dry_run and count > SMOKE_SAMPLE_LIMIT and not force:
        _die(
            f"Refusing a large run of {count} samples per language without --force. "
            f"Smoke runs stay at or below {SMOKE_SAMPLE_LIMIT}."
        )
    return count


def _check_question(question: str) -> None:
    """Reject an id no experiment declares, listing the ones that exist.

    Aggregate and analyze read a question's own files and never need its spec, so
    without this they would report a missing file for a question that was never a
    question at all.
    """
    try:
        spec_for(question)
    except ValueError as error:
        _die(str(error))


def _die(message: str) -> NoReturn:
    """Print an error message and exit with a non-zero status."""
    typer.echo(message)
    raise typer.Exit(code=1)


def _report_run(outcome: RunOutcome) -> None:
    if outcome.skipped:
        typer.echo(
            f"Skipped: an identical run already exists as {outcome.run_id}. "
            f"Results at {outcome.parquet_path}."
        )
        return
    typer.echo(
        f"Run {outcome.run_id} (schema {outcome.manifest.schema_variant}): "
        f"wrote {outcome.rows_written} rows."
    )
    usage = outcome.manifest.usage
    if usage is not None:
        typer.echo(
            f"Usage:    {usage.total.total_tokens} tokens, "
            f"${usage.total.total_cost_usd:.6f}"
        )
    typer.echo(f"Parquet:  {outcome.parquet_path}")
    typer.echo(f"Manifest: {outcome.manifest_path}")


def _report_plan(plan: RunPlan) -> None:
    manifest = plan.manifest
    typer.echo(f"Dry run for {manifest.question_id} via {manifest.backend}:")
    typer.echo(f"  model:     {manifest.model}")
    schema = manifest.schema_name or "free text"
    typer.echo(f"  schema:    {manifest.schema_variant} ({schema})")
    typer.echo(f"  inputs:    {', '.join(sorted(manifest.inputs)) or 'none'}")
    typer.echo(f"  languages: {', '.join(manifest.languages)}")
    typer.echo(f"  samples:   {manifest.samples_per_language} per language")
    typer.echo(f"  requests:  {manifest.total_requests} total")
    if plan.pricing is not None:
        price = plan.pricing
        typer.echo(
            f"  price:     ${price.input}/1M in, ${price.output}/1M out "
            f"(updated {price.last_updated})"
        )
    else:
        typer.echo(
            f"  price:     no entry for {manifest.model}; add it to "
            f"data/pricing.json before running."
        )
    if plan.duplicate is not None:
        typer.echo(
            f"  duplicate: run {plan.duplicate.run_id} already covers this; "
            f"it would be skipped."
        )
    else:
        typer.echo("  duplicate: none; results would be generated and written.")


def _report_normalize(outcome: NormalizeOutcome) -> None:
    written = outcome.parquet_path is not None
    resolved = "resolved by the LLM" if written else "would be resolved by the LLM"
    typer.echo(
        f"{outcome.rows} rows, {outcome.distinct} distinct answers, "
        f"{outcome.llm_calls} {resolved}."
    )
    if written:
        typer.echo(f"Parquet: {outcome.parquet_path}")


def _report_aggregate(outcome: AggregateOutcome) -> None:
    typer.echo(f"Aggregate: {outcome.path}")


def _report_analyze(outcome: "AnalyzeOutcome") -> None:
    typer.echo(f"Drew {len(outcome.charts)} charts for {outcome.question_id}:")
    for chart in outcome.charts:
        typer.echo(f"  {chart.file}  {chart.metric}, {len(chart.arms)} arms")
    typer.echo(f"Index: {outcome.index_path}")


def _report_submit(outcome: RunOutcome) -> None:
    if outcome.skipped:
        typer.echo(
            f"Skipped: an identical run already exists as {outcome.run_id} "
            f"(batch {outcome.batch_id})."
        )
        return
    typer.echo(
        f"Run {outcome.run_id} (schema {outcome.manifest.schema_variant}): "
        f"submitted batch {outcome.batch_id}."
    )
    typer.echo(f"Fetch results with: llmango batch-fetch {outcome.run_id}")


if __name__ == "__main__":
    app()
