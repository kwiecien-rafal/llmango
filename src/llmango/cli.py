"""Command line entry points for the llmango pipeline."""

from typing import Annotated, NoReturn

import typer

from llmango.analyze import AnalyzeOutcome, analyze_question
from llmango.backends.openai_backend import OpenAIBackend
from llmango.backends.openai_batch import OpenAIBatchBackend
from llmango.normalize import NormalizeOutcome, normalize_question
from llmango.runner import RunOutcome, RunPlan, fetch_batch, plan_run, submit_batch
from llmango.runner import run as run_experiment

app = typer.Typer(help="Probe how LLM behavior shifts across languages.")

SMOKE_SAMPLES = 5
SMOKE_SAMPLE_LIMIT = 25

_PIPELINE_ERRORS = (OSError, RuntimeError, ValueError, KeyError)


@app.callback()
def main() -> None:
    """Probe how LLM behavior shifts across languages."""


@app.command()
def run(
    question_id: Annotated[
        str, typer.Argument(help="Question to run.")
    ] = "favorite_fruit",
    model: Annotated[
        str | None, typer.Option("--model", help="Override the meta.yaml model.")
    ] = None,
    samples: Annotated[
        int | None, typer.Option("--samples", "-n", help="Samples per language.")
    ] = None,
    lang: Annotated[
        list[str] | None, typer.Option("--lang", help="Restrict to these languages.")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Sampling seed.")] = None,
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
    """Run one question across languages and persist raw results to Parquet."""
    count = _resolve_samples(samples, smoke, dry_run, force)

    if dry_run:
        backend = OpenAIBatchBackend if batch else OpenAIBackend
        _report_plan(
            plan_run(
                question_id,
                backend.backend_id,
                model=model,
                samples=count,
                languages=lang,
                seed=seed,
            )
        )
        return

    if batch:
        _report_submit(
            submit_batch(
                question_id,
                OpenAIBatchBackend(),
                model=model,
                samples=count,
                languages=lang,
                seed=seed,
            )
        )
        return

    _report_run(
        run_experiment(
            question_id,
            OpenAIBackend(),
            model=model,
            samples=count,
            languages=lang,
            seed=seed,
        )
    )


@app.command()
def normalize(
    question_id: Annotated[
        str, typer.Argument(help="Question to normalize.")
    ] = "favorite_fruit",
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
            question_id,
            make_backend=OpenAIBackend,
            model=model,
            max_llm_calls=None if force else SMOKE_SAMPLE_LIMIT,
            dry_run=dry_run,
        )
    except _PIPELINE_ERRORS as error:
        _die(str(error))
    _report_normalize(outcome)


@app.command()
def analyze(
    question_id: Annotated[
        str, typer.Argument(help="Question to analyze.")
    ] = "favorite_fruit",
) -> None:
    """Aggregate normalized answers into the committed JSON the site reads."""
    try:
        outcome = analyze_question(question_id)
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
    typer.echo(f"Run {outcome.run_id}: wrote {outcome.rows_written} rows.")
    typer.echo(f"Parquet:  {outcome.parquet_path}")
    typer.echo(f"Manifest: {outcome.manifest_path}")


def _report_plan(plan: RunPlan) -> None:
    manifest = plan.manifest
    requests = len(manifest.languages) * manifest.samples
    typer.echo(f"Dry run for {manifest.question_id} via {manifest.backend}:")
    typer.echo(f"  model:     {manifest.model}")
    typer.echo(f"  languages: {', '.join(manifest.languages)}")
    typer.echo(f"  samples:   {manifest.samples} per language")
    typer.echo(f"  requests:  {requests} total")
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


def _report_analyze(outcome: AnalyzeOutcome) -> None:
    typer.echo(f"Wrote {len(outcome.paths)} aggregate files:")
    for path in outcome.paths:
        typer.echo(f"  {path}")


def _report_submit(outcome: RunOutcome) -> None:
    if outcome.skipped:
        typer.echo(
            f"Skipped: an identical run already exists as {outcome.run_id} "
            f"(batch {outcome.batch_id})."
        )
        return
    typer.echo(f"Run {outcome.run_id}: submitted batch {outcome.batch_id}.")
    typer.echo(f"Fetch results with: llmango batch-fetch {outcome.run_id}")


if __name__ == "__main__":
    app()
