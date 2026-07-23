"""Command line entry points for the llmango pipeline."""

from typing import Annotated

import typer

from llmango.backends.openai_backend import OpenAIBackend
from llmango.backends.openai_batch import OpenAIBatchBackend
from llmango.normalize import NormalizeOutcome, normalize_question
from llmango.runner import RunOutcome, fetch_batch, submit_batch
from llmango.runner import run as run_experiment

app = typer.Typer(help="Probe how LLM behavior shifts across languages.")

SMOKE_SAMPLE_LIMIT = 25


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
        int, typer.Option("--samples", "-n", help="Samples per language.")
    ] = 1,
    lang: Annotated[
        list[str] | None, typer.Option("--lang", help="Restrict to these languages.")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Sampling seed.")] = None,
    batch: Annotated[
        bool, typer.Option("--batch", help="Submit via the OpenAI Batch API.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Allow a large paid run.")
    ] = False,
) -> None:
    """Run one question across languages and persist raw results to Parquet."""
    if samples > SMOKE_SAMPLE_LIMIT and not force:
        typer.echo(
            f"Refusing a large run of {samples} samples per language without --force. "
            f"Smoke runs stay at or below {SMOKE_SAMPLE_LIMIT}."
        )
        raise typer.Exit(code=1)

    if batch:
        _report_submit(
            submit_batch(
                question_id,
                OpenAIBatchBackend(),
                model=model,
                samples=samples,
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
            samples=samples,
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
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    _report_normalize(outcome)


@app.command(name="batch-fetch")
def batch_fetch(
    run_id: Annotated[str, typer.Argument(help="Run id of a submitted batch.")],
) -> None:
    """Fetch a previously submitted batch and persist its results to Parquet."""
    try:
        outcome = fetch_batch(run_id, OpenAIBatchBackend())
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    typer.echo(f"Run {outcome.run_id}: wrote {outcome.rows_written} rows.")
    typer.echo(f"Parquet: {outcome.parquet_path}")


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


def _report_normalize(outcome: NormalizeOutcome) -> None:
    typer.echo(
        f"Normalized {outcome.rows} rows: {outcome.distinct} distinct answers, "
        f"{outcome.llm_calls} resolved by the LLM."
    )
    typer.echo(f"Parquet: {outcome.parquet_path}")


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
