"""Command line entry points for the llmango pipeline."""

from typing import Annotated

import typer

from llmango.backends.openai_backend import OpenAIBackend
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

    outcome = run_experiment(
        question_id,
        OpenAIBackend(),
        model=model,
        samples=samples,
        languages=lang,
        seed=seed,
    )

    if outcome.skipped:
        typer.echo(
            f"Skipped: an identical run already exists as {outcome.run_id}. "
            f"Results at {outcome.parquet_path}."
        )
        return

    typer.echo(f"Run {outcome.run_id}: wrote {outcome.rows_written} rows.")
    typer.echo(f"Parquet:  {outcome.parquet_path}")
    typer.echo(f"Manifest: {outcome.manifest_path}")


if __name__ == "__main__":
    app()
