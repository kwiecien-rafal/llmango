"""Run manifest for traceability.

Every run writes a manifest capturing the provider and model, its resolved
snapshot, the temperature, every arm with its prompt hash and response schema,
what the run consumed, and package versions, so any row can be traced back to the
exact configuration that produced it. It mirrors the question's own config, plus
what running it turned out to cost.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from llmango.config import RUNS_DIR
from llmango.inputs import InputDeclarations
from llmango.pricing import PricingEntry

_TRACKED_PACKAGES = (
    "openai",
    "pydantic",
    "polars",
    "pyarrow",
    "pyyaml",
    "typer",
    "python-dotenv",
    "huggingface-hub",
)

_RUN_ID_TIMESTAMP = "%Y%m%dT%H%M%S%f"


def collect_package_versions(
    packages: Iterable[str] = _TRACKED_PACKAGES,
) -> dict[str, str]:
    """Return installed versions for the tracked packages, keyed by name."""
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


class UsageTotals(BaseModel):
    """Rows, outcomes, tokens and cost for a whole run.

    The error and refusal counts sit next to the tokens because rows that failed
    or were refused carry no usage: without them a token total looks complete
    when it is only covering the rows that answered.

    provider_refusals counts only rows where the provider set its own refusal
    field. It is run provenance, not a reported metric: nothing downstream
    measures a refusal rate, and an answer that declines in plain language never
    reaches this count at all.
    """

    rows: int = 0
    errors: int = 0
    provider_refusals: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


class ArmRecord(BaseModel):
    """One arm of a run: the language and the schema it was asked under.

    response_schema is the whole JSON schema the arm was asked under, not a hash
    of it, so an edited schema shows up as a diff of what changed rather than as
    an opaque mismatch. Both schema fields are null for the free-text arm, which
    sends none.
    """

    lang: str
    schema_name: str | None = None
    response_schema: dict[str, Any] | None = None
    template_sha256: str


class Manifest(BaseModel):
    """A traceable record of one run's exact configuration and environment.

    One run covers every arm the question declares, so arms is what varies inside
    it and provider, model and temperature are what it holds constant.

    Prompts are rendered per sample from templates and the question's prompt
    inputs, so the manifest records each input's declaration and hashes the data
    file behind it alongside each arm's template, rather than storing a single
    static prompt. Those inputs plus the sample index reproduce every prompt.

    run_id and created_at are stamped when the run starts rather than when it is
    planned, so a plan that is only priced and never executed claims no id.
    """

    run_id: str = ""
    question_id: str
    provider: str
    model: str
    model_snapshot: str | None = None
    temperature: float
    samples: int
    arms: list[ArmRecord]
    inputs: InputDeclarations = Field(default_factory=dict)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    pricing: PricingEntry | None = None
    batch_id: str | None = None
    usage: UsageTotals | None = None
    package_versions: dict[str, str] = Field(default_factory=collect_package_versions)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def total_requests(self) -> int:
        """How many generations the run covers, across every arm."""
        return self.samples * len(self.arms)


def build_run_id(manifest: Manifest) -> str:
    """Build a run id from the question and the moment its run started.

    A question id and a timestamp name a run on their own, and everything else
    about it, its arms included, is in the manifest and in every row it wrote. The
    stamp carries milliseconds so that two runs of one question, started one after
    the other, cannot land on the same id. Basic ISO form sorts chronologically
    and contains no character a file name rejects.
    """
    stamp = manifest.created_at.astimezone(UTC).strftime(_RUN_ID_TIMESTAMP)[:-3]
    return f"{manifest.question_id}__{stamp}Z"


def manifest_path(run_id: str) -> Path:
    """Return the manifest path for a run id under runs/."""
    return RUNS_DIR / f"{run_id}.json"


def write_manifest(manifest: Manifest) -> Path:
    """Write a manifest to runs/<run_id>.json and return its path.

    A batch is written twice, once at submit and once at fetch to add the usage
    it turned out to consume, so an existing file for the same run id is the
    expected case and is replaced.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = manifest_path(manifest.run_id)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_manifest(run_id: str) -> Manifest:
    """Load a manifest by run id from runs/<run_id>.json."""
    path = manifest_path(run_id)
    if not path.is_file():
        raise FileNotFoundError(f"No manifest for run: {run_id}")
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
