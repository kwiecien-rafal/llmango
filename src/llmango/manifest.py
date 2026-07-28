"""Run manifest: the exact configuration one run was executed under."""

from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from llmango.config import RUNS_DIR
from llmango.inputs import InputDeclarations
from llmango.pricing import PricingEntry
from llmango.spec import FREE_TEXT

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
    """Outcomes, tokens and cost, for one arm or for a whole run."""

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
    """One arm of a run: the schema and language it asked under, and what it used."""

    lang: str
    schema_name: str | None = None
    response_schema: dict[str, Any] | None = None
    template_sha256: str
    usage: UsageTotals | None = None

    @property
    def label(self) -> str:
        """The name this arm is reported under, FREE_TEXT when it sends no schema."""
        return self.schema_name or FREE_TEXT


class Manifest(BaseModel):
    """A traceable record of one run's exact configuration and environment."""

    run_id: str = ""
    question_id: str
    provider: str
    model: str
    temperature: float
    samples_total: int = 0
    samples_per_arm: int
    arms: list[ArmRecord]
    inputs: InputDeclarations = Field(default_factory=dict)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    pricing: PricingEntry | None = None
    batch_id: str | None = None
    usage: UsageTotals | None = None
    package_versions: dict[str, str] = Field(default_factory=collect_package_versions)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _count_samples(self) -> Self:
        """Total the samples a run covers, so the stored count can never drift."""
        self.samples_total = self.samples_per_arm * len(self.arms)
        return self


def build_run_id(manifest: Manifest) -> str:
    """Build a run id from the question and the millisecond its run started."""
    stamp = manifest.created_at.astimezone(UTC).strftime(_RUN_ID_TIMESTAMP)[:-3]
    return f"{manifest.question_id}__{stamp}Z"


def manifest_path(run_id: str) -> Path:
    """Return the manifest path for a run id under runs/."""
    return RUNS_DIR / f"{run_id}.json"


def write_manifest(manifest: Manifest) -> Path:
    """Write a manifest to runs/<run_id>.json, replacing a batch's earlier one."""
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
