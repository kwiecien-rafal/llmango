"""Run manifest for traceability and idempotency.

Every run writes a manifest capturing the model, its resolved snapshot, the
backend, sampling params, per-language prompt and schema hashes, what the run
consumed, and package versions. The content hash covers only the run
configuration, so re-running the same config produces the same hash and the
runner can skip duplicate work; measured outcomes are excluded from it.
"""

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from llmango.config import RUNS_DIR
from llmango.pricing import PricingEntry
from llmango.questions import SamplingParams

_TRACKED_PACKAGES = (
    "openai",
    "pydantic",
    "polars",
    "pyarrow",
    "pyyaml",
    "typer",
    "python-dotenv",
    "huggingface-hub",
    "lingua-language-detector",
)

_CONTENT_EXCLUDE = {
    "run_id",
    "created_at",
    "model_snapshot",
    "batch_id",
    "package_versions",
    "pricing",
    "schema_name",
    "total_requests",
    "usage",
}

_RUN_ID_TIMESTAMP = "%Y%m%dT%H%M%SZ"

_RUN_ID_HASH_CHARS = 6


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
    """Rows, outcomes, tokens and cost for one language or for a whole run.

    The error and refusal counts sit next to the tokens because rows that failed
    or were refused carry no usage: without them a token total looks complete
    when it is only covering the rows that answered.

    provider_refusals counts only rows where the provider set its own refusal
    field. It is deliberately not the experiment's refusal rate, which also
    covers answers that decline in plain language and cannot be known until
    normalization has run. Analyze owns that metric.
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


class RunUsage(BaseModel):
    """What a run actually consumed, in total and broken down per language.

    Measured after generation, so it is absent on a dry run and on a batch that
    has been submitted but not yet fetched.
    """

    measured_at: datetime
    total: UsageTotals
    by_language: dict[str, UsageTotals]


class RunManifest(BaseModel):
    """A traceable record of one run's exact configuration and environment.

    Prompts are rendered per sample from templates and the shared fruit table, so
    the manifest hashes those inputs plus the order strategy and the response
    schema rather than a single static prompt. The same inputs plus seed and
    sample index reproduce every prompt exactly.
    """

    run_id: str
    experiment_id: str
    question_id: str
    backend: str
    model: str
    model_snapshot: str | None = None
    pricing: PricingEntry | None = None
    batch_id: str | None = None
    schema_variant: str = "en"
    schema_name: str | None = None
    schema_sha256: str | None = None
    languages: list[str]
    sampling: SamplingParams
    seed: int | None = None
    samples_per_language: int
    order: str
    order_ids: list[str] | None = None
    template_sha256: dict[str, str]
    fruits_sha256: str
    usage: RunUsage | None = None
    package_versions: dict[str, str] = Field(default_factory=collect_package_versions)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def total_requests(self) -> int:
        """How many generations the run covers, across every language."""
        return self.samples_per_language * len(self.languages)

    def content_hash(self) -> str:
        """Hash the run configuration, ignoring run id, timestamp and environment."""
        payload = self.model_dump(mode="json", exclude=_CONTENT_EXCLUDE)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_run_id(manifest: RunManifest) -> str:
    """Build a run id that sorts usefully in a folder listing.

    Question and schema variant first keep every run of one arm together, and let
    the raw parquet still be found by its question prefix. The timestamp is in
    basic ISO form, which sorts chronologically and contains no character a file
    name rejects. The trailing content hash both prevents collisions and makes
    two runs of the same configuration visibly related.
    """
    stamp = manifest.created_at.astimezone(UTC).strftime(_RUN_ID_TIMESTAMP)
    digest = manifest.content_hash()[:_RUN_ID_HASH_CHARS]
    return f"{manifest.question_id}__{manifest.schema_variant}__{stamp}__{digest}"


def manifest_path(run_id: str) -> Path:
    """Return the manifest path for a run id under runs/."""
    return RUNS_DIR / f"{run_id}.json"


def write_manifest(manifest: RunManifest) -> Path:
    """Write a manifest to runs/<run_id>.json and return its path.

    Refuses to replace the record of a differently configured run, so a run id
    collision fails loudly instead of orphaning a parquet file. Rewriting the
    same run to add its measured usage is allowed, since that leaves the
    configuration untouched.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = manifest_path(manifest.run_id)
    if path.is_file():
        existing = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if existing.content_hash() != manifest.content_hash():
            raise ValueError(
                f"Manifest {path.name} already records a different run "
                f"configuration. Refusing to overwrite it."
            )
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_manifest(run_id: str) -> RunManifest:
    """Load a manifest by run id from runs/<run_id>.json."""
    path = manifest_path(run_id)
    if not path.is_file():
        raise FileNotFoundError(f"No manifest for run: {run_id}")
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def find_manifest_by_content_hash(target_hash: str) -> RunManifest | None:
    """Return the first existing manifest whose content hash matches, if any."""
    if not RUNS_DIR.is_dir():
        return None
    for path in sorted(RUNS_DIR.glob("*.json")):
        manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.content_hash() == target_hash:
            return manifest
    return None
