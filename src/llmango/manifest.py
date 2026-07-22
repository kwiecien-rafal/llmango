"""Run manifest for traceability and idempotency.

Every run writes a manifest capturing the model, its resolved snapshot, the
backend, sampling params, per-language prompt hashes and package versions. The
content hash covers only the run configuration, so re-running the same config
produces the same hash and the runner can skip duplicate work.
"""

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel, Field

from llmango.config import RUNS_DIR
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

_CONTENT_EXCLUDE = {"run_id", "created_at", "model_snapshot", "package_versions"}


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


class RunManifest(BaseModel):
    """A traceable record of one run's exact configuration and environment."""

    run_id: str
    question_id: str
    backend: str
    model: str
    model_snapshot: str | None = None
    languages: list[str]
    sampling: SamplingParams
    seed: int | None = None
    samples: int
    prompt_sha256: dict[str, str]
    package_versions: dict[str, str] = Field(default_factory=collect_package_versions)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def content_hash(self) -> str:
        """Hash the run configuration, ignoring run id, timestamp and environment."""
        payload = self.model_dump(mode="json", exclude=_CONTENT_EXCLUDE)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def manifest_path(run_id: str) -> Path:
    """Return the manifest path for a run id under runs/."""
    return RUNS_DIR / f"{run_id}.json"


def write_manifest(manifest: RunManifest) -> Path:
    """Write a manifest to runs/<run_id>.json and return its path."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = manifest_path(manifest.run_id)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def find_manifest_by_content_hash(target_hash: str) -> RunManifest | None:
    """Return the first existing manifest whose content hash matches, if any."""
    if not RUNS_DIR.is_dir():
        return None
    for path in sorted(RUNS_DIR.glob("*.json")):
        manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.content_hash() == target_hash:
            return manifest
    return None
