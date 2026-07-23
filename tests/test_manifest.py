"""Tests for the run manifest: round-trip and content-hash stability."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llmango import manifest as manifest_module
from llmango.manifest import (
    RunManifest,
    collect_package_versions,
    find_manifest_by_content_hash,
    write_manifest,
)
from llmango.questions import SamplingParams


def _manifest(**overrides: Any) -> RunManifest:
    base: dict[str, Any] = {
        "run_id": "run-001",
        "question_id": "001_favorite_fruit",
        "backend": "openai",
        "model": "gpt-5.6-luna",
        "languages": ["en", "pl"],
        "sampling": SamplingParams(temperature=1.0, seed=7),
        "seed": 7,
        "samples": 5,
        "prompt_sha256": {"en": "aaa", "pl": "bbb"},
    }
    base.update(overrides)
    return RunManifest(**base)


def test_manifest_round_trips_through_json() -> None:
    manifest = _manifest()
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_content_hash_is_independent_of_run_id_and_timestamp() -> None:
    a = _manifest(run_id="run-a", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    b = _manifest(run_id="run-b", created_at=datetime(2030, 6, 6, tzinfo=UTC))
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_with_config() -> None:
    assert _manifest(samples=5).content_hash() != _manifest(samples=10).content_hash()


def test_write_and_find_manifest_by_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manifest_module, "RUNS_DIR", tmp_path)
    manifest = _manifest()

    path = write_manifest(manifest)
    assert path.exists()

    found = find_manifest_by_content_hash(manifest.content_hash())
    assert found is not None
    assert found.run_id == manifest.run_id


def test_collect_package_versions_reports_installed_packages() -> None:
    versions = collect_package_versions()
    assert versions["openai"]
    assert versions["pydantic"]
