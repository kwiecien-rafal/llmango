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
from llmango.pricing import PricingEntry
from llmango.questions import SamplingParams


def _manifest(**overrides: Any) -> RunManifest:
    base: dict[str, Any] = {
        "run_id": "run-001",
        "experiment_id": "001_fruit",
        "question_id": "001a",
        "backend": "openai",
        "model": "gpt-5.6-luna",
        "languages": ["en", "pl"],
        "sampling": SamplingParams(temperature=1.0, seed=7),
        "seed": 7,
        "samples": 5,
        "order": "fixed",
        "order_ids": ["apple", "mango"],
        "template_sha256": {"en": "aaa", "pl": "bbb"},
        "fruits_sha256": "ccc",
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


def test_content_hash_changes_with_schema_lang() -> None:
    assert (
        _manifest(schema_lang="en").content_hash()
        != _manifest(schema_lang="pl").content_hash()
    )


def test_content_hash_changes_with_order() -> None:
    reversed_ids = ["mango", "apple"]
    assert (
        _manifest(order_ids=["apple", "mango"]).content_hash()
        != _manifest(order_ids=reversed_ids).content_hash()
    )


def test_pricing_is_excluded_from_the_content_hash() -> None:
    cheap = PricingEntry(input=0.05, output=0.4, last_updated="2026-07-24")
    dear = PricingEntry(input=99.0, output=99.0, last_updated="2030-01-01")
    assert (
        _manifest(pricing=cheap).content_hash()
        == _manifest(pricing=dear).content_hash()
        == _manifest(pricing=None).content_hash()
    )


def test_manifest_round_trips_pricing() -> None:
    entry = PricingEntry(
        input=0.05, cached_input=0.005, output=0.4, last_updated="2026-07-24"
    )
    manifest = _manifest(pricing=entry)
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored.pricing == entry


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
