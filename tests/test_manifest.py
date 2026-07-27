"""Tests for the run manifest: round-trip and content-hash stability."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llmango import manifest as manifest_module
from llmango.manifest import (
    RunManifest,
    RunUsage,
    UsageTotals,
    build_run_id,
    collect_package_versions,
    find_manifest_by_content_hash,
    write_manifest,
)
from llmango.pricing import PricingEntry
from llmango.questions import SamplingParams


def _manifest(**overrides: Any) -> RunManifest:
    base: dict[str, Any] = {
        "run_id": "run-001",
        "question_id": "001a",
        "backend": "openai",
        "model": "gpt-5.6-luna",
        "schema_variant": "en",
        "languages": ["en", "pl"],
        "sampling": SamplingParams(temperature=1.0, seed=7),
        "seed": 7,
        "samples_per_language": 5,
        "inputs": {"fruit_list": {"order": "fixed", "order_ids": ["apple", "mango"]}},
        "template_sha256": {"en": "aaa", "pl": "bbb"},
        "input_sha256": {"fruit_list": "ccc"},
    }
    base.update(overrides)
    return RunManifest(**base)


def _usage() -> RunUsage:
    totals = UsageTotals(rows=10, prompt_tokens=120, total_cost_usd=0.000213)
    return RunUsage(
        measured_at=datetime(2026, 7, 25, tzinfo=UTC),
        total=totals,
        by_language={"en": totals},
    )


def test_manifest_round_trips_through_json() -> None:
    manifest = _manifest()
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_content_hash_is_independent_of_run_id_and_timestamp() -> None:
    a = _manifest(run_id="run-a", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    b = _manifest(run_id="run-b", created_at=datetime(2030, 6, 6, tzinfo=UTC))
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_with_config() -> None:
    assert (
        _manifest(samples_per_language=5).content_hash()
        != _manifest(samples_per_language=10).content_hash()
    )


def test_content_hash_changes_with_schema_variant() -> None:
    assert (
        _manifest(schema_variant="en").content_hash()
        != _manifest(schema_variant="pl").content_hash()
    )


def test_content_hash_changes_with_the_schema_itself() -> None:
    assert (
        _manifest(schema_sha256="aaa").content_hash()
        != _manifest(schema_sha256="bbb").content_hash()
    )


def test_schema_name_and_usage_are_excluded_from_the_content_hash() -> None:
    plain = _manifest()
    assert (
        _manifest(schema_name="FruitChoice").content_hash()
        == _manifest(schema_name="Renamed").content_hash()
        == _manifest(usage=_usage()).content_hash()
        == plain.content_hash()
    )


def test_total_requests_covers_every_language() -> None:
    manifest = _manifest(languages=["en", "pl", "ja"], samples_per_language=5)

    assert manifest.total_requests == 15
    assert "total_requests" in manifest.model_dump_json()


def test_manifest_round_trips_usage() -> None:
    manifest = _manifest(usage=_usage())
    restored = RunManifest.model_validate_json(manifest.model_dump_json())

    assert restored.usage is not None
    assert restored.usage.total.total_cost_usd == 0.000213
    assert restored.usage.by_language["en"].prompt_tokens == 120


def test_run_id_sorts_by_question_arm_then_time() -> None:
    early = _manifest(created_at=datetime(2026, 7, 25, 14, 25, 30, tzinfo=UTC))
    late = _manifest(created_at=datetime(2026, 7, 26, 9, 12, 4, tzinfo=UTC))

    early_id = build_run_id(early)

    assert early_id.startswith("001a__en__20260725T142530Z__")
    assert early_id < build_run_id(late)
    assert not set(early_id) & set(":/\\ ")


def test_run_id_hash_tracks_the_configuration() -> None:
    started = datetime(2026, 7, 25, tzinfo=UTC)

    assert build_run_id(_manifest(created_at=started)) == build_run_id(
        _manifest(created_at=started, run_id="a-different-id")
    )
    assert build_run_id(_manifest(created_at=started)) != build_run_id(
        _manifest(created_at=started, samples_per_language=99)
    )


def test_content_hash_changes_with_an_input_declaration() -> None:
    forward = {"fruit_list": {"order": "fixed", "order_ids": ["apple", "mango"]}}
    reversed_ids = {"fruit_list": {"order": "fixed", "order_ids": ["mango", "apple"]}}
    assert (
        _manifest(inputs=forward).content_hash()
        != _manifest(inputs=reversed_ids).content_hash()
    )


def test_content_hash_changes_with_an_input_file() -> None:
    assert (
        _manifest(input_sha256={"fruit_list": "ccc"}).content_hash()
        != _manifest(input_sha256={"fruit_list": "ddd"}).content_hash()
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


def test_rewriting_a_run_to_add_usage_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manifest_module, "RUNS_DIR", tmp_path)
    manifest = _manifest()
    write_manifest(manifest)

    manifest.usage = _usage()
    path = write_manifest(manifest)

    assert RunManifest.model_validate_json(path.read_text("utf-8")).usage is not None


def test_writing_over_a_different_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manifest_module, "RUNS_DIR", tmp_path)
    write_manifest(_manifest())

    with pytest.raises(ValueError, match="different run configuration"):
        write_manifest(_manifest(samples_per_language=99))


def test_collect_package_versions_reports_installed_packages() -> None:
    versions = collect_package_versions()
    assert versions["openai"]
    assert versions["pydantic"]
