"""Tests for the run manifest: round-trip, run ids and persistence."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llmango import config as config_module
from llmango.manifest import (
    ArmRecord,
    Manifest,
    UsageTotals,
    build_run_id,
    collect_package_versions,
    write_manifest,
)
from llmango.pricing import PricingEntry


def _arms(*languages: str) -> list[ArmRecord]:
    return [
        ArmRecord(
            lang=lang,
            schema_name="FruitChoice",
            response_schema={"title": "FruitChoice"},
            template_sha256=f"sha-{lang}",
            usage=_usage(),
        )
        for lang in languages
    ]


_PRICING = PricingEntry(
    input=0.05, cached_input=0.005, output=0.4, last_updated="2026-07-24"
)


def _manifest(**overrides: Any) -> Manifest:
    base: dict[str, Any] = {
        "run_id": "run-001",
        "question_id": "001a",
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "temperature": 1.0,
        "samples_total": 10,
        "samples_per_arm": 5,
        "arms": _arms("en", "pl"),
        "inputs": {"fruit_list": {"order": "fixed", "order_ids": ["apple", "mango"]}},
        "input_sha256": {"fruit_list": "ccc"},
        "pricing": _PRICING,
        "usage": _usage(),
        "created_at": datetime(2026, 7, 25, 14, 25, 30, 42000, tzinfo=UTC),
    }
    base.update(overrides)
    return Manifest(**base)


def _usage() -> UsageTotals:
    return UsageTotals(prompt_tokens=120, total_cost_usd=0.000213)


def test_manifest_round_trips_through_json() -> None:
    manifest = _manifest()
    restored = Manifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_an_arm_records_the_schema_it_was_asked_under() -> None:
    """The free-text arm records no schema, which is what tells it apart."""
    free_text = ArmRecord(lang="pl", template_sha256="sha-pl", usage=_usage())
    manifest = _manifest(arms=[*_arms("pl"), free_text])

    restored = Manifest.model_validate_json(manifest.model_dump_json())

    assert restored.arms[1].schema_name is None
    assert restored.arms[1].response_schema is None


def test_manifest_round_trips_usage_and_pricing() -> None:
    manifest = _manifest()
    restored = Manifest.model_validate_json(manifest.model_dump_json())

    assert restored.usage.total_cost_usd == 0.000213
    assert restored.usage.prompt_tokens == 120
    assert restored.pricing == _PRICING


def test_run_id_sorts_by_question_then_time() -> None:
    early = datetime(2026, 7, 25, 14, 25, 30, 42000, tzinfo=UTC)
    late = datetime(2026, 7, 26, 9, 12, 4, tzinfo=UTC)

    early_id = build_run_id("001a", early)

    assert early_id == "001a__20260725T142530042Z"
    assert early_id < build_run_id("001a", late)
    assert not set(early_id) & set(":/\\ ")


def test_run_ids_a_millisecond_apart_do_not_collide() -> None:
    """Runs of one question follow each other closely and must not share an id."""
    first = datetime(2026, 7, 25, 14, 25, 30, 1000, tzinfo=UTC)
    second = datetime(2026, 7, 25, 14, 25, 30, 2000, tzinfo=UTC)

    assert build_run_id("001a", first) != build_run_id("001a", second)


def test_manifest_is_written_under_its_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path)
    manifest = _manifest()

    path = write_manifest(manifest, "e001_fruit")

    assert path == tmp_path / "e001_fruit" / "manifests" / "run-001.json"
    assert Manifest.model_validate_json(path.read_text("utf-8")) == manifest


def test_collect_package_versions_reports_installed_packages() -> None:
    versions = collect_package_versions()
    assert versions["openai"]
    assert versions["pydantic"]
