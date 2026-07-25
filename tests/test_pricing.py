"""Tests for the pricing reference: loading, resolution and cost computation."""

import json
from pathlib import Path

import pytest

from llmango.backends.base import Usage
from llmango.pricing import (
    PricingEntry,
    PricingTable,
    compute_cost,
    load_pricing,
    pricing_version,
    resolve_entry,
)


def _table() -> PricingTable:
    return PricingTable(
        currency="USD",
        unit="per_1m_tokens",
        models={
            "gpt-5.6-luna": PricingEntry(
                input=0.05, cached_input=0.005, output=0.4, last_updated="2026-07-24"
            )
        },
    )


def _usage(cached: int = 4) -> Usage:
    return Usage(
        prompt_tokens=12,
        completion_tokens=3,
        total_tokens=15,
        cached_tokens=cached,
        reasoning_tokens=1,
    )


def test_resolve_entry_matches_the_model_id() -> None:
    entry = resolve_entry(_table(), "gpt-5.6-luna", None)
    assert entry.input == 0.05


def test_resolve_entry_falls_back_to_the_snapshot_base() -> None:
    entry = resolve_entry(_table(), "gpt-5.6-luna-alias", "gpt-5.6-luna-2026-01-01")
    assert entry.output == 0.4


def test_resolve_entry_raises_for_an_unknown_model() -> None:
    with pytest.raises(KeyError):
        resolve_entry(_table(), "unknown-model", None)


def test_compute_cost_applies_the_cached_discount() -> None:
    cost = compute_cost(resolve_entry(_table(), "gpt-5.6-luna", None), _usage())
    assert cost.input_cost_usd == pytest.approx(4.2e-7)
    assert cost.output_cost_usd == pytest.approx(1.2e-6)
    assert cost.total_cost_usd == pytest.approx(1.62e-6)


def test_compute_cost_defaults_cached_rate_to_input() -> None:
    entry = PricingEntry(input=1.0, output=2.0, last_updated="2026-07-24")
    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        total_tokens=1_000_000,
        cached_tokens=500_000,
        reasoning_tokens=0,
    )
    cost = compute_cost(entry, usage)
    assert cost.input_cost_usd == pytest.approx(1.0)


def test_pricing_version_is_the_last_updated_date() -> None:
    entry = resolve_entry(_table(), "gpt-5.6-luna", None)
    assert pricing_version(entry) == "2026-07-24"


def test_load_pricing_reads_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "currency": "USD",
                "unit": "per_1m_tokens",
                "models": {
                    "m": {"input": 0.1, "output": 0.2, "last_updated": "2026-07-24"}
                },
            }
        ),
        encoding="utf-8",
    )
    table = load_pricing(path)
    assert table.models["m"].input == 0.1


def test_load_pricing_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pricing(tmp_path / "nope.json")


def test_committed_pricing_file_includes_the_generation_model() -> None:
    table = load_pricing()
    assert "gpt-5.6-luna" in table.models
