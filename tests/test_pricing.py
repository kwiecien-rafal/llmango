"""Tests for the pricing reference: loading and cost computation."""

import json
from pathlib import Path

import pytest

from llmango.backends.base import Usage
from llmango.pricing import (
    PricingEntry,
    compute_cost,
    load_pricing,
    round_usd,
)


def _entry() -> PricingEntry:
    return PricingEntry(
        input=0.05, cached_input=0.005, output=0.4, last_updated="2026-07-24"
    )


def _usage(cached: int = 4) -> Usage:
    return Usage(
        prompt_tokens=12,
        completion_tokens=3,
        total_tokens=15,
        cached_tokens=cached,
        reasoning_tokens=1,
    )


def test_compute_cost_applies_the_cached_discount() -> None:
    cost = compute_cost(_entry(), _usage())
    assert cost.input_cost_usd == pytest.approx(4.2e-7)
    assert cost.output_cost_usd == pytest.approx(1.2e-6)
    assert cost.total_cost_usd == pytest.approx(1.62e-6)


def test_compute_cost_applies_the_batch_discount() -> None:
    entry = _entry()

    sync = compute_cost(entry, _usage())
    batched = compute_cost(entry, _usage(), batched=True)

    assert batched.total_cost_usd == pytest.approx(sync.total_cost_usd / 2)
    assert batched.input_cost_usd == pytest.approx(2.1e-7)


def test_costs_are_rounded_to_significant_digits() -> None:
    entry = PricingEntry(input=0.15, output=0.6, last_updated="2026-07-24")
    usage = Usage(
        prompt_tokens=137,
        completion_tokens=12,
        total_tokens=149,
        cached_tokens=0,
        reasoning_tokens=0,
    )

    cost = compute_cost(entry, usage)

    assert repr(cost.input_cost_usd) == "2.055e-05"
    assert cost.total_cost_usd == cost.input_cost_usd + cost.output_cost_usd


def test_round_usd_snaps_noise_but_keeps_real_precision() -> None:
    assert round_usd(0.00021299999999999998) == 0.000213
    assert round_usd(0.18518505) == 0.18518505


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
