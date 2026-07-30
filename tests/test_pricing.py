"""Tests for the pricing reference: loading, costing and the paid-call guard."""

import json
from pathlib import Path

import pytest

from llmango import pricing as pricing_module
from llmango.backends.base import Usage
from llmango.pricing import (
    COST_GUARD_CALLS,
    PricingEntry,
    compute_cost,
    guard_cost,
    guard_run,
    load_pricing,
    round_usd,
)


def test_the_guard_refuses_more_paid_calls_than_the_limit() -> None:
    guard_cost(COST_GUARD_CALLS, force=False)

    with pytest.raises(ValueError, match="without --force"):
        guard_cost(COST_GUARD_CALLS + 1, force=False)


def test_force_allows_any_number_of_paid_calls() -> None:
    guard_cost(COST_GUARD_CALLS * 10, force=True)


def test_guard_run_refuses_a_model_with_no_price() -> None:
    """An unpriced run would write rows whose cost could never be reconstructed."""
    with pytest.raises(ValueError, match="No price for model 'gpt-5.6-luna'"):
        guard_run("gpt-5.6-luna", None, 1, force=False)


def test_guard_run_refuses_a_priced_run_over_the_limit() -> None:
    with pytest.raises(ValueError, match="without --force"):
        guard_run("gpt-5.6-luna", _entry(), COST_GUARD_CALLS + 1, force=False)

    guard_run("gpt-5.6-luna", _entry(), COST_GUARD_CALLS + 1, force=True)


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


def test_load_pricing_reads_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(pricing_module, "PRICING_FILE", path)

    assert load_pricing().models["m"].input == 0.1


def test_load_pricing_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pricing_module, "PRICING_FILE", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        load_pricing()


def test_committed_pricing_file_includes_the_generation_model() -> None:
    table = load_pricing()
    assert "gpt-5.6-luna" in table.models
