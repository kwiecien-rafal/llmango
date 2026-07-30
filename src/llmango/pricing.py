"""Pricing reference for costing one generation, and the guard on how many to make."""

from dataclasses import dataclass

from pydantic import BaseModel

from llmango.backends.base import Usage
from llmango.config import PRICING_FILE

COST_GUARD_CALLS = 100

_SIGNIFICANT_DIGITS = 10


class PricingEntry(BaseModel):
    """Per-million-token prices for one model and the date they were recorded."""

    input: float
    output: float
    cached_input: float | None = None
    last_updated: str


class PricingTable(BaseModel):
    """The full pricing reference: a currency, a unit, and per-model entries."""

    currency: str
    unit: str
    models: dict[str, PricingEntry]


@dataclass(frozen=True)
class Cost:
    """The computed cost of one generation, split by input and output."""

    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def guard_cost(calls: int, force: bool) -> None:
    """Refuse to make more paid calls than the unforced limit allows."""
    if calls > COST_GUARD_CALLS and not force:
        raise ValueError(
            f"Refusing {calls} paid calls without --force; the unforced limit is "
            f"{COST_GUARD_CALLS}."
        )


def guard_run(
    model: str, price: PricingEntry | None, calls: int, force: bool
) -> PricingEntry:
    """Refuse a run that is unpriced or too large, and return the price it will use."""
    if price is None:
        raise ValueError(
            f"No price for model '{model}'. Add it to data/pricing.json, prices "
            f"per 1M tokens, before generating."
        )
    guard_cost(calls, force)
    return price


def load_pricing() -> PricingTable:
    """Load and validate the pricing reference from data/pricing.json."""
    if not PRICING_FILE.is_file():
        raise FileNotFoundError(
            f"No pricing file at {PRICING_FILE}. Create data/pricing.json with the "
            f"models you plan to run, prices per 1M tokens, before generating."
        )
    return PricingTable.model_validate_json(PRICING_FILE.read_text(encoding="utf-8"))


def round_usd(value: float) -> float:
    """Round a cost to ten significant digits."""
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def compute_cost(entry: PricingEntry, usage: Usage) -> Cost:
    """Compute the cost of one generation from its token usage and a pricing entry."""
    cached_rate = entry.cached_input if entry.cached_input is not None else entry.input
    non_cached = usage.prompt_tokens - usage.cached_tokens
    input_cost = round_usd(
        non_cached / 1_000_000 * entry.input
        + usage.cached_tokens / 1_000_000 * cached_rate
    )
    output_cost = round_usd(usage.completion_tokens / 1_000_000 * entry.output)
    return Cost(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=round_usd(input_cost + output_cost),
    )
