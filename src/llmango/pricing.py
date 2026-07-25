"""Pricing reference for computing the cost of one generation.

Prices live in data/pricing.json, updated manually."""

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from llmango.backends.base import Usage
from llmango.config import PRICING_FILE

_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")

_SIGNIFICANT_DIGITS = 10


class PricingEntry(BaseModel):
    """Per-million-token prices for one model and the date they were recorded."""

    input: float
    output: float
    cached_input: float | None = None
    batch_discount: float = 0.5
    last_updated: str
    source: str | None = None


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


def load_pricing(path: Path = PRICING_FILE) -> PricingTable:
    """Load and validate the pricing reference from pricing.json."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No pricing file at {path}. Create data/pricing.json with the models "
            f"you plan to run, prices per 1M tokens, before generating."
        )
    return PricingTable.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_entry(
    table: PricingTable, model: str, model_snapshot: str | None
) -> PricingEntry:
    """Find the pricing entry for a model, trying its id then its snapshot base."""
    if model in table.models:
        return table.models[model]
    if model_snapshot is not None:
        if model_snapshot in table.models:
            return table.models[model_snapshot]
        base = _DATE_SUFFIX.sub("", model_snapshot)
        if base in table.models:
            return table.models[base]
    raise KeyError(
        f"No pricing for model '{model}' in the pricing file. Add it to "
        f"data/pricing.json, prices per 1M tokens, before generating."
    )


def round_usd(value: float) -> float:
    """Round a cost to ten significant digits."""
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def compute_cost(entry: PricingEntry, usage: Usage, *, batched: bool = False) -> Cost:
    """Compute the cost of one generation from its token usage and a pricing entry."""
    cached_rate = entry.cached_input if entry.cached_input is not None else entry.input
    non_cached = usage.prompt_tokens - usage.cached_tokens
    discount = entry.batch_discount if batched else 1.0
    input_cost = round_usd(
        (
            non_cached / 1_000_000 * entry.input
            + usage.cached_tokens / 1_000_000 * cached_rate
        )
        * discount
    )
    output_cost = round_usd(
        usage.completion_tokens / 1_000_000 * entry.output * discount
    )
    return Cost(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=round_usd(input_cost + output_cost),
    )


def pricing_version(entry: PricingEntry) -> str:
    """The version string tying a computed cost back to a frozen price."""
    return entry.last_updated
