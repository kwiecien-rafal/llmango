"""Pricing reference for computing the cost of one generation.

Prices live in a committed pricing.json, updated manually just before a run.
Cost is computed post-hoc from token usage times the pinned price, so a raw
dataset stays cost-attributable and reproducible even after prices change.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from llmango.backends.base import Usage
from llmango.config import PRICING_FILE

_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


class PricingEntry(BaseModel):
    """Per-million-token prices for one model and the date they were recorded."""

    input: float
    output: float
    cached_input: float | None = None
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


def compute_cost(entry: PricingEntry, usage: Usage) -> Cost:
    """Compute the cost of one generation from its token usage and a pricing entry.

    Cached prompt tokens are billed at the discounted cached rate; the rest of the
    prompt tokens at the input rate. Reasoning tokens are already counted inside
    completion_tokens, so they are not billed a second time.
    """
    cached_rate = entry.cached_input if entry.cached_input is not None else entry.input
    non_cached = usage.prompt_tokens - usage.cached_tokens
    input_cost = (
        non_cached / 1_000_000 * entry.input
        + usage.cached_tokens / 1_000_000 * cached_rate
    )
    output_cost = usage.completion_tokens / 1_000_000 * entry.output
    return Cost(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )


def pricing_version(entry: PricingEntry) -> str:
    """The version string tying a computed cost back to a frozen price."""
    return entry.last_updated
