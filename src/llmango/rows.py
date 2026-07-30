"""The raw result table: every column one run writes, its dtype and its value."""

import json
from collections import defaultdict
from dataclasses import dataclass

import polars as pl
from pydantic import BaseModel

from llmango.backends.base import GenResult, Usage
from llmango.config import sha256_text
from llmango.manifest import Manifest, UsageTotals
from llmango.pricing import Cost, PricingEntry, compute_cost, round_usd
from llmango.questions import Arm
from llmango.spec import (
    ArmKey,
    ExperimentSpec,
    ExtraRawDtypes,
    answer_field,
    schema_name,
)

LEADING_COLUMNS: dict[str, pl.DataType] = {
    "question_id": pl.String(),
    "lang": pl.String(),
    "model": pl.String(),
    "provider": pl.String(),
    "run_id": pl.String(),
    "sample_idx": pl.Int64(),
    "temperature": pl.Float64(),
    "prompt_sha256": pl.String(),
    "prompt": pl.String(),
    "prompt_inputs": pl.String(),
    "raw_json": pl.String(),
    "answer": pl.String(),
}

TRAILING_COLUMNS: dict[str, pl.DataType] = {
    "model_snapshot": pl.String(),
    "finish_reason": pl.String(),
    "refusal": pl.String(),
    "error": pl.String(),
    "response_id": pl.String(),
    "service_tier": pl.String(),
    "provider_created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "response_schema": pl.String(),
    "request_envelope": pl.String(),
    "response_envelope": pl.String(),
    "prompt_tokens": pl.Int64(),
    "completion_tokens": pl.Int64(),
    "total_tokens": pl.Int64(),
    "cached_tokens": pl.Int64(),
    "reasoning_tokens": pl.Int64(),
    "input_cost_usd": pl.Float64(),
    "output_cost_usd": pl.Float64(),
    "total_cost_usd": pl.Float64(),
    "pricing_version": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}


@dataclass(frozen=True)
class Sample:
    """One arm's sample: the prompt it asks and the provenance its row records."""

    arm: Arm
    sample_idx: int
    prompt_inputs: str
    prompt: str


@dataclass(frozen=True)
class Generation:
    """One sample, the result it came back as, and what that cost."""

    sample: Sample
    result: GenResult
    cost: Cost | None


def dtypes(extra: ExtraRawDtypes) -> dict[str, pl.DataType]:
    """The dtype of every column a row carries, the experiment's extras included."""
    return {**LEADING_COLUMNS, **TRAILING_COLUMNS, **extra}


def costed(
    samples: list[Sample], results: list[GenResult], pricing: PricingEntry | None
) -> list[Generation]:
    """Pair each sample with the result it came back as, costing every one once."""
    return [
        Generation(sample, result, _cost(result.usage, pricing))
        for sample, result in zip(samples, results, strict=True)
    ]


def build(
    generations: list[Generation], manifest: Manifest, spec: ExperimentSpec
) -> list[dict[str, object]]:
    """Turn every generation into the row the raw parquet stores it as."""
    response_schemas = {
        record.schema_name: _schema_column(record.response_schema)
        for record in manifest.arms
    }
    return [
        _row(generation, manifest, spec, response_schemas) for generation in generations
    ]


def usage_totals(generations: list[Generation]) -> UsageTotals:
    """Sum generations and their costs into their token and cost totals."""
    results = [generation.result for generation in generations]
    usages = [result.usage for result in results if result.usage is not None]
    priced = [
        generation.cost for generation in generations if generation.cost is not None
    ]
    return UsageTotals(
        errors=sum(result.error is not None for result in results),
        provider_refusals=sum(result.refusal is not None for result in results),
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        cached_tokens=sum(usage.cached_tokens for usage in usages),
        reasoning_tokens=sum(usage.reasoning_tokens for usage in usages),
        input_cost_usd=round_usd(sum(cost.input_cost_usd for cost in priced)),
        output_cost_usd=round_usd(sum(cost.output_cost_usd for cost in priced)),
        total_cost_usd=round_usd(sum(cost.total_cost_usd for cost in priced)),
    )


def usage_by_arm(generations: list[Generation]) -> dict[ArmKey, UsageTotals]:
    """Total what each arm of a run used, in one pass over its generations."""
    grouped: dict[ArmKey, list[Generation]] = defaultdict(list)
    for generation in generations:
        grouped[generation.sample.arm.key].append(generation)
    return {key: usage_totals(group) for key, group in grouped.items()}


def _row(
    generation: Generation,
    manifest: Manifest,
    spec: ExperimentSpec,
    response_schemas: dict[str | None, str | None],
) -> dict[str, object]:
    """Combine the common columns, the experiment's extras, provenance and cost."""
    sample, result = generation.sample, generation.result
    answer = _answer(result.parsed, result.raw_json)
    extra = (
        spec.extra_raw_columns(result.parsed, answer) if spec.extra_raw_columns else {}
    )
    return {
        "question_id": manifest.question_id,
        "lang": sample.arm.lang,
        "model": manifest.model,
        "provider": manifest.provider,
        "run_id": manifest.run_id,
        "sample_idx": sample.sample_idx,
        "temperature": manifest.temperature,
        "prompt_sha256": sha256_text(sample.prompt),
        "prompt": sample.prompt,
        "prompt_inputs": sample.prompt_inputs,
        "raw_json": result.raw_json,
        "answer": answer,
        **extra,
        "model_snapshot": result.model_snapshot,
        "finish_reason": result.finish_reason,
        "refusal": result.refusal,
        "error": result.error,
        "response_id": result.response_id,
        "service_tier": result.service_tier,
        "provider_created_at": result.provider_created_at,
        "response_schema": response_schemas[schema_name(sample.arm.schema)],
        "request_envelope": result.request_envelope,
        "response_envelope": result.response_envelope,
        **_usage_columns(result.usage),
        **_cost_columns(generation.cost, manifest.pricing),
        "created_at": result.created_at,
    }


def _answer(parsed: BaseModel | None, raw_json: str | None) -> str:
    """Read the answer off a parsed response, or off free text when there is none."""
    if parsed is None:
        return raw_json or ""
    return str(getattr(parsed, answer_field(type(parsed))))


def _schema_column(response_schema: dict[str, object] | None) -> str | None:
    """Serialize an arm's response schema once, for every row that arm wrote."""
    if response_schema is None:
        return None
    return json.dumps(response_schema, ensure_ascii=False)


def _cost(usage: Usage | None, pricing: PricingEntry | None) -> Cost | None:
    """Cost one generation, None when its usage or its model's price is missing."""
    if usage is None or pricing is None:
        return None
    return compute_cost(pricing, usage)


def _usage_columns(usage: Usage | None) -> dict[str, object]:
    """Map token usage to its columns, null in each when the provider reported none."""
    return {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "cached_tokens": usage.cached_tokens if usage else None,
        "reasoning_tokens": usage.reasoning_tokens if usage else None,
    }


def _cost_columns(cost: Cost | None, pricing: PricingEntry | None) -> dict[str, object]:
    """Map one generation's cost to its columns, null in each when it has no cost."""
    return {
        "input_cost_usd": cost.input_cost_usd if cost else None,
        "output_cost_usd": cost.output_cost_usd if cost else None,
        "total_cost_usd": cost.total_cost_usd if cost else None,
        "pricing_version": pricing.last_updated if cost and pricing else None,
    }
