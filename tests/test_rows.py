"""Tests for the raw table: the columns a row carries, and per-arm usage totals."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import polars as pl

from llmango.backends.base import GenRequest, GenResult, Usage
from llmango.experiments.e001_fruit.experiment import FRUIT, FruitChoice
from llmango.manifest import ArmRecord, Manifest, UsageTotals
from llmango.pricing import PricingEntry
from llmango.questions import Arm
from llmango.rows import (
    LEADING_COLUMNS,
    TRAILING_COLUMNS,
    Sample,
    build_row,
    column_dtypes,
    cost_sample,
    schema_columns,
    usage_by_arm,
    usage_totals,
)
from llmango.spec import ExperimentSpec

_PRICING = PricingEntry(
    input=0.05, cached_input=0.005, output=0.4, last_updated="2026-07-24"
)


def _manifest(*arms: ArmRecord) -> Manifest:
    return Manifest(
        run_id="001a__20260720T101500000Z",
        question_id="001a",
        provider="openai",
        model="gpt-5.6-luna",
        temperature=1.0,
        samples_total=len(arms),
        samples_per_arm=1,
        arms=list(arms),
        pricing=_PRICING,
        usage=UsageTotals(),
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
    )


def _arm_record(lang: str, schema_name: str | None = "FruitChoice") -> ArmRecord:
    return ArmRecord(
        lang=lang,
        schema_name=schema_name,
        response_schema={"title": schema_name} if schema_name else None,
        template_sha256=f"sha-{lang}",
        usage=UsageTotals(),
    )


def _sample(lang: str, sample_idx: int = 0) -> Sample:
    return Sample(
        arm=Arm(schema=FruitChoice, lang=lang),
        sample_idx=sample_idx,
        prompt_inputs='{"fruit_list": ["apple", "mango"]}',
        prompt=f"Pick one random fruit ({lang})",
    )


def _result(sample: Sample, *, usage: Usage | None = None, fruit: str = "apple"):
    parsed = FruitChoice(fruit=fruit)
    return GenResult(
        request=GenRequest(
            model="gpt-5.6-luna",
            prompt=sample.prompt,
            response_schema=FruitChoice,
        ),
        raw_json=parsed.model_dump_json(),
        parsed=parsed,
        model_snapshot="gpt-5.6-luna-2026-01-01",
        finish_reason="stop",
        refusal=None,
        error=None,
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
        generation_seconds=0.5,
        usage=usage,
    )


def _usage(prompt_tokens: int = 12) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=3,
        total_tokens=prompt_tokens + 3,
        cached_tokens=4,
        reasoning_tokens=1,
    )


def _built_row(
    sample: Sample,
    manifest: Manifest,
    spec: ExperimentSpec = FRUIT,
    *,
    usage: Usage | None = None,
) -> dict[str, object]:
    """Cost one sample and build its row, the way the runner does per call."""
    costed_sample = cost_sample(sample, _result(sample, usage=usage), _PRICING)

    return build_row(costed_sample, manifest, spec, schema_columns(manifest))


def test_a_row_carries_every_declared_column_in_order() -> None:
    """Column order is the table's contract, so a row emits it rather than storage."""
    row = _built_row(_sample("en"), _manifest(_arm_record("en")), usage=_usage())

    assert list(row) == [*LEADING_COLUMNS, *TRAILING_COLUMNS]
    assert row["question_id"] == "001a"
    assert row["lang"] == "en"
    assert row["answer"] == "apple"
    assert row["generation_seconds"] == 0.5
    assert row["prompt_sha256"] != row["prompt"]
    assert json.loads(str(row["response_schema"])) == {"title": "FruitChoice"}


def test_a_row_without_usage_carries_null_tokens_and_costs() -> None:
    """Every column is present in every row, so the raw schema cannot vary."""
    row = _built_row(_sample("en"), _manifest(_arm_record("en")))

    assert list(row) == [*LEADING_COLUMNS, *TRAILING_COLUMNS]
    assert row["prompt_tokens"] is None
    assert row["total_cost_usd"] is None
    assert row["pricing_version"] is None


def test_a_priced_row_carries_its_cost_and_the_price_it_used() -> None:
    row = _built_row(_sample("en"), _manifest(_arm_record("en")), usage=_usage())

    assert isinstance(row["total_cost_usd"], float)
    assert row["total_cost_usd"] > 0
    assert row["pricing_version"] == "2026-07-24"


def test_an_experiment_column_sits_between_the_answer_and_the_provenance() -> None:
    """An experiment appends columns in one slot, right after the shared answer."""
    spec = replace(FRUIT, extra_raw_columns=lambda parsed, answer: {"ripeness": "ripe"})

    columns = list(
        _built_row(_sample("en"), _manifest(_arm_record("en")), spec, usage=_usage())
    )

    start = columns.index("answer")
    assert columns[start : start + 3] == ["answer", "ripeness", "model_snapshot"]


def test_the_free_text_arm_stores_no_response_schema() -> None:
    sample = replace(_sample("pl"), arm=Arm(schema=None, lang="pl"))

    row = _built_row(sample, _manifest(_arm_record("pl", None)), usage=_usage())

    assert row["response_schema"] is None


def test_usage_is_totalled_per_arm_as_well_as_for_the_run() -> None:
    samples = [_sample("en"), _sample("en", 1), _sample("pl")]
    costed_samples = [
        cost_sample(sample, _result(sample, usage=_usage()), _PRICING)
        for sample in samples
    ]

    by_arm = usage_by_arm(costed_samples)

    assert set(by_arm) == {("en", "FruitChoice"), ("pl", "FruitChoice")}
    assert by_arm[("en", "FruitChoice")].prompt_tokens == 24
    assert by_arm[("pl", "FruitChoice")].prompt_tokens == 12
    assert usage_totals(costed_samples).prompt_tokens == 36


def test_usage_totals_count_outcomes_that_carried_no_tokens() -> None:
    sample = _sample("en")
    refused = replace(
        _result(sample), parsed=None, raw_json=None, refusal="I can't help with that."
    )
    errored = replace(_result(sample), parsed=None, raw_json=None, error="timeout")

    totals = usage_totals(
        [cost_sample(sample, result, _PRICING) for result in (refused, errored)]
    )

    assert totals.provider_refusals == 1
    assert totals.errors == 1
    assert totals.total_tokens == 0
    assert totals.total_cost_usd == 0.0


def test_column_dtypes_cover_every_column_and_the_experiments_extras() -> None:
    declared = column_dtypes({"ripeness": pl.String()})

    assert set(declared) == {*LEADING_COLUMNS, *TRAILING_COLUMNS, "ripeness"}
    assert declared["sample_idx"] == pl.Int64()
