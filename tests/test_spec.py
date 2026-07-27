"""Tests for the experiment spec: schema variants and the optional hooks."""

import pytest

from llmango.schemas import LLMResponse
from llmango.spec import ExperimentSpec, SchemaVariant


class ThrowawayResponse(LLMResponse):
    value: str


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        folder="900_throwaway",
        questions=("900a",),
        schema_variants={"en": SchemaVariant(ThrowawayResponse, "value")},
    )


def test_variant_lookup_raises_on_unknown_schema_variant() -> None:
    with pytest.raises(ValueError, match="schema variant"):
        _spec().variant("pl")


def test_schema_identity_names_and_hashes_the_response_schema() -> None:
    variant = SchemaVariant(ThrowawayResponse, "value")

    assert variant.schema_name == "ThrowawayResponse"
    assert len(variant.schema_sha256 or "") == 64


def test_schema_hash_changes_when_a_field_changes() -> None:
    class Renamed(LLMResponse):
        other: str

    assert (
        SchemaVariant(ThrowawayResponse, "value").schema_sha256
        != SchemaVariant(Renamed, "other").schema_sha256
    )


def test_free_text_variant_has_no_schema_identity() -> None:
    variant = SchemaVariant(None, None)

    assert variant.schema_name is None
    assert variant.schema_sha256 is None


def test_column_hooks_are_optional() -> None:
    """An experiment that appends no columns registers none of the extra hooks."""
    spec = _spec()

    assert spec.extra_raw_columns is None
    assert spec.extra_raw_dtypes == {}
    assert spec.extra_normalized_columns is None


def test_prompt_input_hooks_are_optional() -> None:
    """An experiment with no prompt inputs registers none of their hooks."""
    spec = _spec()

    assert spec.build_input is None
    assert spec.mapping_seed is None
