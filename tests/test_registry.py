"""Tests for the experiment registry."""

import pytest

from llmango.registry import (
    ExperimentSpec,
    SchemaVariant,
    UnknownExperimentError,
    get_experiment,
    register_experiment,
    resolve_experiment,
    resolve_experiment_id,
)
from llmango.schemas import LLMResponse


class ThrowawayResponse(LLMResponse):
    value: str


def _spec(experiment_id: str) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        schema_variants={"en": SchemaVariant(ThrowawayResponse, "value")},
    )


def test_register_get_and_resolve() -> None:
    spec = _spec("throwaway")
    register_experiment(spec)
    assert get_experiment("throwaway") is spec


def test_register_rejects_duplicate() -> None:
    spec = _spec("dupe")
    register_experiment(spec)
    with pytest.raises(ValueError):
        register_experiment(spec)


def test_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        get_experiment("does_not_exist")


def test_variant_lookup_raises_on_unknown_schema_variant() -> None:
    spec = _spec("variants")
    with pytest.raises(ValueError, match="schema variant"):
        spec.variant("pl")


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


def test_resolve_experiment_accepts_number_id_and_question() -> None:
    for ref in ("001", "1", "001_fruit", "001a"):
        assert resolve_experiment_id(ref) == "001_fruit"


def test_resolve_experiment_unknown_ref_raises() -> None:
    with pytest.raises(KeyError):
        resolve_experiment("does_not_exist")


def test_resolve_experiment_handles_non_decimal_digits() -> None:
    with pytest.raises(UnknownExperimentError):
        resolve_experiment("²")


def test_unknown_experiment_error_renders_plainly() -> None:
    assert str(UnknownExperimentError("plain message")) == "plain message"


def test_column_hooks_are_optional() -> None:
    """An experiment that appends no columns registers none of the extra hooks."""
    spec = _spec("plain_columns")

    assert spec.extra_raw_columns is None
    assert spec.extra_raw_dtypes == {}
    assert spec.extra_normalized_columns is None


def test_prompt_input_hooks_are_optional() -> None:
    """An experiment with no prompt inputs registers none of their hooks."""
    spec = _spec("plain_inputs")

    assert spec.build_input is None
    assert spec.mapping_seed is None


def test_an_experiment_supplies_its_own_prompt_input_hooks() -> None:
    spec = get_experiment("001_fruit")

    assert spec.build_input is not None
    assert spec.mapping_seed is not None
