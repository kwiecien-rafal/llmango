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
    resolve_schema,
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
    assert resolve_schema("throwaway") is ThrowawayResponse


def test_register_rejects_duplicate() -> None:
    spec = _spec("dupe")
    register_experiment(spec)
    with pytest.raises(ValueError):
        register_experiment(spec)


def test_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        get_experiment("does_not_exist")


def test_variant_lookup_raises_on_unknown_schema_lang() -> None:
    spec = _spec("variants")
    with pytest.raises(ValueError, match="schema variant"):
        spec.variant("pl")


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
