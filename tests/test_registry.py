"""Tests for the experiment registry."""

import pytest

from llmango.registry import (
    ExperimentSpec,
    UnknownExperimentError,
    get_experiment,
    register_experiment,
    resolve_experiment,
    resolve_question_id,
    resolve_schema,
)
from llmango.schemas import LLMResponse


class ThrowawayResponse(LLMResponse):
    value: str


def test_register_get_and_resolve() -> None:
    spec = ExperimentSpec(question_id="throwaway", response_schema=ThrowawayResponse)
    register_experiment(spec)
    assert get_experiment("throwaway") is spec
    assert resolve_schema("throwaway") is ThrowawayResponse


def test_register_rejects_duplicate() -> None:
    spec = ExperimentSpec(question_id="dupe", response_schema=ThrowawayResponse)
    register_experiment(spec)
    with pytest.raises(ValueError):
        register_experiment(spec)


def test_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        get_experiment("does_not_exist")


def test_resolve_experiment_accepts_number_and_full_id() -> None:
    for ref in ("001", "1", "001_favorite_fruit"):
        assert resolve_question_id(ref) == "001_favorite_fruit"


def test_resolve_experiment_unknown_ref_raises() -> None:
    with pytest.raises(KeyError):
        resolve_experiment("does_not_exist")


def test_resolve_experiment_handles_non_decimal_digits() -> None:
    with pytest.raises(UnknownExperimentError):
        resolve_experiment("²")


def test_unknown_experiment_error_renders_plainly() -> None:
    assert str(UnknownExperimentError("plain message")) == "plain message"
