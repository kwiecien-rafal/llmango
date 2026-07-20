"""Tests for the experiment registry."""

import pytest

from llmango.registry import (
    ExperimentSpec,
    get_experiment,
    register_experiment,
    resolve_schema,
)
from llmango.schemas import LLMResponse


class ThrowawayResponse(LLMResponse):
    value: str


def test_register_get_and_resolve() -> None:
    spec = ExperimentSpec(question_id="throwaway", response_model=ThrowawayResponse)
    register_experiment(spec)
    assert get_experiment("throwaway") is spec
    assert resolve_schema("throwaway") is ThrowawayResponse


def test_register_rejects_duplicate() -> None:
    spec = ExperimentSpec(question_id="dupe", response_model=ThrowawayResponse)
    register_experiment(spec)
    with pytest.raises(ValueError):
        register_experiment(spec)


def test_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        get_experiment("does_not_exist")
