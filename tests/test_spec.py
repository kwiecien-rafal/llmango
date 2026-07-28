"""Tests for the experiment spec: its registered schemas and its optional hooks."""

import pytest

from llmango.schemas import LLMResponse
from llmango.spec import ExperimentSpec, answer_field


class ThrowawayResponse(LLMResponse):
    value: str


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        folder="900_throwaway",
        questions=("900a",),
        schemas=(ThrowawayResponse,),
    )


def test_a_schema_is_looked_up_by_the_name_a_question_declares() -> None:
    assert _spec().schema_named("ThrowawayResponse") is ThrowawayResponse


def test_an_unregistered_schema_name_lists_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="Known schemas: ThrowawayResponse"):
        _spec().schema_named("Missing")


def test_the_answer_is_the_one_field_an_answer_schema_declares() -> None:
    assert answer_field(ThrowawayResponse) == "value"


def test_a_schema_with_more_than_one_field_is_refused_on_declaration() -> None:
    """A registered schema is checked at import, before any run is planned."""

    class TwoFields(LLMResponse):
        value: str
        reason: str

    with pytest.raises(ValueError, match="exactly one"):
        ExperimentSpec(
            folder="900_throwaway", questions=("900a",), schemas=(TwoFields,)
        )


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
