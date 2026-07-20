"""Experiment registry.

An experiment registers a lightweight ExperimentSpec describing its response
model and a few optional hooks. The runner, storage, normalize and analyze code
stay experiment-agnostic and are driven entirely by the registered spec.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything the generic pipeline needs to run one experiment."""

    question_id: str
    response_model: type[BaseModel]
    to_row: Callable[[BaseModel], dict[str, object]] | None = None
    normalization_model: type[BaseModel] | None = None
    preprocess: Callable[[str], str] | None = None


_REGISTRY: dict[str, ExperimentSpec] = {}


def register_experiment(spec: ExperimentSpec) -> ExperimentSpec:
    """Register an experiment spec, keyed by its question_id."""
    if spec.question_id in _REGISTRY:
        raise ValueError(f"Experiment already registered: {spec.question_id}")
    _REGISTRY[spec.question_id] = spec
    return spec


def get_experiment(question_id: str) -> ExperimentSpec:
    """Return the registered spec for question_id, or raise if unknown."""
    try:
        return _REGISTRY[question_id]
    except KeyError:
        raise KeyError(f"Unknown experiment: {question_id}") from None


def resolve_schema(question_id: str) -> type[BaseModel]:
    """Return the response model class registered for question_id."""
    return get_experiment(question_id).response_model
