"""Experiment registry.

An experiment registers a lightweight ExperimentSpec describing its response
model and a few optional hooks. The runner, storage, normalize and analyze code
stay experiment-agnostic and are driven entirely by the registered spec.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

_NUMBER_PREFIX = re.compile(r"^(\d+)")


class UnknownExperimentError(KeyError):
    """Raised when an experiment reference cannot be resolved.

    Subclasses KeyError so existing handlers still catch it, but renders its
    message plainly rather than in KeyError's repr-quoted form.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else super().__str__()


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything the generic pipeline needs to run one experiment."""

    question_id: str
    response_schema: type[BaseModel]
    to_row: Callable[[BaseModel | None], dict[str, object]] | None = None
    normalization_schema: type[BaseModel] | None = None
    preprocess: Callable[[str], str] | None = None
    raw_column: str = "raw"
    canonical_column: str = "canonical"
    canonical_values: frozenset[str] | None = None


_REGISTRY: dict[str, ExperimentSpec] = {}


def register_experiment(spec: ExperimentSpec) -> ExperimentSpec:
    """Register an experiment spec, keyed by its question_id."""
    if spec.question_id in _REGISTRY:
        raise ValueError(f"Experiment already registered: {spec.question_id}")
    _REGISTRY[spec.question_id] = spec
    return spec


def get_experiment(question_id: str) -> ExperimentSpec:
    """Return the registered spec for question_id, or raise if unknown.

    Ensures every experiment is registered first, so any caller that reaches the
    registry (the runner, normalize, analyze) sees a populated table.
    """
    _ensure_registered()
    try:
        return _REGISTRY[question_id]
    except KeyError:
        raise UnknownExperimentError(f"Unknown experiment: {question_id}") from None


def resolve_schema(question_id: str) -> type[BaseModel]:
    """Return the response schema class registered for question_id."""
    return get_experiment(question_id).response_schema


def experiment_number(spec: ExperimentSpec) -> str | None:
    """Return the leading number of an experiment's id, e.g. '001'."""
    match = _NUMBER_PREFIX.match(spec.question_id)
    return match.group(1) if match else None


def resolve_experiment(ref: str) -> ExperimentSpec:
    """Resolve an experiment reference to its registered spec.

    Accepts the full id (001_favorite_fruit) or just its number (001 or 1).
    This is the single front door that lets the CLI and Justfile refer to an
    experiment however is convenient.
    """
    _ensure_registered()
    needle = ref.strip()
    for spec in _REGISTRY.values():
        if needle == spec.question_id:
            return spec
        number = experiment_number(spec)
        if number and needle.isdecimal() and int(needle) == int(number):
            return spec
    known = ", ".join(sorted(_REGISTRY)) or "none registered"
    raise UnknownExperimentError(
        f"Unknown experiment: {ref!r}. Known experiments: {known}."
    )


def resolve_question_id(ref: str) -> str:
    """Resolve any experiment reference to its canonical question_id."""
    return resolve_experiment(ref).question_id


def _ensure_registered() -> None:
    """Import the experiments package so every spec is registered."""
    from llmango.experiments import ensure_registered

    ensure_registered()
