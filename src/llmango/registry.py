"""Experiment registry.

An experiment registers a lightweight ExperimentSpec describing its response
schema variants and a few optional hooks. The runner, storage, normalize and
analyze code stay experiment-agnostic and are driven entirely by the registered
spec. One experiment holds several questions (001a, 001b, ...); questions are
filesystem entities resolved in questions.py, while the spec here carries the
code-level schema and normalization shared across an experiment's questions.
"""

import hashlib
import json
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
class SchemaVariant:
    """One way to request an answer: a response schema and its raw-answer field.

    schema is None for the free-text variant, which sends no structured output
    and reads the answer straight from the plain text the model returns.
    """

    schema: type[BaseModel] | None
    field: str | None

    def extract(self, parsed: BaseModel | None, raw_json: str | None) -> str:
        """Return the raw answer string from a parsed model or free text."""
        if self.schema is not None and self.field is not None and parsed is not None:
            return str(getattr(parsed, self.field))
        return raw_json or ""

    @property
    def schema_name(self) -> str | None:
        """The response schema class name, or None for the free-text variant."""
        return self.schema.__name__ if self.schema is not None else None

    @property
    def schema_sha256(self) -> str | None:
        """Hash the variant's JSON schema, or None for the free-text variant.

        The schema is itself part of the prompt: its field names, their order and
        any enums all reach the model. Hashing it makes an edit to a schema as
        traceable as an edit to a prompt template, and keys are left in
        declaration order rather than sorted, because that order is one of the
        things the model sees.
        """
        if self.schema is None:
            return None
        encoded = json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything the generic pipeline needs to run one experiment."""

    experiment_id: str
    schema_variants: dict[str, SchemaVariant]
    to_row: Callable[[BaseModel | None, str], dict[str, object]] | None = None
    normalization_schema: type[BaseModel] | None = None
    preprocess: Callable[[str], str] | None = None
    raw_column: str = "raw"
    canonical_column: str = "canonical"
    canonical_values: frozenset[str] | None = None
    detect_language_drift: bool = False
    default_variant: str = "en"

    @property
    def response_schema(self) -> type[BaseModel]:
        """The default variant's response schema, used for name validation."""
        variant = self.schema_variants[self.default_variant]
        if variant.schema is None:
            raise ValueError(
                f"Experiment {self.experiment_id} has no default response schema."
            )
        return variant.schema

    def variant(self, schema_variant: str) -> SchemaVariant:
        """Return the registered schema variant a run asked for."""
        try:
            return self.schema_variants[schema_variant]
        except KeyError:
            known = ", ".join(sorted(self.schema_variants))
            raise ValueError(
                f"Experiment {self.experiment_id} has no schema variant "
                f"'{schema_variant}'. Known variants: {known}."
            ) from None


_REGISTRY: dict[str, ExperimentSpec] = {}


def register_experiment(spec: ExperimentSpec) -> ExperimentSpec:
    """Register an experiment spec, keyed by its experiment_id."""
    if spec.experiment_id in _REGISTRY:
        raise ValueError(f"Experiment already registered: {spec.experiment_id}")
    _REGISTRY[spec.experiment_id] = spec
    return spec


def get_experiment(experiment_id: str) -> ExperimentSpec:
    """Return the registered spec for experiment_id, or raise if unknown.

    Ensures every experiment is registered first, so any caller that reaches the
    registry (the runner, normalize, analyze) sees a populated table.
    """
    _ensure_registered()
    try:
        return _REGISTRY[experiment_id]
    except KeyError:
        raise UnknownExperimentError(f"Unknown experiment: {experiment_id}") from None


def resolve_schema(experiment_id: str) -> type[BaseModel]:
    """Return the default response schema class registered for experiment_id."""
    return get_experiment(experiment_id).response_schema


def experiment_number(spec: ExperimentSpec) -> str | None:
    """Return the leading number of an experiment's id, e.g. '001'."""
    match = _NUMBER_PREFIX.match(spec.experiment_id)
    return match.group(1) if match else None


def resolve_experiment(ref: str) -> ExperimentSpec:
    """Resolve an experiment reference to its registered spec.

    Accepts the full id (001_fruit), just its number (001 or 1), or a question
    reference (001a), all of which point at the same owning experiment. This is
    the single front door that lets the CLI and Justfile refer to an experiment
    however is convenient.
    """
    _ensure_registered()
    ref = ref.strip()
    if ref in _REGISTRY:
        return _REGISTRY[ref]
    match = _NUMBER_PREFIX.match(ref)
    number = match.group(1) if match else ref
    if number.isdecimal():
        for spec in _REGISTRY.values():
            spec_number = experiment_number(spec)
            if spec_number and int(number) == int(spec_number):
                return spec
    known = ", ".join(sorted(_REGISTRY)) or "none registered"
    raise UnknownExperimentError(
        f"Unknown experiment: {ref!r}. Known experiments: {known}."
    )


def resolve_experiment_id(ref: str) -> str:
    """Resolve any experiment or question reference to its experiment_id."""
    return resolve_experiment(ref).experiment_id


def _ensure_registered() -> None:
    """Import the experiments package so every spec is registered."""
    from llmango.experiments import ensure_registered

    ensure_registered()
