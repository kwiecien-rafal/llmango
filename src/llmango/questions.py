"""Experiment and question config, and the prompt templates they name.

An experiment is declared by experiment.yaml plus normalize.md. It contains
questions (001a, 001b, ...), each a subfolder with its own meta.yaml and one
prompt template per language. A template's placeholders are filled per sample by
the prompt inputs the question declares, so a rendered prompt is produced per
sample rather than read once from a static file. Loading a question checks that
its templates and its declared inputs name each other exactly.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmango.config import experiment_dir, question_dir, sha256_text
from llmango.inputs import InputDeclarations, validate_placeholders
from llmango.registry import get_experiment, resolve_experiment_id, resolve_schema

_EXPERIMENT_FILE = "experiment.yaml"
_META_FILE = "meta.yaml"


class SamplingParams(BaseModel):
    """Sampling parameters passed to a generation backend."""

    temperature: float = 1.0
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None


class ExperimentConfig(BaseModel):
    """Parsed contents of an experiment's experiment.yaml manifest."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    experiment_id: str
    schema_name: str = Field(alias="schema")
    model: str | None = None
    normalize_model: str | None = None
    sampling: SamplingParams = Field(default_factory=SamplingParams)


class QuestionMeta(BaseModel):
    """Parsed contents of a question's meta.yaml manifest.

    Each entry under inputs is keyed by the template placeholder it fills, and
    its body is passed through to the experiment untouched.
    """

    model_config = ConfigDict(extra="forbid")

    languages: list[str]
    schema_variants: list[str] = Field(default_factory=lambda: ["en"])
    inputs: InputDeclarations = Field(default_factory=dict)
    sampling: SamplingParams | None = None


@dataclass(frozen=True)
class QuestionConfig:
    """A question resolved against its experiment, ready to run."""

    question_id: str
    experiment_id: str
    languages: list[str]
    schema_variants: list[str]
    inputs: InputDeclarations
    sampling: SamplingParams
    model: str | None


@dataclass(frozen=True)
class PromptTemplate:
    """A loaded prompt template with its text and content hash."""

    lang: str
    path: Path
    text: str
    sha256: str


def load_experiment_config(experiment_id: str) -> ExperimentConfig:
    """Load and validate an experiment's experiment.yaml manifest."""
    path = experiment_dir(experiment_id) / _EXPERIMENT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(data)
    if config.experiment_id != experiment_id:
        raise ValueError(
            f"experiment.yaml experiment_id '{config.experiment_id}' does not match "
            f"requested '{experiment_id}'"
        )
    registered = resolve_schema(experiment_id).__name__
    if registered != config.schema_name:
        raise ValueError(
            f"experiment.yaml schema '{config.schema_name}' does not match registered "
            f"schema '{registered}'"
        )
    return config


def list_questions(experiment_id: str) -> list[str]:
    """Return the sorted question ids (subfolders with a meta.yaml) of an experiment."""
    directory = experiment_dir(experiment_id)
    if not directory.is_dir():
        return []
    return sorted(
        child.name
        for child in directory.iterdir()
        if child.is_dir() and (child / _META_FILE).is_file()
    )


def load_question(ref: str) -> QuestionConfig:
    """Load and validate a question by reference (e.g. 001a).

    Resolves the owning experiment, reads the question's meta.yaml, and checks
    that every declared language has a template, that each template's
    placeholders match the declared prompt inputs, and that every schema variant
    is registered.
    """
    experiment_id = resolve_experiment_id(ref)
    question_id = ref.strip()
    directory = question_dir(experiment_id, question_id)
    meta_path = directory / _META_FILE
    if not meta_path.is_file():
        available = ", ".join(list_questions(experiment_id)) or "none"
        raise FileNotFoundError(
            f"Unknown question {ref!r} in {experiment_id}. Available: {available}."
        )

    meta = QuestionMeta.model_validate(yaml.safe_load(meta_path.read_text("utf-8")))
    exp_config = load_experiment_config(experiment_id)

    missing = [
        lang for lang in meta.languages if not (directory / f"{lang}.md").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing prompt templates for {question_id}: {', '.join(missing)}"
        )

    for lang in meta.languages:
        template = load_template(experiment_id, question_id, lang)
        validate_placeholders(template.text, meta.inputs, f"{question_id}/{lang}.md")
    _validate_schema_variants(meta.schema_variants, experiment_id)

    return QuestionConfig(
        question_id=question_id,
        experiment_id=experiment_id,
        languages=meta.languages,
        schema_variants=meta.schema_variants,
        inputs=meta.inputs,
        sampling=meta.sampling or exp_config.sampling,
        model=exp_config.model,
    )


def load_template(experiment_id: str, question_id: str, lang: str) -> PromptTemplate:
    """Load one language's prompt template for a question."""
    path = question_dir(experiment_id, question_id) / f"{lang}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    text = path.read_text(encoding="utf-8")
    return PromptTemplate(lang=lang, path=path, text=text, sha256=sha256_text(text))


def _validate_schema_variants(schema_variants: list[str], experiment_id: str) -> None:
    """Check every declared schema variant is registered for the experiment."""
    known = get_experiment(experiment_id).schema_variants
    unknown = [variant for variant in schema_variants if variant not in known]
    if unknown:
        raise ValueError(
            f"{experiment_id} has no schema variant(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}."
        )
