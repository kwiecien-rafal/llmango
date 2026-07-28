"""Experiment and question config, and the prompt templates they name."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmango.config import experiment_dir, question_dir, sha256_text
from llmango.experiments import spec_for
from llmango.inputs import InputDeclarations, validate_placeholders
from llmango.spec import ExperimentSpec

_EXPERIMENT_FILE = "experiment.yaml"
_QUESTION_FILE = "question.yaml"


class ExperimentConfig(BaseModel):
    """Parsed contents of an experiment's experiment.yaml manifest."""

    model_config = ConfigDict(extra="forbid")

    normalize_provider: str = "openai"
    normalize_model: str


class LanguageAsk(BaseModel):
    """One language the question is asked in, and the schemas it is asked under."""

    model_config = ConfigDict(extra="forbid")

    language: str
    schemas: list[str | None] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_entry_per_schema(self) -> Self:
        """Reject a schema listed twice, which would ask one arm twice over."""
        names = self.schemas
        repeated = sorted(
            {name or "no schema" for name in names if names.count(name) > 1}
        )
        if repeated:
            raise ValueError(
                f"Language {self.language} asks under {', '.join(repeated)} more "
                f"than once; each schema it is asked under is listed once."
            )
        return self


class QuestionConfig(BaseModel):
    """Parsed contents of a question's question.yaml manifest."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "openai"
    model: str
    temperature: float = 1.0
    ask: list[LanguageAsk] = Field(min_length=1)
    inputs: InputDeclarations = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_entry_per_language(self) -> Self:
        """Reject a language declared twice, since one entry holds all its schemas."""
        languages = [entry.language for entry in self.ask]
        repeated = sorted({lang for lang in languages if languages.count(lang) > 1})
        if repeated:
            raise ValueError(
                f"Languages declared more than once: {', '.join(repeated)}. A "
                f"language gets one entry, listing every schema it is asked under."
            )
        return self


@dataclass(frozen=True)
class PromptTemplate:
    """A loaded prompt template with its text and content hash."""

    lang: str
    path: Path
    text: str
    sha256: str


@dataclass(frozen=True)
class Arm:
    """One language asked under one response schema, None for the free-text arm."""

    schema: type[BaseModel] | None
    lang: str


@dataclass(frozen=True)
class Question:
    """A question resolved against its experiment, ready to run."""

    question_id: str
    spec: ExperimentSpec
    provider: str
    model: str
    temperature: float
    arms: list[Arm]
    inputs: InputDeclarations
    templates: dict[str, PromptTemplate]

    @property
    def languages(self) -> list[str]:
        """Every language the question is asked in, in the order it declares them."""
        return list(dict.fromkeys(arm.lang for arm in self.arms))


def load_experiment_config(folder: str) -> ExperimentConfig:
    """Load and validate an experiment's experiment.yaml manifest."""
    path = experiment_dir(folder) / _EXPERIMENT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)


def load_question(question_id: str) -> Question:
    """Load and validate a question by its id (e.g. 001a)."""
    spec = spec_for(question_id)
    directory = question_dir(spec.folder, question_id)
    path = directory / _QUESTION_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing question manifest: {path}")

    config = QuestionConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))
    languages = [entry.language for entry in config.ask]
    templates = _load_templates(spec.folder, question_id, languages)

    for lang, template in templates.items():
        validate_placeholders(template.text, config.inputs, f"{question_id}/{lang}.md")

    return Question(
        question_id=question_id,
        spec=spec,
        provider=config.provider,
        model=config.model,
        temperature=config.temperature,
        arms=_resolve_arms(config.ask, spec),
        inputs=config.inputs,
        templates=templates,
    )


def load_template(folder: str, question_id: str, lang: str) -> PromptTemplate:
    """Load one language's prompt template for a question."""
    path = question_dir(folder, question_id) / f"{lang}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    text = path.read_text(encoding="utf-8")
    return PromptTemplate(lang=lang, path=path, text=text, sha256=sha256_text(text))


def _load_templates(
    folder: str, question_id: str, languages: list[str]
) -> dict[str, PromptTemplate]:
    """Load one template per declared language, naming every missing one at once."""
    directory = question_dir(folder, question_id)
    missing = [lang for lang in languages if not (directory / f"{lang}.md").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing prompt templates for {question_id}: {', '.join(missing)}"
        )
    return {lang: load_template(folder, question_id, lang) for lang in languages}


def _resolve_arms(ask: list[LanguageAsk], spec: ExperimentSpec) -> list[Arm]:
    """Pair every declared language with each schema it is asked under."""
    return [
        Arm(
            schema=spec.schema_named(name) if name is not None else None,
            lang=entry.language,
        )
        for entry in ask
        for name in entry.schemas
    ]
