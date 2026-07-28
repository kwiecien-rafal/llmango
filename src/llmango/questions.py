"""Experiment and question config, and the prompt templates they name.

An experiment's shared files are experiment.yaml plus normalize.md, in the folder
its spec names. Each question it owns is a subfolder with its own question.yaml
and one prompt template per language.

A question declares who answers it and how it is asked: one provider, one model
and one temperature, then every language it is asked in with the response schemas
it is asked under. There is one entry per language and no second list to keep in
step with it, so the three shapes a question can take all read the same way: one
schema for every language, a schema of its own per language, or one language
asked under several schemas.

A template's placeholders are filled per sample by the prompt inputs the question
declares, so a rendered prompt is produced per sample rather than read once from a
static file. Loading a question checks that its templates and its declared inputs
name each other exactly, and hands back the templates it read, so nothing
downstream reads them from disk a second time.
"""

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
    """Parsed contents of an experiment's experiment.yaml manifest.

    Normalization is the one thing an experiment configures for all its
    questions, since it pools every question's answers into one mapping.
    """

    model_config = ConfigDict(extra="forbid")

    normalize_provider: str = "openai"
    normalize_model: str


class LanguageAsk(BaseModel):
    """One language the question is asked in, and the schemas it is asked under.

    schemas names response schemas the experiment registers, by class name, and
    holds several where one language is asked several ways. A null names no schema
    at all, the arm that sends none and reads the plain text back.
    """

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
    """Parsed contents of a question's question.yaml manifest.

    provider, model and temperature are declared once and apply to every arm, so
    what varies within a question is the language and the schema alone.

    ask is one entry per language, in the order the question declares them.

    Each key under inputs is the template placeholder it fills, and its body is
    passed through to the experiment untouched.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "openai"
    model: str
    temperature: float = 1.0
    ask: list[LanguageAsk] = Field(min_length=1)
    inputs: InputDeclarations = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_entry_per_language(self) -> Self:
        """Reject a language declared twice, since one entry holds all its schemas.

        Two entries for a language would put it in the language list twice, and
        every count keyed by language downstream would double it.
        """
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
    """One language asked under one response schema.

    An arm is what a chart plots as a series and what aggregate keys its counts
    by, so it is what a run varies and nothing else. schema is None for the arm
    that sends none and reads the plain text back.
    """

    schema: type[BaseModel] | None
    lang: str


@dataclass(frozen=True)
class Question:
    """A question resolved against its experiment, ready to run.

    arms lists every (schema, language) pair the question is asked as, in the
    order it declares them. templates holds one loaded template per language,
    since loading the question had to read them all to validate them.
    """

    question_id: str
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
