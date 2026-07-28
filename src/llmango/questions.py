"""Experiment and question config, and the prompt templates they name.

An experiment's shared files are experiment.yaml plus normalize.md, in the folder
its spec names. Each question it owns is a subfolder with its own meta.yaml and
one prompt template per language.

A question declares one thing: every language it is asked in, each with the
response schemas it is asked under. There is one entry per language and no second
list to keep in step with it, so the three shapes a question can take all read the
same way: one schema for every language, a schema of its own per language, or one
language asked under several schemas.

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
_META_FILE = "meta.yaml"


class SamplingParams(BaseModel):
    """Sampling parameters passed to a generation backend."""

    temperature: float = 1.0
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None


class ExperimentConfig(BaseModel):
    """Parsed contents of an experiment's experiment.yaml manifest."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    normalize_model: str | None = None
    sampling: SamplingParams = Field(default_factory=SamplingParams)


class LanguageAsk(BaseModel):
    """One language the question is asked in, and the schemas it is asked under.

    schemas names response schemas the experiment registers, by class name, and
    holds several where one language is asked several ways. A null names no schema
    at all, the arm that sends none and reads the plain text back.
    """

    model_config = ConfigDict(extra="forbid")

    language: str
    schemas: list[str | None] = Field(min_length=1)


class QuestionMeta(BaseModel):
    """Parsed contents of a question's meta.yaml manifest.

    ask is one entry per language, in the order the question declares them, which
    is the language list as well.

    Each key under inputs is the template placeholder it fills, and its body is
    passed through to the experiment untouched.
    """

    model_config = ConfigDict(extra="forbid")

    ask: list[LanguageAsk] = Field(min_length=1)
    inputs: InputDeclarations = Field(default_factory=dict)
    sampling: SamplingParams | None = None

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
    """One response schema and every language a question asks under it.

    A run covers one arm, so a question asked the same way in three languages is
    one run of three languages, and a question asking one language under three
    schemas is three runs of one. schema is None for the arm that sends none.
    """

    schema: type[BaseModel] | None
    languages: list[str]


@dataclass(frozen=True)
class QuestionConfig:
    """A question resolved against its experiment, ready to run.

    languages lists every language the question declares, in the order it declares
    them; arms groups those same languages by the schema they are asked under.
    templates holds one loaded template per language, since loading the question
    had to read them all to validate them.
    """

    question_id: str
    languages: list[str]
    arms: list[Arm]
    inputs: InputDeclarations
    sampling: SamplingParams
    model: str | None
    templates: dict[str, PromptTemplate]


def load_experiment_config(folder: str) -> ExperimentConfig:
    """Load and validate an experiment's experiment.yaml manifest."""
    path = experiment_dir(folder) / _EXPERIMENT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)


def load_question(question_id: str) -> QuestionConfig:
    """Load and validate a question by its id (e.g. 001a)."""
    spec = spec_for(question_id)
    directory = question_dir(spec.folder, question_id)
    meta_path = directory / _META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing question manifest: {meta_path}")

    meta = QuestionMeta.model_validate(yaml.safe_load(meta_path.read_text("utf-8")))
    exp_config = load_experiment_config(spec.folder)
    languages = [entry.language for entry in meta.ask]
    templates = _load_templates(spec.folder, question_id, languages)

    for lang, template in templates.items():
        validate_placeholders(template.text, meta.inputs, f"{question_id}/{lang}.md")

    return QuestionConfig(
        question_id=question_id,
        languages=languages,
        arms=_resolve_arms(question_id, meta.ask, spec),
        inputs=meta.inputs,
        sampling=meta.sampling or exp_config.sampling,
        model=exp_config.model,
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


def _resolve_arms(
    question_id: str, ask: list[LanguageAsk], spec: ExperimentSpec
) -> list[Arm]:
    """Group the declared languages into one arm per schema.

    Every language asked under the same schema belongs to one arm, in the order
    the schemas are first named, so a question asked the same way in three
    languages runs once rather than three times.
    """
    grouped: dict[str | None, list[str]] = {}
    for entry in ask:
        for name in entry.schemas:
            languages = grouped.setdefault(name, [])
            if entry.language in languages:
                raise ValueError(
                    f"Question {question_id} asks {entry.language} under "
                    f"{name or 'no schema'} more than once."
                )
            languages.append(entry.language)
    return [
        Arm(
            schema=spec.schema_named(name) if name is not None else None,
            languages=langs,
        )
        for name, langs in grouped.items()
    ]
