"""Question config, and the prompt templates and inputs a question names."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmango.config import get_experiment_dir, get_question_dir, sha256_text
from llmango.experiments import spec_for
from llmango.inputs import (
    InputDeclarations,
    InputRequest,
    InputSource,
    ResolvedInput,
    load_input_sources,
    validate_placeholders,
)
from llmango.spec import FREE_TEXT, ArmKey, ExperimentSpec, schema_name

_QUESTION_FILE = "question.yaml"


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

    @property
    def key(self) -> ArmKey:
        """What identifies this arm within its question, and in a manifest."""
        return self.lang, schema_name(self.schema)

    @property
    def label(self) -> str:
        """The name this arm is reported under, FREE_TEXT when it sends no schema."""
        return schema_name(self.schema) or FREE_TEXT


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
    input_sources: dict[str, InputSource]
    prompt_templates: dict[str, PromptTemplate]

    @property
    def languages(self) -> list[str]:
        """Every language the question is asked in, in the order it declares them."""
        return list(dict.fromkeys(arm.lang for arm in self.arms))

    @property
    def input_sha256(self) -> dict[str, str]:
        """The content hash of every input's data file, as a manifest records it."""
        return {name: source.sha256 for name, source in self.input_sources.items()}

    def resolve(self, lang: str, sample_idx: int) -> dict[str, ResolvedInput]:
        """Build every declared input for one sample through the experiment's hook."""
        build_input = self.spec.build_input
        if build_input is None:
            return {}
        return {
            name: build_input(
                InputRequest(
                    name=name,
                    data=self.input_sources[name].data,
                    declaration=declaration,
                    lang=lang,
                    sample_idx=sample_idx,
                )
            )
            for name, declaration in self.inputs.items()
        }


def load_question(question_id: str) -> Question:
    """Load and validate a question by its id (e.g. 001a)."""
    spec = spec_for(question_id)
    question_dir = get_question_dir(spec.folder, question_id)
    question_yaml_path = question_dir / _QUESTION_FILE
    if not question_yaml_path.is_file():
        raise FileNotFoundError(f"Missing question manifest: {question_yaml_path}")

    question_config = QuestionConfig.model_validate(
        yaml.safe_load(question_yaml_path.read_text("utf-8"))
    )
    languages = [entry.language for entry in question_config.ask]
    prompt_templates = _load_prompt_templates(question_dir, languages)
    input_sources = load_input_sources(
        [question_dir, get_experiment_dir(spec.folder)], list(question_config.inputs)
    )

    for lang, prompt_template in prompt_templates.items():
        validate_placeholders(
            prompt_template.text, question_config.inputs, f"{question_id}/{lang}.md"
        )
    if question_config.inputs and spec.build_input is None:
        raise ValueError(
            f"Question {question_id} declares prompt input(s) "
            f"{', '.join(sorted(question_config.inputs))} but experiment {spec.folder} "
            f"registers no build_input hook."
        )

    return Question(
        question_id=question_id,
        spec=spec,
        provider=question_config.provider,
        model=question_config.model,
        temperature=question_config.temperature,
        arms=_resolve_arms(question_config.ask, spec),
        inputs=question_config.inputs,
        input_sources=input_sources,
        prompt_templates=prompt_templates,
    )


def load_prompt_template(question_dir: Path, lang: str) -> PromptTemplate:
    """Load one language's prompt template from a question's folder."""
    prompt_template_path = question_dir / f"{lang}.md"
    if not prompt_template_path.is_file():
        raise FileNotFoundError(f"Missing prompt template: {prompt_template_path}")
    text = prompt_template_path.read_text(encoding="utf-8")
    return PromptTemplate(
        lang=lang, path=prompt_template_path, text=text, sha256=sha256_text(text)
    )


def _load_prompt_templates(
    question_dir: Path, languages: list[str]
) -> dict[str, PromptTemplate]:
    """Load one template per declared language, naming every missing one at once."""
    missing = [
        lang for lang in languages if not (question_dir / f"{lang}.md").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing prompt templates in {question_dir}: {', '.join(missing)}"
        )
    return {lang: load_prompt_template(question_dir, lang) for lang in languages}


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
