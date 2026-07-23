"""Question config and prompt loading.

A question is declared by a meta.yaml manifest plus one prompt file per language
under prompts/<question_id>/. This module reads those files, hashes each prompt
for traceability, and validates that the manifest agrees with the registered
experiment spec.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmango.config import PROMPTS_DIR
from llmango.registry import get_experiment, resolve_schema


class SamplingParams(BaseModel):
    """Sampling parameters passed to a generation backend."""

    temperature: float = 1.0
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None


class QuestionConfig(BaseModel):
    """Parsed contents of a question's meta.yaml manifest."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    question_id: str
    schema_name: str = Field(alias="schema")
    languages: list[str]
    model: str | None = None
    normalize_model: str | None = None
    sampling: SamplingParams = Field(default_factory=SamplingParams)


@dataclass(frozen=True)
class PromptFile:
    """A loaded prompt file with its text and content hash."""

    lang: str
    path: Path
    text: str
    sha256: str


def prompt_sha256(text: str) -> str:
    """Return the hex SHA-256 of a prompt's text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def experiment_dir(question_id: str) -> Path:
    """Return an experiment's folder, named by its slug and holding its prompts."""
    _register_experiments()
    try:
        slug = get_experiment(question_id).slug or question_id
    except KeyError:
        slug = question_id
    return PROMPTS_DIR / slug


def load_prompt(question_id: str, lang: str) -> PromptFile:
    """Load one language's prompt file for a question."""
    path = experiment_dir(question_id) / f"{lang}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    text = path.read_text(encoding="utf-8")
    return PromptFile(lang=lang, path=path, text=text, sha256=prompt_sha256(text))


def load_question(question_id: str) -> QuestionConfig:
    """Load and validate a question's meta.yaml manifest.

    Checks that the manifest names its own question_id, that its declared schema
    matches the registered experiment model, and that every declared language
    has a prompt file.
    """
    directory = experiment_dir(question_id)
    meta_path = directory / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing question manifest: {meta_path}")

    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    config = QuestionConfig.model_validate(data)

    if config.question_id != question_id:
        raise ValueError(
            f"meta.yaml question_id '{config.question_id}' does not match "
            f"requested '{question_id}'"
        )

    registered = resolve_schema(question_id).__name__
    if registered != config.schema_name:
        raise ValueError(
            f"meta.yaml schema '{config.schema_name}' does not match registered "
            f"model '{registered}'"
        )

    missing = [
        lang for lang in config.languages if not (directory / f"{lang}.md").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing prompt files for {question_id}: {', '.join(missing)}"
        )

    return config


def _register_experiments() -> None:
    """Import the experiments package so every ExperimentSpec is registered."""
    from llmango.experiments import ensure_registered

    ensure_registered()
