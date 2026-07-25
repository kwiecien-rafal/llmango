"""Experiment and question config, the shared fruit table, and prompt rendering.

An experiment (001_fruit) is declared by experiment.yaml plus a shared fruits.yaml
and normalize.md. It contains questions (001a, 001b, ...), each a subfolder with
its own meta.yaml and one prompt template per language. A template carries a
{fruit_list} placeholder; the shown order of that list is either a fixed
permutation or a per-sample shuffle, so a rendered prompt is produced per sample
rather than read once from a static file.
"""

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmango.config import PROMPTS_DIR
from llmango.registry import get_experiment, resolve_experiment_id, resolve_schema

_EXPERIMENT_FILE = "experiment.yaml"
_FRUITS_FILE = "fruits.yaml"
_META_FILE = "meta.yaml"
_LIST_PLACEHOLDER = "{fruit_list}"


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
    """Parsed contents of a question's meta.yaml manifest."""

    model_config = ConfigDict(extra="forbid")

    languages: list[str]
    order: Literal["fixed", "shuffle"]
    order_ids: list[str] | None = None
    schema_variants: list[str] = Field(default_factory=lambda: ["en"])
    sampling: SamplingParams | None = None


@dataclass(frozen=True)
class QuestionConfig:
    """A question resolved against its experiment, ready to run."""

    question_id: str
    experiment_id: str
    languages: list[str]
    order: str
    order_ids: list[str] | None
    schema_variants: list[str]
    sampling: SamplingParams
    model: str | None


@dataclass(frozen=True)
class FruitTable:
    """The shared, ordered fruit list with its per-language labels."""

    order: list[str]
    labels: dict[str, dict[str, str]]
    sha256: str

    def canonical_ids(self) -> list[str]:
        """Return the canonical ids in their identity (file) order."""
        return list(self.order)

    def label(self, canonical: str, lang: str) -> str:
        """Return the localized label for one fruit, or raise if absent."""
        labels = self.labels.get(canonical)
        if labels is None:
            raise KeyError(f"fruits.yaml has no fruit {canonical!r}")
        if lang not in labels:
            raise KeyError(f"fruits.yaml has no {lang} label for {canonical}")
        return labels[lang]


@dataclass(frozen=True)
class PromptTemplate:
    """A loaded prompt template with its text and content hash."""

    lang: str
    path: Path
    text: str
    sha256: str


def prompt_sha256(text: str) -> str:
    """Return the hex SHA-256 of a prompt's text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def experiment_dir(experiment_id: str) -> Path:
    """Return an experiment's folder, holding its shared files and questions."""
    return PROMPTS_DIR / experiment_id


def question_dir(experiment_id: str, question_id: str) -> Path:
    """Return one question's folder under its experiment."""
    return experiment_dir(experiment_id) / question_id


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
    that every declared language has a template, that a fixed order is a
    permutation of the fruit set, and that every schema variant is registered.
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

    _validate_order(meta, experiment_id, question_id)
    _validate_schema_variants(meta.schema_variants, experiment_id)

    return QuestionConfig(
        question_id=question_id,
        experiment_id=experiment_id,
        languages=meta.languages,
        order=meta.order,
        order_ids=meta.order_ids,
        schema_variants=meta.schema_variants,
        sampling=meta.sampling or exp_config.sampling,
        model=exp_config.model,
    )


def load_fruits(experiment_id: str) -> FruitTable:
    """Load the experiment's shared fruit table and hash its file contents."""
    path = experiment_dir(experiment_id) / _FRUITS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing fruit table: {path}")
    text = path.read_text(encoding="utf-8")
    entries = cast(list[dict[str, Any]], yaml.safe_load(text) or [])
    order: list[str] = []
    labels: dict[str, dict[str, str]] = {}
    for entry in entries:
        canonical = str(entry["canonical"])
        entry_labels = cast(dict[str, str], entry["labels"])
        order.append(canonical)
        labels[canonical] = {str(k): str(v) for k, v in entry_labels.items()}
    return FruitTable(order=order, labels=labels, sha256=prompt_sha256(text))


def load_template(experiment_id: str, question_id: str, lang: str) -> PromptTemplate:
    """Load one language's prompt template for a question."""
    path = question_dir(experiment_id, question_id) / f"{lang}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    text = path.read_text(encoding="utf-8")
    return PromptTemplate(lang=lang, path=path, text=text, sha256=prompt_sha256(text))


def shown_order(
    table: FruitTable,
    order: str,
    order_ids: list[str] | None,
    sample_idx: int,
    seed: int | None,
) -> list[str]:
    """Return the canonical ids in the order shown for one sample.

    A fixed order is the declared permutation, identical across samples and
    languages. A shuffle is deterministic in (seed, sample_idx) and so is shared
    across languages for a given sample, keeping option position a controlled
    variable.
    """
    if order == "fixed":
        if order_ids is None:
            raise ValueError("A fixed order needs order_ids.")
        return list(order_ids)
    return _shuffled(table.canonical_ids(), seed, sample_idx)


def render_prompt(
    template: PromptTemplate,
    table: FruitTable,
    order: str,
    order_ids: list[str] | None,
    sample_idx: int,
    seed: int | None,
) -> tuple[str, list[str]]:
    """Render one sample's prompt and return it with the shown option order."""
    shown = shown_order(table, order, order_ids, sample_idx, seed)
    labels = ", ".join(table.label(canonical, template.lang) for canonical in shown)
    return template.text.replace(_LIST_PLACEHOLDER, labels), shown


def _shuffled(ids: list[str], seed: int | None, sample_idx: int) -> list[str]:
    """Deterministically shuffle ids from a stable (seed, sample_idx) key."""
    key = f"{seed}:{sample_idx}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return shuffled


def _validate_order(meta: QuestionMeta, experiment_id: str, question_id: str) -> None:
    """Check a fixed order names each fruit exactly once."""
    if meta.order != "fixed":
        return
    canonical = set(load_fruits(experiment_id).canonical_ids())
    order_ids = meta.order_ids or []
    if sorted(order_ids) != sorted(canonical):
        raise ValueError(
            f"{question_id} order_ids must be a permutation of the fruit set; "
            f"got {order_ids}."
        )


def _validate_schema_variants(schema_variants: list[str], experiment_id: str) -> None:
    """Check every declared schema variant is registered for the experiment."""
    known = get_experiment(experiment_id).schema_variants
    unknown = [variant for variant in schema_variants if variant not in known]
    if unknown:
        raise ValueError(
            f"{experiment_id} has no schema variant(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}."
        )
