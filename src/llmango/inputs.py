"""Prompt inputs: the named values that fill a template's placeholders."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llmango.config import experiment_dir, question_dir, sha256_text

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

InputDeclarations = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class InputSource:
    """One input's data file as loaded from disk, with its content hash."""

    data: Any
    sha256: str


@dataclass(frozen=True)
class InputRequest:
    """Everything an experiment needs to build one input for one sample."""

    name: str
    data: Any
    declaration: Mapping[str, Any]
    lang: str
    sample_idx: int


@dataclass(frozen=True)
class ResolvedInput:
    """One input resolved for one sample: text to render, JSON value to record."""

    text: str
    value: Any = None


BuildInput = Callable[[InputRequest], ResolvedInput]


def placeholders(text: str) -> set[str]:
    """Return the input names a template asks for."""
    return set(_PLACEHOLDER.findall(text))


def load_input_sources(
    folder: str, question_id: str | None, names: list[str]
) -> dict[str, InputSource]:
    """Load the data file behind each named input, question folder winning."""
    return {name: _load_source(folder, question_id, name) for name in names}


def resolve(
    build_input: BuildInput | None,
    sources: Mapping[str, InputSource],
    declarations: InputDeclarations,
    lang: str,
    sample_idx: int,
    question_id: str,
) -> dict[str, ResolvedInput]:
    """Build every declared input for one sample through the experiment's hook."""
    if not declarations:
        return {}
    if build_input is None:
        raise ValueError(
            f"Question {question_id} declares prompt input(s) "
            f"{', '.join(sorted(declarations))} but its experiment registers no "
            f"build_input hook."
        )
    return {
        name: build_input(
            InputRequest(
                name=name,
                data=sources[name].data,
                declaration=declaration,
                lang=lang,
                sample_idx=sample_idx,
            )
        )
        for name, declaration in declarations.items()
    }


def render(template_text: str, resolved: Mapping[str, ResolvedInput]) -> str:
    """Substitute every resolved input into a template's placeholders."""
    text = template_text
    for name, value in resolved.items():
        text = text.replace(f"{{{name}}}", value.text)
    return text


def validate_placeholders(
    template_text: str, declarations: InputDeclarations, label: str
) -> None:
    """Check a template's placeholders and a question's inputs name each other."""
    wanted = placeholders(template_text)
    declared = set(declarations)
    if undeclared := sorted(wanted - declared):
        raise ValueError(
            f"{label} uses undeclared prompt input(s): {', '.join(undeclared)}. "
            f"Declare them under 'inputs' in question.yaml."
        )
    if unused := sorted(declared - wanted):
        raise ValueError(
            f"{label} declares prompt input(s) it never uses: {', '.join(unused)}."
        )


def _load_source(folder: str, question_id: str | None, name: str) -> InputSource:
    """Read one input's YAML file, or record its absence."""
    path = _find_file(folder, question_id, name)
    if path is None:
        return InputSource(data=None, sha256=sha256_text(""))
    text = path.read_text(encoding="utf-8")
    return InputSource(data=yaml.safe_load(text), sha256=sha256_text(text))


def _find_file(folder: str, question_id: str | None, name: str) -> Path | None:
    """Return the input's data file, preferring the question's own copy."""
    candidates = (
        [question_dir(folder, question_id) / f"{name}.yaml"] if question_id else []
    )
    candidates.append(experiment_dir(folder) / f"{name}.yaml")
    return next((path for path in candidates if path.is_file()), None)
