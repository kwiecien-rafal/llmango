"""Prompt inputs: the named values that fill a template's placeholders.

A template placeholder {name} is filled by the prompt input named name. The
engine discovers each input's data file, hashes it for the manifest and
substitutes the rendered text. What an input's data means, and how a question's
declaration turns it into text for one sample, belongs to the experiment, which
supplies a build_input hook. Nothing here knows what an experiment asks about.

An input's data file is named after the placeholder it fills, looked up in the
question's folder first and the experiment's folder second, so an input can be
shared across questions or overridden by one. An input with no file at all is
allowed, for a value computed from the declaration alone.
"""

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
    """One input's data file as loaded from disk, with its content hash.

    data is None when the input has no file, which leaves the declaration as the
    experiment's only material. The hash covers the file text verbatim so an edit
    to an input is as traceable in a manifest as an edit to a prompt template.
    """

    data: Any
    sha256: str


@dataclass(frozen=True)
class InputRequest:
    """Everything an experiment needs to build one input for one sample.

    declaration is the question's meta.yaml block for this input, passed through
    verbatim. The engine never reads inside it, so an experiment declares its
    inputs in whatever words fit what it varies.
    """

    name: str
    data: Any
    declaration: Mapping[str, Any]
    lang: str
    sample_idx: int
    seed: int | None


@dataclass(frozen=True)
class ResolvedInput:
    """One input resolved for one sample: what to render, and what to record.

    text replaces the placeholder in the template. value is recorded in the raw
    parquet's prompt_inputs column, so it must be JSON-serializable; None records
    nothing, for an input whose rendered text is already the whole story.
    """

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
    seed: int | None,
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
                seed=seed,
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
    """Check a template's placeholders and a question's inputs name each other.

    Run at config load so a mistyped placeholder fails immediately, rather than
    reaching a model as a literal brace in the prompt.
    """
    wanted = placeholders(template_text)
    declared = set(declarations)
    if undeclared := sorted(wanted - declared):
        raise ValueError(
            f"{label} uses undeclared prompt input(s): {', '.join(undeclared)}. "
            f"Declare them under 'inputs' in meta.yaml."
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
