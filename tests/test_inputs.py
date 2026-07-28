"""Tests for prompt input discovery, hashing, rendering and validation."""

from pathlib import Path

import pytest

from llmango import config as config_module
from llmango.config import sha256_text
from llmango.inputs import (
    InputRequest,
    ResolvedInput,
    load_input_sources,
    placeholders,
    render,
    resolve,
    validate_placeholders,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def prompts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point input discovery at a temporary prompts tree."""
    monkeypatch.setattr(config_module, "PROMPTS_DIR", tmp_path)
    return tmp_path


def test_placeholders_reads_the_names_a_template_asks_for() -> None:
    assert placeholders("pick from {fruit_list} or {other_list}") == {
        "fruit_list",
        "other_list",
    }
    assert placeholders("no placeholders here") == set()


def test_input_file_is_found_at_the_experiment_level(prompts_dir: Path) -> None:
    _write(prompts_dir / "001_fruit" / "fruit_list.yaml", "- apple\n")
    sources = load_input_sources("001_fruit", "001a", ["fruit_list"])
    assert sources["fruit_list"].data == ["apple"]


def test_a_questions_own_file_wins_over_the_experiments(prompts_dir: Path) -> None:
    _write(prompts_dir / "001_fruit" / "fruit_list.yaml", "- apple\n")
    _write(prompts_dir / "001_fruit" / "001a" / "fruit_list.yaml", "- banana\n")
    shared = load_input_sources("001_fruit", "001b", ["fruit_list"])["fruit_list"]
    own = load_input_sources("001_fruit", "001a", ["fruit_list"])["fruit_list"]
    assert shared.data == ["apple"]
    assert own.data == ["banana"]
    assert own.sha256 != shared.sha256


def test_an_input_may_have_no_file(prompts_dir: Path) -> None:
    source = load_input_sources("001_fruit", "001a", ["computed"])["computed"]
    assert source.data is None
    assert source.sha256 == sha256_text("")


def test_source_hash_tracks_the_file_text(prompts_dir: Path) -> None:
    path = prompts_dir / "001_fruit" / "fruit_list.yaml"
    _write(path, "- apple\n")
    first = load_input_sources("001_fruit", None, ["fruit_list"])["fruit_list"].sha256
    assert first == sha256_text("- apple\n")
    _write(path, "- apple\n- banana\n")
    second = load_input_sources("001_fruit", None, ["fruit_list"])["fruit_list"].sha256
    assert second != first


def test_resolve_passes_the_declaration_through_untouched(prompts_dir: Path) -> None:
    seen: list[InputRequest] = []

    def build(request: InputRequest) -> ResolvedInput:
        seen.append(request)
        return ResolvedInput(text="a, b", value=["a", "b"])

    declarations = {"fruit_list": {"order": "swap", "pairs": [[1, 2]]}}
    sources = load_input_sources("001_fruit", "001a", ["fruit_list"])
    resolved = resolve(build, sources, declarations, "pl", 3, "001a")

    assert resolved["fruit_list"].value == ["a", "b"]
    assert seen[0].declaration == {"order": "swap", "pairs": [[1, 2]]}
    assert (seen[0].lang, seen[0].sample_idx) == ("pl", 3)


def test_resolve_without_a_hook_names_the_question() -> None:
    with pytest.raises(ValueError, match="Question 001a declares prompt input"):
        resolve(None, {}, {"fruit_list": {}}, "en", 0, "001a")


def test_resolve_with_no_declared_inputs_is_empty() -> None:
    assert resolve(None, {}, {}, "en", 0, "001a") == {}


def test_render_substitutes_every_input() -> None:
    resolved = {
        "fruit_list": ResolvedInput(text="apple, pear"),
        "tone": ResolvedInput(text="briefly"),
    }
    assert render("Pick {briefly}", {}) == "Pick {briefly}"
    assert render("{tone}: {fruit_list}", resolved) == "briefly: apple, pear"


def test_validate_placeholders_rejects_an_undeclared_placeholder() -> None:
    with pytest.raises(ValueError, match="undeclared prompt input\\(s\\): fruit_list"):
        validate_placeholders("pick from {fruit_list}", {}, "001a/en.md")


def test_validate_placeholders_rejects_an_unused_input() -> None:
    with pytest.raises(ValueError, match="never uses: fruit_list"):
        validate_placeholders("pick a fruit", {"fruit_list": {}}, "001a/en.md")


def test_validate_placeholders_accepts_a_matching_pair() -> None:
    validate_placeholders("pick from {fruit_list}", {"fruit_list": {}}, "001a/en.md")
