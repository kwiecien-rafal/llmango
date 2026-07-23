"""Tests for question config loading and prompt file access."""

import pytest

from llmango.experiments.favorite_fruit import FruitChoice
from llmango.questions import load_prompt, load_question, prompt_sha256
from llmango.registry import resolve_schema

QUESTION_ID = "001_favorite_fruit"


def test_load_question_reads_the_manifest() -> None:
    config = load_question(QUESTION_ID)
    assert config.question_id == QUESTION_ID
    assert config.schema_name == "FruitChoice"
    assert config.model == "gpt-5.6-luna"
    assert config.languages == ["en", "pl"]


def test_every_declared_language_has_a_prompt_file() -> None:
    config = load_question(QUESTION_ID)
    for lang in config.languages:
        prompt = load_prompt(QUESTION_ID, lang)
        assert prompt.lang == lang
        assert prompt.text.strip()


def test_prompt_sha256_is_deterministic() -> None:
    assert prompt_sha256("hello") == prompt_sha256("hello")
    assert prompt_sha256("hello") != prompt_sha256("world")


def test_load_prompt_hashes_its_own_text() -> None:
    prompt = load_prompt(QUESTION_ID, "en")
    assert prompt.sha256 == prompt_sha256(prompt.text)


def test_schema_name_resolves_to_the_registered_model() -> None:
    config = load_question(QUESTION_ID)
    assert resolve_schema(QUESTION_ID) is FruitChoice
    assert resolve_schema(QUESTION_ID).__name__ == config.schema_name


def test_load_question_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_question("does_not_exist")


def test_load_prompt_missing_language_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt(QUESTION_ID, "xx")
