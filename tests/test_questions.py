"""Tests for experiment and question config loading."""

import pytest

from llmango.experiments.fruit import FruitChoice
from llmango.questions import (
    load_experiment_config,
    load_question,
    load_template,
)
from llmango.registry import resolve_schema

EXPERIMENT_ID = "001_fruit"


def test_load_experiment_config_reads_the_manifest() -> None:
    config = load_experiment_config(EXPERIMENT_ID)
    assert config.experiment_id == EXPERIMENT_ID
    assert config.schema_name == "FruitChoice"
    assert config.model == "gpt-5.6-luna"


def test_load_question_resolves_against_its_experiment() -> None:
    config = load_question("001a")
    assert config.question_id == "001a"
    assert config.experiment_id == EXPERIMENT_ID
    assert config.languages == ["en", "pl", "ja"]
    assert config.schema_variants == ["en"]
    assert config.model == "gpt-5.6-luna"


def test_question_declares_its_prompt_inputs() -> None:
    fixed = load_question("001a")
    assert fixed.inputs["fruit_list"]["order"] == "fixed"
    assert len(fixed.inputs["fruit_list"]["order_ids"]) == 10

    shuffled = load_question("001c")
    assert shuffled.inputs["fruit_list"] == {"order": "shuffle"}


def test_question_with_schema_variants() -> None:
    config = load_question("001d")
    assert config.languages == ["pl"]
    assert config.schema_variants == ["en", "pl", "none"]


def test_every_declared_language_has_a_template() -> None:
    config = load_question("001a")
    for lang in config.languages:
        template = load_template(EXPERIMENT_ID, "001a", lang)
        assert template.lang == lang
        assert "{fruit_list}" in template.text


def test_schema_name_resolves_to_the_registered_schema() -> None:
    config = load_experiment_config(EXPERIMENT_ID)
    assert resolve_schema(EXPERIMENT_ID) is FruitChoice
    assert resolve_schema(EXPERIMENT_ID).__name__ == config.schema_name


def test_load_question_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_question("001z")


def test_load_template_missing_language_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_template(EXPERIMENT_ID, "001a", "xx")
