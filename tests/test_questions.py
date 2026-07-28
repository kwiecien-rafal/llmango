"""Tests for experiment and question config loading."""

import pytest
from pydantic import ValidationError

from llmango.experiments.fruit import FRUIT, FruitChoice, WyborOwocu
from llmango.questions import (
    LanguageAsk,
    QuestionConfig,
    _resolve_arms,
    load_experiment_config,
    load_question,
    load_template,
)

FOLDER = "001_fruit"


def _ask(**schemas_by_language: list[str | None]) -> list[LanguageAsk]:
    """Build the ask entries a question's question.yaml would have declared."""
    return [
        LanguageAsk(language=language, schemas=schemas)
        for language, schemas in schemas_by_language.items()
    ]


def test_load_experiment_config_reads_the_manifest() -> None:
    config = load_experiment_config(FOLDER)
    assert config.normalize_provider == "openai"
    assert config.normalize_model == "gpt-5.4-mini"


def test_a_question_declares_who_answers_it() -> None:
    """Provider, model and temperature are the question's own, one per question."""
    question = load_question("001a")
    assert question.question_id == "001a"
    assert question.provider == "openai"
    assert question.model == "gpt-5.6-luna"
    assert question.temperature == 1.0


def test_an_arm_is_one_language_asked_under_one_schema() -> None:
    """001a asks three languages under one schema, so it is three arms."""
    question = load_question("001a")

    assert [(arm.schema, arm.lang) for arm in question.arms] == [
        (FruitChoice, "en"),
        (FruitChoice, "pl"),
        (FruitChoice, "ja"),
    ]
    assert question.languages == ["en", "pl", "ja"]


def test_question_declares_its_prompt_inputs() -> None:
    fixed = load_question("001a")
    assert fixed.inputs["fruit_list"]["order"] == "fixed"
    assert len(fixed.inputs["fruit_list"]["order_ids"]) == 10

    shuffled = load_question("001c")
    assert shuffled.inputs["fruit_list"] == {"order": "shuffle"}


def test_one_language_asked_several_ways_is_several_arms() -> None:
    """001d asks Polish three ways, so it is three arms of one language."""
    question = load_question("001d")

    assert [arm.schema for arm in question.arms] == [FruitChoice, WyborOwocu, None]
    assert all(arm.lang == "pl" for arm in question.arms)
    assert question.languages == ["pl"]


def test_each_language_may_name_its_own_schema() -> None:
    """A question whose schema follows its language is one arm per language."""
    arms = _resolve_arms(_ask(en=["FruitChoice"], pl=["WyborOwocu"]), FRUIT)

    assert [(arm.schema, arm.lang) for arm in arms] == [
        (FruitChoice, "en"),
        (WyborOwocu, "pl"),
    ]


def test_a_schema_the_experiment_does_not_register_raises() -> None:
    with pytest.raises(ValueError, match="registers no schema named 'Missing'"):
        _resolve_arms(_ask(en=["Missing"]), FRUIT)


def test_a_language_asked_twice_under_one_schema_raises() -> None:
    with pytest.raises(ValidationError, match="asks under FruitChoice more than once"):
        LanguageAsk(language="en", schemas=["FruitChoice", "FruitChoice"])


def test_a_language_declared_twice_raises() -> None:
    """One entry per language, so every schema a language is asked under is together."""
    config = {
        "model": "gpt-5.6-luna",
        "ask": [
            {"language": "en", "schemas": ["FruitChoice"]},
            {"language": "en", "schemas": [None]},
        ],
    }

    with pytest.raises(ValidationError, match="declared more than once: en"):
        QuestionConfig.model_validate(config)


def test_every_declared_language_has_a_template() -> None:
    question = load_question("001a")
    for lang in question.languages:
        template = load_template(FOLDER, "001a", lang)
        assert template.lang == lang
        assert "{fruit_list}" in template.text


def test_load_question_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown question"):
        load_question("001z")


def test_load_template_missing_language_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_template(FOLDER, "001a", "xx")
