"""Tests for experiment and question config loading."""

import pytest
from pydantic import ValidationError

from llmango.experiments.fruit import FRUIT, FruitChoice, WyborOwocu
from llmango.questions import (
    LanguageAsk,
    QuestionMeta,
    _resolve_arms,
    load_experiment_config,
    load_question,
    load_template,
)

FOLDER = "001_fruit"


def _ask(**schemas_by_language: list[str | None]) -> list[LanguageAsk]:
    """Build the ask entries a question's meta.yaml would have declared."""
    return [
        LanguageAsk(language=language, schemas=schemas)
        for language, schemas in schemas_by_language.items()
    ]


def test_load_experiment_config_reads_the_manifest() -> None:
    config = load_experiment_config(FOLDER)
    assert config.model == "gpt-5.6-luna"
    assert config.normalize_model == "gpt-5.4-mini"


def test_load_question_resolves_against_its_experiment() -> None:
    config = load_question("001a")
    assert config.question_id == "001a"
    assert config.languages == ["en", "pl", "ja"]
    assert config.model == "gpt-5.6-luna"


def test_languages_asked_the_same_way_are_one_arm() -> None:
    """001a asks three languages under one schema, so it is one run of three."""
    arms = load_question("001a").arms

    assert len(arms) == 1
    assert arms[0].schema is FruitChoice
    assert arms[0].languages == ["en", "pl", "ja"]


def test_question_declares_its_prompt_inputs() -> None:
    fixed = load_question("001a")
    assert fixed.inputs["fruit_list"]["order"] == "fixed"
    assert len(fixed.inputs["fruit_list"]["order_ids"]) == 10

    shuffled = load_question("001c")
    assert shuffled.inputs["fruit_list"] == {"order": "shuffle"}


def test_one_language_asked_several_ways_is_several_arms() -> None:
    """001d asks Polish under three schemas, so it is three runs of one language."""
    config = load_question("001d")

    assert config.languages == ["pl"]
    assert [arm.schema for arm in config.arms] == [FruitChoice, WyborOwocu, None]
    assert all(arm.languages == ["pl"] for arm in config.arms)


def test_each_language_may_name_its_own_schema() -> None:
    """A question whose schema follows its language is one arm per language."""
    arms = _resolve_arms("001x", _ask(en=["FruitChoice"], pl=["WyborOwocu"]), FRUIT)

    assert [(arm.schema, arm.languages) for arm in arms] == [
        (FruitChoice, ["en"]),
        (WyborOwocu, ["pl"]),
    ]


def test_a_schema_the_experiment_does_not_register_raises() -> None:
    with pytest.raises(ValueError, match="registers no schema named 'Missing'"):
        _resolve_arms("001x", _ask(en=["Missing"]), FRUIT)


def test_a_language_asked_twice_under_one_schema_raises() -> None:
    with pytest.raises(ValueError, match="under FruitChoice more than once"):
        _resolve_arms("001x", _ask(en=["FruitChoice", "FruitChoice"]), FRUIT)


def test_a_language_declared_twice_raises() -> None:
    """One entry per language, so every schema a language is asked under is together."""
    meta = {
        "ask": [
            {"language": "en", "schemas": ["FruitChoice"]},
            {"language": "en", "schemas": [None]},
        ]
    }

    with pytest.raises(ValidationError, match="declared more than once: en"):
        QuestionMeta.model_validate(meta)


def test_every_declared_language_has_a_template() -> None:
    config = load_question("001a")
    for lang in config.languages:
        template = load_template(FOLDER, "001a", lang)
        assert template.lang == lang
        assert "{fruit_list}" in template.text


def test_load_question_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown question"):
        load_question("001z")


def test_load_template_missing_language_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_template(FOLDER, "001a", "xx")
