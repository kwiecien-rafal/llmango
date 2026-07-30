"""Tests for question config and prompt template loading."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from llmango import questions as questions_module
from llmango.config import get_question_dir
from llmango.experiments.e001_fruit.experiment import FRUIT, FruitChoice, WyborOwocu
from llmango.questions import (
    LanguageAsk,
    QuestionConfig,
    _resolve_arms,
    load_prompt_template,
    load_question,
)

FOLDER = "e001_fruit"


def _ask(**schemas_by_language: list[str | None]) -> list[LanguageAsk]:
    """Build the ask entries a question's question.yaml would have declared."""
    return [
        LanguageAsk(language=language, schemas=schemas)
        for language, schemas in schemas_by_language.items()
    ]


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


def test_a_question_loads_its_inputs_and_hashes_them() -> None:
    """The question owns the data behind its inputs, so it can hash and build them."""
    question = load_question("001a")

    source = question.input_sources["fruit_list"]
    assert source.data is not None
    assert question.input_sha256 == {"fruit_list": source.sha256}


def test_resolve_builds_each_declared_input_for_one_sample() -> None:
    question = load_question("001a")

    resolved = question.resolve("pl", 3)

    assert set(resolved) == {"fruit_list"}
    assert resolved["fruit_list"].value == question.inputs["fruit_list"]["order_ids"]
    assert "jabłko" in resolved["fruit_list"].text


def test_resolve_asks_the_hook_for_the_sample_it_is_given() -> None:
    """A shuffled question resolves per sample, so the index reaches the hook."""
    question = load_question("001c")

    first = question.resolve("en", 0)
    second = question.resolve("en", 1)

    assert first["fruit_list"].value != second["fruit_list"].value
    assert sorted(first["fruit_list"].value) == sorted(second["fruit_list"].value)


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
        template = load_prompt_template(get_question_dir(FOLDER, "001a"), lang)
        assert template.lang == lang
        assert "{fruit_list}" in template.text


def test_a_question_declaring_inputs_needs_a_hook_to_build_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused at load, not once per sample, since the pairing cannot change."""
    monkeypatch.setattr(
        questions_module, "spec_for", lambda _: replace(FRUIT, build_input=None)
    )

    with pytest.raises(ValueError, match="registers no build_input hook"):
        load_question("001a")


def test_load_question_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown question"):
        load_question("001z")


def test_load_template_missing_language_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt_template(get_question_dir(FOLDER, "001a"), "xx")
