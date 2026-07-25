"""Tests for experiment/question config loading and prompt rendering."""

import pytest

from llmango.experiments.fruit import FruitChoice
from llmango.questions import (
    load_experiment_config,
    load_fruits,
    load_question,
    load_template,
    prompt_sha256,
    render_prompt,
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
    assert config.order == "fixed"
    assert config.schema_variants == ["en"]
    assert config.model == "gpt-5.6-luna"


def test_question_with_schema_variants() -> None:
    config = load_question("001d")
    assert config.languages == ["pl"]
    assert config.order == "shuffle"
    assert config.schema_variants == ["en", "pl", "none"]


def test_every_declared_language_has_a_template() -> None:
    config = load_question("001a")
    for lang in config.languages:
        template = load_template(EXPERIMENT_ID, "001a", lang)
        assert template.lang == lang
        assert "{fruit_list}" in template.text


def test_prompt_sha256_is_deterministic() -> None:
    assert prompt_sha256("hello") == prompt_sha256("hello")
    assert prompt_sha256("hello") != prompt_sha256("world")


def test_schema_name_resolves_to_the_registered_schema() -> None:
    config = load_experiment_config(EXPERIMENT_ID)
    assert resolve_schema(EXPERIMENT_ID) is FruitChoice
    assert resolve_schema(EXPERIMENT_ID).__name__ == config.schema_name


def test_fixed_order_is_stable_across_samples() -> None:
    config = load_question("001a")
    table = load_fruits(EXPERIMENT_ID)
    template = load_template(EXPERIMENT_ID, "001a", "en")
    _, first = render_prompt(template, table, config.order, config.order_ids, 0, 42)
    _, second = render_prompt(template, table, config.order, config.order_ids, 9, 42)
    assert first == second == config.order_ids


def test_shuffle_varies_per_sample_but_is_shared_across_languages() -> None:
    config = load_question("001c")
    table = load_fruits(EXPERIMENT_ID)
    en = load_template(EXPERIMENT_ID, "001c", "en")
    ja = load_template(EXPERIMENT_ID, "001c", "ja")
    _, en0 = render_prompt(en, table, config.order, config.order_ids, 0, 42)
    _, en1 = render_prompt(en, table, config.order, config.order_ids, 1, 42)
    _, ja0 = render_prompt(ja, table, config.order, config.order_ids, 0, 42)
    assert en0 != en1
    assert en0 == ja0
    assert sorted(en0) == sorted(table.canonical_ids())


def test_rendered_prompt_lists_localized_labels_in_order() -> None:
    config = load_question("001a")
    table = load_fruits(EXPERIMENT_ID)
    template = load_template(EXPERIMENT_ID, "001a", "pl")
    prompt, order = render_prompt(template, table, config.order, config.order_ids, 0, 1)
    expected = ", ".join(table.label(canonical, "pl") for canonical in order)
    assert expected in prompt
    assert "{fruit_list}" not in prompt


def test_load_question_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_question("001z")


def test_load_template_missing_language_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_template(EXPERIMENT_ID, "001a", "xx")
