"""Tests for experiment 001's own hooks: how it arranges and labels its fruit list."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from llmango import config as config_module
from llmango.experiments.fruit import (
    FOLDER,
    FRUIT_LIST,
    FruitEnum,
    build_input,
    extra_normalized_columns,
    mapping_seed,
    preprocess,
)
from llmango.inputs import InputRequest, load_input_sources
from llmango.questions import load_question

_DATA: list[dict[str, Any]] = [
    {"canonical": "apple", "labels": {"en": "apple", "pl": "jabłko"}},
    {"canonical": "pear", "labels": {"en": "pear", "pl": "gruszka"}},
    {"canonical": "mango", "labels": {"en": "mango", "pl": "mango"}},
]

_FIXED: dict[str, Any] = {"order": "fixed", "order_ids": ["mango", "apple", "pear"]}
_SHUFFLE: dict[str, Any] = {"order": "shuffle"}

_SHARED_LIST = "- canonical: apple\n  labels: { en: apple, pl: jabłko }\n"
_OWN_LIST = "- canonical: cherry\n  labels: { en: cherry, pl: wiśnia }\n"


@pytest.fixture
def prompts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point input discovery at a temporary prompts tree."""
    monkeypatch.setattr(config_module, "PROMPTS_DIR", tmp_path)
    return tmp_path


def _question_data(question_id: str) -> Any:
    """Load the real fruit list a question runs against."""
    return load_input_sources(FOLDER, question_id, [FRUIT_LIST])[FRUIT_LIST].data


def _request(
    declaration: dict[str, Any],
    lang: str = "en",
    sample_idx: int = 0,
    data: Any = None,
) -> InputRequest:
    return InputRequest(
        name=FRUIT_LIST,
        data=_DATA if data is None else data,
        declaration=declaration,
        lang=lang,
        sample_idx=sample_idx,
    )


def test_fixed_order_is_the_declared_permutation() -> None:
    resolved = build_input(_request(_FIXED))
    assert resolved.value == ["mango", "apple", "pear"]
    assert resolved.text == "mango, apple, pear"


def test_fixed_order_is_stable_across_samples() -> None:
    first = build_input(_request(_FIXED, sample_idx=0)).value
    ninth = build_input(_request(_FIXED, sample_idx=9)).value
    assert first == ninth == _FIXED["order_ids"]


def test_shuffle_is_a_permutation_that_varies_per_sample() -> None:
    data = _question_data("001c")
    shown = [
        build_input(_request(_SHUFFLE, sample_idx=idx, data=data)).value
        for idx in range(5)
    ]
    assert all(sorted(order) == sorted(shown[0]) for order in shown)
    assert len({tuple(order) for order in shown}) == 5


def test_shuffle_is_identical_across_a_questions_languages() -> None:
    question = load_question("001c")
    declaration = question.inputs[FRUIT_LIST]
    data = _question_data("001c")
    shown = [
        build_input(_request(declaration, lang=lang, sample_idx=7, data=data)).value
        for lang in question.languages
    ]
    assert len(question.languages) == 3
    assert shown[0] == shown[1] == shown[2]


def test_labels_follow_the_prompts_language() -> None:
    assert build_input(_request(_FIXED, lang="pl")).text == "mango, jabłko, gruszka"


def test_fixed_order_must_be_a_permutation() -> None:
    declaration: dict[str, Any] = {"order": "fixed", "order_ids": ["apple", "pear"]}
    with pytest.raises(ValueError, match="must be a permutation"):
        build_input(_request(declaration))


def test_unknown_order_strategy_raises() -> None:
    with pytest.raises(ValueError, match="order must be"):
        build_input(_request({"order": "swap"}))


def test_unknown_input_name_raises() -> None:
    request = InputRequest(
        name="vegetable_list",
        data=_DATA,
        declaration=_FIXED,
        lang="en",
        sample_idx=0,
    )
    with pytest.raises(ValueError, match="no prompt input 'vegetable_list'"):
        build_input(request)


def test_mapping_seed_covers_every_label_in_every_language() -> None:
    seed = mapping_seed()
    assert seed["apple"] == "apple"
    assert seed["jabłko"] == "apple"
    assert seed["りんご"] == "apple"
    assert set(seed.values()) <= {member.value for member in FruitEnum}


def test_mapping_seed_covers_a_questions_own_fruit_list(prompts_dir: Path) -> None:
    experiment = prompts_dir / FOLDER
    (experiment / "001b").mkdir(parents=True)
    (experiment / f"{FRUIT_LIST}.yaml").write_text(_SHARED_LIST, encoding="utf-8")
    (experiment / "001b" / f"{FRUIT_LIST}.yaml").write_text(_OWN_LIST, encoding="utf-8")

    seed = mapping_seed()

    assert seed["jabłko"] == "apple"
    assert seed["wiśnia"] == "cherry"


def test_preprocess_drops_articles_and_qualifiers() -> None:
    assert preprocess("a ripe apple") == "apple"


def _normalized(rows: list[tuple[str, str | None]]) -> pl.DataFrame:
    """Build the two normalized columns the chosen_position hook reads."""
    return pl.DataFrame(
        {
            "prompt_inputs": [shown for shown, _ in rows],
            "canonical": [canonical for _, canonical in rows],
        },
        schema={"prompt_inputs": pl.String(), "canonical": pl.String()},
    )


def test_chosen_position_reports_where_the_answer_was_shown() -> None:
    shown = '{"fruit_list": ["mango", "apple", "banana"]}'
    frame = _normalized([(shown, "mango"), (shown, "apple"), (shown, "banana")])

    columns = extra_normalized_columns(frame)

    assert columns["chosen_position"].to_list() == [1, 2, 3]


def test_chosen_position_is_null_when_the_answer_was_not_shown() -> None:
    """A refusal, an 'other' answer or an off-list fruit all sit at no position."""
    shown = '{"fruit_list": ["mango", "apple"]}'
    frame = _normalized([(shown, "kiwi"), (shown, "other"), (shown, None)])

    columns = extra_normalized_columns(frame)

    assert columns["chosen_position"].to_list() == [None, None, None]
