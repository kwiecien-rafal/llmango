"""Experiment 001: favorite_fruit.

Asks a model to name its favorite fruit, then normalizes the free-text answer to
a canonical English category.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from llmango.registry import ExperimentSpec, register_experiment
from llmango.schemas import LLMResponse

QUESTION_ID = "favorite_fruit"


class FruitChoice(LLMResponse):
    """A model's favorite fruit"""

    fruit: str


class FruitEnum(StrEnum):
    """Canonical fruit categories, seeded from common answers.

    Culture-specific fruits keep their own value rather than collapsing into a
    nearby Western fruit, so the variation being studied is preserved.
    """

    APPLE = "apple"
    BANANA = "banana"
    ORANGE = "orange"
    MANGO = "mango"
    STRAWBERRY = "strawberry"
    GRAPE = "grape"
    WATERMELON = "watermelon"
    PINEAPPLE = "pineapple"
    PEACH = "peach"
    PEAR = "pear"
    CHERRY = "cherry"
    LEMON = "lemon"
    LIME = "lime"
    KIWI = "kiwi"
    BLUEBERRY = "blueberry"
    RASPBERRY = "raspberry"
    BLACKBERRY = "blackberry"
    PLUM = "plum"
    POMEGRANATE = "pomegranate"
    APRICOT = "apricot"
    FIG = "fig"
    MELON = "melon"
    COCONUT = "coconut"
    PAPAYA = "papaya"
    AVOCADO = "avocado"
    TOMATO = "tomato"
    PERSIMMON = "persimmon"
    LYCHEE = "lychee"
    DRAGONFRUIT = "dragonfruit"
    GUAVA = "guava"
    PASSIONFRUIT = "passionfruit"
    DURIAN = "durian"


class FruitNormalization(LLMResponse):
    """A raw fruit answer mapped to a canonical category."""

    raw: str
    canonical: FruitEnum | Literal["other"]
    is_fruit: bool
    multiple: bool


_QUALIFIERS = {"a", "an", "the", "my", "favorite", "favourite", "fresh", "ripe"}


def preprocess(text: str) -> str:
    """Drop leading articles and qualifiers so answers match the mapping table."""
    tokens = [token for token in text.split() if token not in _QUALIFIERS]
    return " ".join(tokens)


def to_row(parsed: BaseModel | None) -> dict[str, object]:
    """Map a parsed response to its parsed columns, empty on refusal or error."""
    fruit = parsed.fruit if isinstance(parsed, FruitChoice) else ""
    return {"fruit_raw": fruit}


register_experiment(
    ExperimentSpec(
        question_id=QUESTION_ID,
        response_model=FruitChoice,
        slug="001_favorite_fruit",
        to_row=to_row,
        normalization_model=FruitNormalization,
        preprocess=preprocess,
        raw_column="fruit_raw",
        canonical_column="fruit_canonical",
        canonical_values=frozenset(member.value for member in FruitEnum) | {"other"},
    )
)
