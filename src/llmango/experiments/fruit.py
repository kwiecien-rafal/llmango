"""Experiment 001: fruit.

Measures how random a model is when asked to pick a fruit from a fixed list,
how the prompt language shifts that pick, and how requesting a structured
response in a non-target language shifts it. The raw pick is preserved and
normalized post-hoc to a canonical English category.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from llmango.registry import ExperimentSpec, SchemaVariant, register_experiment
from llmango.schemas import LLMResponse

EXPERIMENT_ID = "001_fruit"


class FruitChoice(LLMResponse):
    fruit: str


# 001d polish schema
class WyborOwocu(LLMResponse):
    owoc: str


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


_QUALIFIERS = {"a", "an", "the", "fresh", "ripe"}


def preprocess(text: str) -> str:
    """Drop leading articles and qualifiers so answers match the mapping table."""
    tokens = [token for token in text.split() if token not in _QUALIFIERS]
    return " ".join(tokens)


def to_row(parsed: BaseModel | None, raw_text: str) -> dict[str, object]:
    """Map an extracted answer to its parsed column."""
    return {"fruit_raw": raw_text}


register_experiment(
    ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        schema_variants={
            "en": SchemaVariant(schema=FruitChoice, field="fruit"),
            "pl": SchemaVariant(schema=WyborOwocu, field="owoc"),
            "none": SchemaVariant(schema=None, field=None),
        },
        to_row=to_row,
        normalization_schema=FruitNormalization,
        preprocess=preprocess,
        raw_column="fruit_raw",
        canonical_column="fruit_canonical",
        canonical_values=frozenset(member.value for member in FruitEnum) | {"other"},
    )
)
