"""Experiment 001: fruit.

Measures how random a model is when asked to pick a fruit from a fixed list,
how the prompt language shifts that pick, and how requesting a structured
response in a non-target language shifts it. The raw pick is preserved and
normalized post-hoc to a canonical English category.

The one prompt input is fruit_list, whose data file holds the canonical ids with
their per-language labels. A question declares how that list is arranged, either
a fixed permutation or a per-sample shuffle. Both live here rather than in the
engine, because the arrangement is what this experiment varies; a later
experiment arranges its own inputs its own way without touching shared code.
"""

import hashlib
import random
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import BaseModel

from llmango.inputs import InputRequest, ResolvedInput, load_input_sources
from llmango.registry import (
    FREE_TEXT_VARIANT,
    ExperimentSpec,
    SchemaVariant,
    register_experiment,
)
from llmango.schemas import LLMResponse

EXPERIMENT_ID = "001_fruit"
FRUIT_LIST = "fruit_list"

_ORDER_FIXED = "fixed"
_ORDER_SHUFFLE = "shuffle"


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


def build_input(request: InputRequest) -> ResolvedInput:
    """Render the fruit list for one sample in one language.

    Returns the localized labels as the model sees them, and the canonical ids in
    the same order as the value recorded per row, which is what lets chosen
    position be resolved later from a free-text answer.
    """
    if request.name != FRUIT_LIST:
        raise ValueError(f"{EXPERIMENT_ID} has no prompt input {request.name!r}.")
    table = _table(request.data)
    shown = _shown_order(table, request.declaration, request.sample_idx, request.seed)
    labels = ", ".join(_label(table, canonical, request.lang) for canonical in shown)
    return ResolvedInput(text=labels, value=shown)


def mapping_seed(question_ids: list[str]) -> dict[str, str]:
    """Map every fruit label in every language onto its canonical id.

    Seeds normalization so an answer that named one of the listed fruits resolves
    without an LLM call, whichever language it was asked in. The experiment's own
    fruit list is the base and every question is read on top of it, since a
    question that overrides the list with its own file shows labels the experiment
    file does not have.
    """
    mapping: dict[str, str] = {}
    for question_id in [None, *question_ids]:
        sources = load_input_sources(EXPERIMENT_ID, question_id, [FRUIT_LIST])
        for canonical, labels in _table(sources[FRUIT_LIST].data).items():
            for label in labels.values():
                mapping[label] = canonical
    return mapping


def _table(data: Any) -> dict[str, dict[str, str]]:
    """Read the fruit list file into canonical ids with their labels, in file order."""
    if not isinstance(data, list):
        raise ValueError(f"{FRUIT_LIST}.yaml must hold a list of fruits.")
    entries = cast(list[dict[str, Any]], data)
    return {
        str(entry["canonical"]): {
            str(lang): str(label) for lang, label in entry["labels"].items()
        }
        for entry in entries
    }


def _label(table: dict[str, dict[str, str]], canonical: str, lang: str) -> str:
    """Return one fruit's label in the prompt's language, or raise if absent."""
    labels = table.get(canonical)
    if labels is None:
        raise ValueError(f"{FRUIT_LIST}.yaml has no fruit {canonical!r}")
    if lang not in labels:
        raise ValueError(f"{FRUIT_LIST}.yaml has no {lang} label for {canonical}")
    return labels[lang]


def _shown_order(
    table: dict[str, dict[str, str]],
    declaration: Mapping[str, Any],
    sample_idx: int,
    seed: int | None,
) -> list[str]:
    """Return the canonical ids in the order one sample shows them.

    A fixed order is the declared permutation, identical across samples and
    languages. A shuffle is deterministic in (seed, sample_idx) and so is shared
    across languages for a given sample, keeping fruit position a controlled
    variable rather than a second thing varying alongside language.
    """
    order = declaration.get("order")
    if order == _ORDER_SHUFFLE:
        return _shuffled(list(table), seed, sample_idx)
    if order != _ORDER_FIXED:
        raise ValueError(
            f"{FRUIT_LIST} order must be {_ORDER_FIXED!r} or {_ORDER_SHUFFLE!r}; "
            f"got {order!r}."
        )
    declared = cast(list[Any], declaration.get("order_ids") or [])
    order_ids = [str(canonical) for canonical in declared]
    if sorted(order_ids) != sorted(table):
        raise ValueError(
            f"{FRUIT_LIST} order_ids must be a permutation of the fruit list; "
            f"got {order_ids}."
        )
    return order_ids


def _shuffled(ids: list[str], seed: int | None, sample_idx: int) -> list[str]:
    """Deterministically shuffle ids from a stable (seed, sample_idx) key."""
    key = f"{seed}:{sample_idx}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return shuffled


register_experiment(
    ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        schema_variants={
            "en": SchemaVariant(schema=FruitChoice, field="fruit"),
            "pl": SchemaVariant(schema=WyborOwocu, field="owoc"),
            FREE_TEXT_VARIANT: SchemaVariant(schema=None, field=None),
        },
        to_row=to_row,
        normalization_schema=FruitNormalization,
        preprocess=preprocess,
        build_input=build_input,
        mapping_seed=mapping_seed,
        position_input=FRUIT_LIST,
        raw_column="fruit_raw",
        canonical_column="fruit_canonical",
        valid_column="is_fruit",
    )
)
