"""Experiment 001: how a model picks one fruit, and how language shifts the pick."""

import hashlib
import random
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, cast

import polars as pl

from llmango.inputs import InputRequest, ResolvedInput, load_input_sources
from llmango.schemas import LLMResponse
from llmango.spec import OTHER_CATEGORY, ExperimentSpec

FOLDER = "001_fruit"
QUESTIONS = ("001a", "001b", "001c", "001d")
FRUIT_LIST = "fruit_list"

_ORDER_FIXED = "fixed"
_ORDER_SHUFFLE = "shuffle"


class FruitChoice(LLMResponse):
    fruit: str


class WyborOwocu(LLMResponse):
    """The Polish parallel of FruitChoice, used only by the 001d pl arm."""

    owoc: str


class FruitEnum(StrEnum):
    """Canonical fruit categories, closed by 'other' for anything unnamed here."""

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
    OTHER = OTHER_CATEGORY


class FruitNormalization(LLMResponse):
    """A raw fruit answer mapped to a canonical category."""

    canonical: FruitEnum
    is_valid: bool
    multiple: bool


_QUALIFIERS = {"a", "an", "the", "fresh", "ripe"}


def preprocess(text: str) -> str:
    """Drop leading articles and qualifiers so answers match the mapping table."""
    tokens = [token for token in text.split() if token not in _QUALIFIERS]
    return " ".join(tokens)


def build_input(request: InputRequest) -> ResolvedInput:
    """Render one sample's fruit list: localized labels shown, canonical ids kept."""
    if request.name != FRUIT_LIST:
        raise ValueError(f"{FOLDER} has no prompt input {request.name!r}.")
    table = _table(request.data)
    shown = _shown_order(table, request.declaration, request.sample_idx)
    labels = ", ".join(_label(table, canonical, request.lang) for canonical in shown)
    return ResolvedInput(text=labels, value=shown)


def mapping_seed() -> dict[str, str]:
    """Map every question's fruit labels, in every language, onto canonical ids."""
    mapping: dict[str, str] = {}
    for question_id in [None, *QUESTIONS]:
        sources = load_input_sources(FOLDER, question_id, [FRUIT_LIST])
        for canonical, labels in _table(sources[FRUIT_LIST].data).items():
            for label in labels.values():
                mapping[label] = canonical
    return mapping


def extra_normalized_columns(frame: pl.DataFrame) -> dict[str, pl.Series]:
    """Add chosen_position: where the picked fruit sat in the list that row saw."""
    shown = (
        frame.get_column("prompt_inputs")
        .str.json_decode(pl.Struct({FRUIT_LIST: pl.List(pl.String())}))
        .struct.field(FRUIT_LIST)
        .to_list()
    )
    canonicals = frame.get_column("canonical").to_list()
    positions = [
        _position(order, canonical)
        for order, canonical in zip(shown, canonicals, strict=True)
    ]
    return {"chosen_position": pl.Series(positions, dtype=pl.Int64())}


def _position(order: list[str] | None, canonical: str | None) -> int | None:
    """Return the 1-based place of one canonical answer among the fruits shown."""
    if order is None or canonical is None or canonical not in order:
        return None
    return order.index(canonical) + 1


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
) -> list[str]:
    """Return the canonical ids in the order one sample shows them."""
    order = declaration.get("order")
    if order == _ORDER_SHUFFLE:
        return _shuffled(list(table), sample_idx)
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


def _shuffled(ids: list[str], sample_idx: int) -> list[str]:
    """Deterministically shuffle ids from the sample index alone."""
    key = str(sample_idx).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return shuffled


FRUIT = ExperimentSpec(
    folder=FOLDER,
    questions=QUESTIONS,
    schemas=(FruitChoice, WyborOwocu),
    normalization_schema=FruitNormalization,
    preprocess=preprocess,
    build_input=build_input,
    mapping_seed=mapping_seed,
    extra_normalized_columns=extra_normalized_columns,
)
