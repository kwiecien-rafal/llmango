"""Experiment 001: how a model picks one fruit, and how language shifts the pick."""

import hashlib
import random
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml

from llmango.config import get_experiment_dir, get_question_dir
from llmango.inputs import InputRequest, ResolvedInput, load_input_sources
from llmango.schemas import LLMResponse
from llmango.spec import OTHER_CATEGORY, ExperimentSpec, NormalizationMap

FOLDER = "e001_fruit"
QUESTIONS = ("001a", "001b", "001c", "001d")
FRUIT_LIST = "fruit_list"
_ORDER_FIXED = "fixed"
_ORDER_SHUFFLE = "shuffle"


# ----------------------------------------
# Main run path
# ----------------------------------------


class FruitChoice(LLMResponse):
    fruit: str


# 001d's PL schema. No docstring because it's also passed to the LLM.
class WyborOwocu(LLMResponse):
    owoc: str


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


def build_input(request: InputRequest) -> ResolvedInput:
    """Render one sample's fruit list: localized labels shown, canonical ids kept."""
    if request.name != FRUIT_LIST:
        raise ValueError(f"{FOLDER} has no prompt input {request.name!r}.")
    table = _table(request.data)
    shown = _shown_order(table, request.declaration, request.sample_idx)
    labels = ", ".join(_label(table, canonical, request.lang) for canonical in shown)
    return ResolvedInput(text=labels, value=shown)


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


# ----------------------------------------
# Normalization
# ----------------------------------------


_NORMALIZATION_MAP = Path(__file__).parent / "normalization_map.yaml"
_MAP_HEADER = (
    "# Answer -> canonical fruit, null when the answer named none. Hand-written\n"
    "# spellings the fruit labels miss, plus every answer normalize has paid for.\n"
)
_QUALIFIERS = {"a", "an", "the", "fresh", "ripe"}


class FruitEnum(StrEnum):
    """Canonical 10 fruits and 'other'."""

    APPLE = "apple"
    BANANA = "banana"
    ORANGE = "orange"
    MANGO = "mango"
    STRAWBERRY = "strawberry"
    GRAPE = "grape"
    WATERMELON = "watermelon"
    PINEAPPLE = "pineapple"
    POMEGRANATE = "pomegranate"
    LYCHEE = "lychee"
    OTHER = OTHER_CATEGORY


class FruitNormalization(LLMResponse):
    """A raw fruit answer mapped to a canonical category."""

    canonical: FruitEnum
    is_valid: bool


def preprocess(text: str) -> str:
    """Drop leading articles and qualifiers so answers match the mapping table."""
    tokens = [token for token in text.split() if token not in _QUALIFIERS]
    return " ".join(tokens)


def normalization_map() -> NormalizationMap:
    """Map the fruit labels of every question, plus the stored answers, onto ids."""
    question_dirs = [
        get_experiment_dir(FOLDER),
        *(get_question_dir(FOLDER, question_id) for question_id in QUESTIONS),
    ]
    mapping: NormalizationMap = {}
    for question_dir in question_dirs:
        source = load_input_sources([question_dir], [FRUIT_LIST])[FRUIT_LIST]
        if source.data is None:
            continue
        for canonical, labels in _table(source.data).items():
            for label in labels.values():
                mapping[label] = canonical
    return mapping | _stored_map()


def promote_normalizations(entries: NormalizationMap) -> None:
    """Add what the LLM decided to the committed map, so it is never paid for twice."""
    stored = _stored_map() | entries
    body = yaml.safe_dump(stored, allow_unicode=True, sort_keys=True)
    _NORMALIZATION_MAP.write_text(_MAP_HEADER + body, encoding="utf-8")


def _stored_map() -> NormalizationMap:
    """Read the committed map, where a null value means the answer named no fruit."""
    text = _NORMALIZATION_MAP.read_text(encoding="utf-8")
    entries = cast(dict[Any, Any], yaml.safe_load(text) or {})
    return {
        str(answer): None if canonical is None else str(canonical)
        for answer, canonical in entries.items()
    }


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


# ----------------------------------------
# FRUIT spec
# ----------------------------------------


FRUIT = ExperimentSpec(
    folder=FOLDER,
    questions=QUESTIONS,
    schemas=(FruitChoice, WyborOwocu),
    normalization_schema=FruitNormalization,
    preprocess=preprocess,
    build_input=build_input,
    normalization_map=normalization_map,
    promote_normalizations=promote_normalizations,
    extra_normalized_columns=extra_normalized_columns,
)
