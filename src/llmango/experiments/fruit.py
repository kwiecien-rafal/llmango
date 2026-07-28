"""Experiment 001: fruit.

Measures how random a model is when asked to pick a fruit from a fixed list,
how the prompt language shifts that pick, and how requesting a structured
response in a non-target language shifts it. The raw pick is preserved and
normalized post-hoc to a canonical English category.

QUESTIONS names the four questions this experiment owns. The spec declares them
so a question id resolves to a spec, and the mapping seed reads the same constant
to cover every fruit list any of them shows.

The one prompt input is fruit_list, whose data file holds the canonical ids with
their per-language labels. A question declares how that list is arranged, either
a fixed permutation or a per-sample shuffle. Both live here rather than in the
engine, because the arrangement is what this experiment varies; a later
experiment arranges its own inputs its own way without touching shared code.

001 appends one column of its own, chosen_position, for the same reason: only
this experiment knows its prompt input is an ordered list a pick can sit inside.
It adds nothing to the raw parquet, since the answer is already a core column.
"""

import hashlib
import random
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, cast

import polars as pl

from llmango.inputs import InputRequest, ResolvedInput, load_input_sources
from llmango.schemas import LLMResponse
from llmango.spec import ExperimentSpec

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
    is_valid: bool
    multiple: bool


_QUALIFIERS = {"a", "an", "the", "fresh", "ripe"}


def preprocess(text: str) -> str:
    """Drop leading articles and qualifiers so answers match the mapping table."""
    tokens = [token for token in text.split() if token not in _QUALIFIERS]
    return " ".join(tokens)


def build_input(request: InputRequest) -> ResolvedInput:
    """Render the fruit list for one sample in one language.

    Returns the localized labels as the model sees them, and the canonical ids in
    the same order as the value recorded per row, which is what lets chosen
    position be resolved later from a free-text answer.
    """
    if request.name != FRUIT_LIST:
        raise ValueError(f"{FOLDER} has no prompt input {request.name!r}.")
    table = _table(request.data)
    shown = _shown_order(table, request.declaration, request.sample_idx, request.seed)
    labels = ", ".join(_label(table, canonical, request.lang) for canonical in shown)
    return ResolvedInput(text=labels, value=shown)


def mapping_seed() -> dict[str, str]:
    """Map every fruit label in every language onto its canonical id.

    Seeds normalization so an answer that named one of the listed fruits resolves
    without an LLM call, whichever language it was asked in. The experiment's own
    fruit list is the base and every question is read on top of it, since a
    question that overrides the list with its own file shows labels the experiment
    file does not have.
    """
    mapping: dict[str, str] = {}
    for question_id in [None, *QUESTIONS]:
        sources = load_input_sources(FOLDER, question_id, [FRUIT_LIST])
        for canonical, labels in _table(sources[FRUIT_LIST].data).items():
            for label in labels.values():
                mapping[label] = canonical
    return mapping


def extra_normalized_columns(frame: pl.DataFrame) -> dict[str, pl.Series]:
    """Add chosen_position: where the picked fruit sat in the list that row saw.

    Only this experiment knows fruit_list resolves to an ordered list of canonical
    ids, so the position of an answer within it is 001's own column rather than a
    pipeline concept. It is computed after normalization because it takes a
    canonical answer to locate, while the answer itself is free text in the
    prompt's language.
    """
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
    """Return the 1-based place of one canonical answer among the fruits shown.

    Null whenever the answer is not one of them: a refusal, an 'other' answer, or
    a fruit that exists but was not on this sample's list.
    """
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
