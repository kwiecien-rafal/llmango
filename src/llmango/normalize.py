"""Post-hoc normalization of free-text answers to canonical categories.

Runs per experiment: reads the raw answers of every question in the experiment
and maps each onto a canonical English category in layers, cheapest first. The
experiment's own mapping seed resolves every answer it already has a label for;
leftover strings (off-list or free-text) fall through to the mapping file and
then an LLM. Raw answers are never overwritten. Normalization
only adds the canonical, validity, multiple and chosen_position columns and
writes a separate normalized Parquet file. Every LLM result is cached and
promoted, so reruns never pay for the same string twice.

The engine calls the validity flag is_valid throughout; the column it lands in
and the field the model fills are both named by the experiment's valid_column,
so an experiment keeps its own wording without the engine adopting it.

Position is resolved here rather than at generation time because it takes a
canonical answer to locate: the raw answer is free text in the prompt's language,
while prompt_inputs records what the question's inputs resolved to for that row.
"""

import json
import string
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import get_args

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict

from llmango.backends.base import GenerationBackend, GenRequest
from llmango.config import MAPPINGS_DIR, experiment_dir
from llmango.questions import (
    SamplingParams,
    list_questions,
    load_experiment_config,
)
from llmango.registry import (
    ExperimentSpec,
    get_experiment,
    resolve_experiment_id,
)
from llmango.storage import read_results, write_normalized

_MAPPING_FILE = "mapping.yaml"
_CACHE_FILE = "normalization_cache.json"
_PROMPT_FILE = "normalize.md"

_PUNCTUATION = string.punctuation + "«»„“”‘’¿¡… "


class Resolution(BaseModel):
    """The canonical category a raw answer resolves to.

    is_valid is the engine's name for the flag saying the answer named a category
    at all. An experiment spells the same flag its own way, so a resolution is
    read from and written back to the experiment's field names.
    """

    model_config = ConfigDict(frozen=True)

    canonical: str
    is_valid: bool
    multiple: bool


@dataclass(frozen=True)
class NormalizeOutcome:
    """What one normalization run produced, or a dry run would produce.

    parquet_path is None on a dry run, which resolves nothing and writes nothing.
    """

    parquet_path: Path | None
    rows: int
    distinct: int
    llm_calls: int


def preprocess(raw: str, spec: ExperimentSpec) -> str:
    """Normalize a raw answer for matching: NFKC, lowercase, strip punctuation."""
    text = unicodedata.normalize("NFKC", raw).lower().strip(_PUNCTUATION)
    if spec.preprocess is not None:
        text = spec.preprocess(text)
    return text


def _read_resolution(fields: Mapping[str, object], spec: ExperimentSpec) -> Resolution:
    """Read a resolution from fields keyed the experiment's way.

    Both the model's normalization response and the cache on disk spell the
    validity flag with the experiment's valid_column, so it is renamed to the
    engine's is_valid before validation. Any other field, such as the echoed raw
    answer, is ignored.
    """
    payload = dict(fields)
    payload["is_valid"] = payload.get(spec.valid_column)
    return Resolution.model_validate(payload)


def _write_resolution(
    resolution: Resolution, spec: ExperimentSpec
) -> dict[str, object]:
    """Write a resolution back out under the experiment's own field names."""
    return {
        "canonical": resolution.canonical,
        spec.valid_column: resolution.is_valid,
        "multiple": resolution.multiple,
    }


def normalize_experiment(
    experiment_id: str,
    *,
    make_backend: Callable[[], GenerationBackend] | None = None,
    model: str | None = None,
    max_llm_calls: int | None = None,
    dry_run: bool = False,
) -> NormalizeOutcome:
    """Add canonical categories to an experiment's raw answers and write them out.

    Reads every question's raw results, resolves each distinct answer per language
    through the deterministic layers and then the LLM for the rest, and writes a
    normalized Parquet file that leaves the raw answers untouched. The backend is
    built lazily, so a run resolved entirely offline needs no API key. A dry run
    stops after the offline layers and reports how many answers the LLM would
    resolve, without calling it or writing anything.
    """
    experiment_id = resolve_experiment_id(experiment_id)
    spec = get_experiment(experiment_id)
    normalization_schema = spec.normalization_schema
    if normalization_schema is None:
        raise ValueError(f"Experiment {experiment_id} has no normalization schema.")

    frame = _read_experiment_raw(experiment_id)

    directory = MAPPINGS_DIR / experiment_id
    mapping = _load_mapping(directory, spec)
    cache = _load_cache(directory)
    pairs = _distinct_pairs(frame, spec)

    resolutions: dict[tuple[str, str], Resolution] = {}
    unresolved: list[tuple[str, str]] = []
    for lang, raw in pairs:
        offline = _resolve_offline(lang, raw, spec, mapping, cache)
        if offline is not None:
            resolutions[(lang, raw)] = offline
        else:
            unresolved.append((lang, raw))

    if dry_run:
        return NormalizeOutcome(
            parquet_path=None,
            rows=frame.height,
            distinct=len(pairs),
            llm_calls=len(unresolved),
        )

    if unresolved:
        resolutions.update(
            _resolve_online(
                unresolved,
                spec,
                normalization_schema,
                make_backend,
                model,
                max_llm_calls,
                cache,
            )
        )
        _save_cache(directory, cache)
        _require_all_resolved(unresolved, resolutions)

    normalized = _join_resolutions(frame, resolutions, spec)
    parquet_path = write_normalized(normalized, experiment_id)
    return NormalizeOutcome(
        parquet_path=parquet_path,
        rows=frame.height,
        distinct=len(pairs),
        llm_calls=len(unresolved),
    )


def _read_experiment_raw(experiment_id: str) -> pl.DataFrame:
    """Read and concatenate every question's raw results for an experiment."""
    frames = [
        frame
        for question_id in list_questions(experiment_id)
        if not (frame := read_results(f"{question_id}__*.parquet")).is_empty()
    ]
    if not frames:
        raise FileNotFoundError(f"No raw results to normalize for {experiment_id}.")
    return pl.concat(frames)


def _distinct_pairs(frame: pl.DataFrame, spec: ExperimentSpec) -> list[tuple[str, str]]:
    """Return the sorted, deduped (lang, raw answer) pairs from the raw frame."""
    langs = frame.get_column("lang").to_list()
    raws = frame.get_column(spec.raw_column).to_list()
    return sorted(
        {(str(lang), str(raw)) for lang, raw in zip(langs, raws, strict=True)}
    )


def _resolve_offline(
    lang: str,
    raw: str,
    spec: ExperimentSpec,
    mapping: dict[str, str],
    cache: dict[str, dict[str, dict[str, object]]],
) -> Resolution | None:
    """Resolve a raw answer without an LLM: refusal, mapping table, then cache."""
    if not raw.strip():
        return Resolution(canonical="", is_valid=False, multiple=False)
    canonical = mapping.get(preprocess(raw, spec))
    if canonical is not None:
        return Resolution(canonical=canonical, is_valid=True, multiple=False)
    cached = cache.get(lang, {}).get(raw)
    if cached is not None:
        return _read_resolution(cached, spec)
    return None


def _resolve_online(
    unresolved: list[tuple[str, str]],
    spec: ExperimentSpec,
    response_schema: type[BaseModel],
    make_backend: Callable[[], GenerationBackend] | None,
    model: str | None,
    max_llm_calls: int | None,
    cache: dict[str, dict[str, dict[str, object]]],
) -> dict[tuple[str, str], Resolution]:
    """Guard cost, build the backend lazily, and resolve the leftover answers.

    Only a parsed response becomes a resolution. Anything else is left out, so the
    caller can save what was paid for and then report what is missing.
    """
    experiment_id = spec.experiment_id
    if max_llm_calls is not None and len(unresolved) > max_llm_calls:
        raise ValueError(
            f"{len(unresolved)} answers need the LLM layer, above the smoke limit "
            f"of {max_llm_calls}. Re-run with --force to allow the paid calls."
        )
    if make_backend is None:
        raise ValueError(
            f"{len(unresolved)} answers need the LLM layer but no backend given."
        )
    resolved_model = model or _normalize_model(experiment_id)
    if not resolved_model:
        raise ValueError(f"No model given to normalize {experiment_id}.")

    template = _load_prompt(experiment_id)
    requests = [
        GenRequest(
            question_id=experiment_id,
            lang=lang,
            model=resolved_model,
            prompt=template.replace("{lang}", lang).replace("{raw}", raw),
            prompt_sha256="",
            sample_idx=index,
            seed=None,
            sampling=SamplingParams(temperature=0.0),
            response_schema=response_schema,
        )
        for index, (lang, raw) in enumerate(unresolved)
    ]
    results = make_backend().generate_many(requests)

    resolved: dict[tuple[str, str], Resolution] = {}
    for (lang, raw), result in zip(unresolved, results, strict=True):
        if result.parsed is None:
            continue
        resolution = _read_resolution(result.parsed.model_dump(mode="json"), spec)
        resolved[(lang, raw)] = resolution
        cache.setdefault(lang, {})[raw] = _write_resolution(resolution, spec)
    return resolved


def _require_all_resolved(
    unresolved: list[tuple[str, str]],
    resolutions: dict[tuple[str, str], Resolution],
) -> None:
    """Fail rather than let a call that answered nothing become a category.

    A call that came back with no parsed response, whether it errored, was refused
    or was cut short, carries no answer, so writing it out under any category
    would put a transport failure into the distributions. Everything that did come
    back is cached by the time this runs, so a rerun pays only for these again.
    """
    failed = [pair for pair in unresolved if pair not in resolutions]
    if not failed:
        return
    preview = ", ".join(f"{lang}:{raw!r}" for lang, raw in failed[:3])
    raise ValueError(
        f"{len(failed)} of {len(unresolved)} answers came back unparsed and were "
        f"not written: {preview}. Everything else is cached, so a rerun retries "
        f"only these."
    )


def _join_resolutions(
    frame: pl.DataFrame,
    resolutions: dict[tuple[str, str], Resolution],
    spec: ExperimentSpec,
) -> pl.DataFrame:
    """Attach the canonical columns to every raw row via its (lang, answer).

    An answer that resolved to no category at all, such as a refusal, is written
    as a null rather than an empty string, so the column never carries a category
    that does not exist.
    """
    rows = [
        {
            "lang": lang,
            spec.raw_column: raw,
            spec.canonical_column: resolution.canonical or None,
            spec.valid_column: resolution.is_valid,
            "multiple": resolution.multiple,
        }
        for (lang, raw), resolution in resolutions.items()
    ]
    schema: dict[str, pl.DataType] = {
        "lang": pl.String(),
        spec.raw_column: pl.String(),
        spec.canonical_column: pl.String(),
        spec.valid_column: pl.Boolean(),
        "multiple": pl.Boolean(),
    }
    resolution_frame = pl.DataFrame(rows, schema_overrides=schema)
    joined = frame.join(resolution_frame, on=["lang", spec.raw_column], how="left")
    return _order_columns(joined.with_columns(_positions(joined, spec)), spec)


def _position(order: list[str] | None, canonical: str | None) -> int | None:
    """Return the 1-based place of one canonical answer among the values shown.

    Null whenever the answer is not one of them: a refusal, an 'other' answer, or
    a category that exists but was not on this sample's list.
    """
    if order is None or canonical is None or canonical not in order:
        return None
    return order.index(canonical) + 1


def _positions(frame: pl.DataFrame, spec: ExperimentSpec) -> pl.Series:
    """Locate every row's canonical answer in the input it was shown.

    Empty when the experiment names no positional input, which is any experiment
    whose prompt does not present an ordered list to choose from.
    """
    if spec.position_input is None:
        return pl.Series("chosen_position", [None] * frame.height, dtype=pl.Int64())
    shown = (
        frame.get_column("prompt_inputs")
        .str.json_decode(pl.Struct({spec.position_input: pl.List(pl.String())}))
        .struct.field(spec.position_input)
        .to_list()
    )
    canonicals = frame.get_column(spec.canonical_column).to_list()
    positions = [
        _position(order, canonical)
        for order, canonical in zip(shown, canonicals, strict=True)
    ]
    return pl.Series("chosen_position", positions, dtype=pl.Int64())


def _order_columns(frame: pl.DataFrame, spec: ExperimentSpec) -> pl.DataFrame:
    """Move the added columns to sit directly after the raw answer they describe."""
    added = [spec.canonical_column, spec.valid_column, "multiple", "chosen_position"]
    kept = [column for column in frame.columns if column not in added]
    cut = kept.index(spec.raw_column) + 1
    return frame.select(kept[:cut] + added + kept[cut:])


def _normalize_model(experiment_id: str) -> str | None:
    """Return the configured normalization model, falling back to the run model."""
    config = load_experiment_config(experiment_id)
    return config.normalize_model or config.model


def _load_mapping(directory: Path, spec: ExperimentSpec) -> dict[str, str]:
    """Load the deterministic mapping, seeded by the experiment's own labels.

    The seed resolves every answer the experiment already knows a label for, in
    any language; mapping.yaml adds or overrides entries for anything else.
    """
    mapping = _seed_mapping(spec)
    path = directory / _MAPPING_FILE
    if path.is_file():
        raw_map: dict[str, str] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mapping.update((preprocess(key, spec), value) for key, value in raw_map.items())
    canonical_values = _canonical_values(spec)
    if canonical_values is not None:
        invalid = sorted(set(mapping.values()) - canonical_values)
        if invalid:
            raise ValueError(
                f"mapping has values outside the canonical set: {', '.join(invalid)}"
            )
    return mapping


def _canonical_values(spec: ExperimentSpec) -> frozenset[str] | None:
    """Read the closed category set off the normalization schema's canonical field.

    The schema the LLM layer fills already declares every category it may return,
    as an enum, a literal or a union of both. Reading the set from that annotation
    keeps the mapping file checked against the one declaration the model sees,
    rather than a second list that could drift away from it.
    """
    if spec.normalization_schema is None:
        return None
    annotation = spec.normalization_schema.model_fields["canonical"].annotation
    values: set[str] = set()
    for member in get_args(annotation) or (annotation,):
        if isinstance(member, type) and issubclass(member, Enum):
            values.update(str(entry.value) for entry in member)
        else:
            values.update(str(literal) for literal in get_args(member))
    return frozenset(values)


def _seed_mapping(spec: ExperimentSpec) -> dict[str, str]:
    """Preprocess the experiment's label-to-canonical seed, empty when it has none.

    The experiment is given its question ids, because a question may override an
    input with its own data file while normalization spans every question at once.
    """
    if spec.mapping_seed is None:
        return {}
    seed = spec.mapping_seed(list_questions(spec.experiment_id))
    return {preprocess(label, spec): canonical for label, canonical in seed.items()}


def _load_cache(directory: Path) -> dict[str, dict[str, dict[str, object]]]:
    """Load the promoted LLM results, nested as {lang: {raw: fields}}."""
    path = directory / _CACHE_FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(
    directory: Path, cache: dict[str, dict[str, dict[str, object]]]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _CACHE_FILE).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_prompt(experiment_id: str) -> str:
    path = experiment_dir(experiment_id) / _PROMPT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization prompt: {path}")
    return path.read_text(encoding="utf-8")
