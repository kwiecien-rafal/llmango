"""Post-hoc normalization of free-text answers to canonical categories.

Runs per experiment: reads the raw answers of every question in the experiment
and maps each onto a canonical English category in layers, cheapest first. The
shared fruit table seeds the deterministic mapping so every in-list answer
resolves for free; leftover strings (off-list or free-text) fall through to the
mapping file and then an LLM. Raw answers are never overwritten. Normalization
only adds the canonical, is_fruit, multiple and chosen_position columns and
writes a separate normalized Parquet file. Every LLM result is cached and
promoted, so reruns never pay for the same string twice.

Position is resolved here rather than at generation time because it takes a
canonical answer to locate: the raw answer is free text in the prompt's language,
while option_order records which canonical ids were shown and in what order.
"""

import json
import string
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict

from llmango.backends.base import GenerationBackend, GenRequest
from llmango.config import NORMALIZATION_DIR
from llmango.questions import (
    SamplingParams,
    experiment_dir,
    list_questions,
    load_experiment_config,
    load_fruits,
)
from llmango.registry import (
    OTHER_CATEGORY,
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
    """The canonical category a raw answer resolves to."""

    model_config = ConfigDict(frozen=True)

    canonical: str
    is_fruit: bool
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

    directory = NORMALIZATION_DIR / experiment_id
    mapping = _load_mapping(directory, spec, experiment_id)
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
                experiment_id,
                normalization_schema,
                make_backend,
                model,
                max_llm_calls,
                cache,
            )
        )
        _save_cache(directory, cache)

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
        return Resolution(canonical="", is_fruit=False, multiple=False)
    canonical = mapping.get(preprocess(raw, spec))
    if canonical is not None:
        return Resolution(canonical=canonical, is_fruit=True, multiple=False)
    cached = cache.get(lang, {}).get(raw)
    if cached is not None:
        return Resolution.model_validate(cached)
    return None


def _resolve_online(
    unresolved: list[tuple[str, str]],
    experiment_id: str,
    response_schema: type[BaseModel],
    make_backend: Callable[[], GenerationBackend] | None,
    model: str | None,
    max_llm_calls: int | None,
    cache: dict[str, dict[str, dict[str, object]]],
) -> dict[tuple[str, str], Resolution]:
    """Guard cost, build the backend lazily, and resolve the leftover answers."""
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
            resolved[(lang, raw)] = Resolution(
                canonical=OTHER_CATEGORY, is_fruit=True, multiple=False
            )
            continue
        resolution = Resolution.model_validate(result.parsed.model_dump(mode="json"))
        resolved[(lang, raw)] = resolution
        cache.setdefault(lang, {})[raw] = resolution.model_dump()
    return resolved


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
            "is_fruit": resolution.is_fruit,
            "multiple": resolution.multiple,
        }
        for (lang, raw), resolution in resolutions.items()
    ]
    schema: dict[str, pl.DataType] = {
        "lang": pl.String(),
        spec.raw_column: pl.String(),
        spec.canonical_column: pl.String(),
        "is_fruit": pl.Boolean(),
        "multiple": pl.Boolean(),
    }
    resolution_frame = pl.DataFrame(rows, schema_overrides=schema)
    joined = frame.join(resolution_frame, on=["lang", spec.raw_column], how="left")
    return _order_columns(joined.with_columns(_positions(joined, spec)), spec)


def _position(order: list[str] | None, canonical: str | None) -> int | None:
    """Return the 1-based place of one canonical answer among the options shown.

    Null whenever the answer is not one of them: a refusal, an 'other' answer, or
    a category that exists but was not on this sample's list.
    """
    if order is None or canonical is None or canonical not in order:
        return None
    return order.index(canonical) + 1


def _positions(frame: pl.DataFrame, spec: ExperimentSpec) -> pl.Series:
    """Locate every row's canonical answer in the option order it was shown."""
    shown = frame.get_column("option_order").str.json_decode(pl.List(pl.String()))
    canonicals = frame.get_column(spec.canonical_column).to_list()
    positions = [
        _position(order, canonical)
        for order, canonical in zip(shown.to_list(), canonicals, strict=True)
    ]
    return pl.Series("chosen_position", positions, dtype=pl.Int64())


def _order_columns(frame: pl.DataFrame, spec: ExperimentSpec) -> pl.DataFrame:
    """Move the added columns to sit directly after the raw answer they describe."""
    added = [spec.canonical_column, "is_fruit", "multiple", "chosen_position"]
    kept = [column for column in frame.columns if column not in added]
    cut = kept.index(spec.raw_column) + 1
    return frame.select(kept[:cut] + added + kept[cut:])


def _normalize_model(experiment_id: str) -> str | None:
    """Return the configured normalization model, falling back to the run model."""
    config = load_experiment_config(experiment_id)
    return config.normalize_model or config.model


def _load_mapping(
    directory: Path, spec: ExperimentSpec, experiment_id: str
) -> dict[str, str]:
    """Load the deterministic mapping, seeded by the fruit table labels.

    The fruit table's per-language labels resolve every in-list answer for free;
    mapping.yaml adds or overrides entries for anything else.
    """
    mapping = _fruit_label_mapping(experiment_id, spec)
    path = directory / _MAPPING_FILE
    if path.is_file():
        raw_map: dict[str, str] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mapping.update((preprocess(key, spec), value) for key, value in raw_map.items())
    if spec.canonical_values is not None:
        invalid = sorted(set(mapping.values()) - spec.canonical_values)
        if invalid:
            raise ValueError(
                f"mapping has values outside the canonical set: {', '.join(invalid)}"
            )
    return mapping


def _fruit_label_mapping(experiment_id: str, spec: ExperimentSpec) -> dict[str, str]:
    """Build a label-to-canonical mapping from the experiment's fruit table."""
    table = load_fruits(experiment_id)
    return {
        preprocess(label, spec): canonical
        for canonical, labels in table.labels.items()
        for label in labels.values()
    }


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
