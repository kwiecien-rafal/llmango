"""Post-hoc normalization of free-text answers to canonical categories.

Maps each raw answer onto a canonical English category in layers, cheapest
first: dedupe the distinct answers per language, resolve them against a
deterministic mapping table, then fall back to an LLM for whatever is left. Raw
answers are never overwritten. Normalization only adds the canonical, is_fruit
and multiple columns and writes a separate normalized Parquet file. Every LLM
result is cached and promoted, so reruns never pay for the same string twice.
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
from llmango.questions import SamplingParams, experiment_dir, load_question
from llmango.registry import ExperimentSpec, get_experiment
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
    """What one normalization run produced."""

    parquet_path: Path
    rows: int
    distinct: int
    llm_calls: int


def preprocess(raw: str, spec: ExperimentSpec) -> str:
    """Normalize a raw answer for matching: NFKC, lowercase, strip punctuation."""
    text = unicodedata.normalize("NFKC", raw).lower().strip(_PUNCTUATION)
    if spec.preprocess is not None:
        text = spec.preprocess(text)
    return text


def normalize_question(
    question_id: str,
    *,
    make_backend: Callable[[], GenerationBackend] | None = None,
    model: str | None = None,
    max_llm_calls: int | None = None,
) -> NormalizeOutcome:
    """Add canonical categories to a question's raw answers and write them out.

    Reads every raw result for the question, resolves each distinct answer per
    language through the deterministic layers and then the LLM for the rest, and
    writes a normalized Parquet file that leaves the raw answers untouched. The
    backend is built lazily, so a run resolved entirely offline needs no API key.
    """
    from llmango.experiments import ensure_registered

    ensure_registered()
    spec = get_experiment(question_id)
    normalization_schema = spec.normalization_schema
    if normalization_schema is None:
        raise ValueError(f"Experiment {question_id} has no normalization schema.")

    frame = read_results(f"{question_id}__*.parquet")
    if frame.is_empty():
        raise FileNotFoundError(f"No raw results to normalize for {question_id}.")

    directory = NORMALIZATION_DIR / (spec.slug or question_id)
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

    if unresolved:
        resolutions.update(
            _resolve_online(
                unresolved,
                question_id,
                normalization_schema,
                make_backend,
                model,
                max_llm_calls,
                cache,
            )
        )
        _save_cache(directory, cache)

    normalized = _join_resolutions(frame, resolutions, spec)
    parquet_path = write_normalized(normalized, question_id)
    return NormalizeOutcome(
        parquet_path=parquet_path,
        rows=frame.height,
        distinct=len(pairs),
        llm_calls=len(unresolved),
    )


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
    question_id: str,
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
    resolved_model = model or _normalize_model(question_id)
    if not resolved_model:
        raise ValueError(f"No model given to normalize {question_id}.")

    template = _load_prompt(question_id)
    requests = [
        GenRequest(
            question_id=question_id,
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
                canonical="other", is_fruit=True, multiple=False
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
    """Attach the canonical columns to every raw row via its (lang, answer)."""
    rows = [
        {
            "lang": lang,
            spec.raw_column: raw,
            spec.canonical_column: resolution.canonical,
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
    return frame.join(resolution_frame, on=["lang", spec.raw_column], how="left")


def _normalize_model(question_id: str) -> str | None:
    """Return the configured normalization model, falling back to the run model."""
    config = load_question(question_id)
    return config.normalize_model or config.model


def _load_mapping(directory: Path, spec: ExperimentSpec) -> dict[str, str]:
    """Load the deterministic mapping table, keyed by preprocessed answer."""
    path = directory / _MAPPING_FILE
    if not path.is_file():
        return {}
    raw_map: dict[str, str] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping = {preprocess(key, spec): value for key, value in raw_map.items()}
    if spec.canonical_values is not None:
        invalid = sorted(set(mapping.values()) - spec.canonical_values)
        if invalid:
            raise ValueError(
                f"mapping.yaml has values outside the canonical set: "
                f"{', '.join(invalid)}"
            )
    return mapping


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


def _load_prompt(question_id: str) -> str:
    path = experiment_dir(question_id) / _PROMPT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization prompt: {path}")
    return path.read_text(encoding="utf-8")
