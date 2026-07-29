"""Map one question's raw answers onto canonical categories, cheapest layer first."""

import json
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict

from llmango.backends import backend_for
from llmango.backends.base import Backend, GenRequest
from llmango.config import MAPPINGS_DIR, experiment_dir
from llmango.experiments import spec_for
from llmango.pricing import guard_cost
from llmango.questions import load_experiment_config
from llmango.spec import ExperimentSpec, canonical_values
from llmango.storage import read_results, write_normalized

_MAPPING_FILE = "mapping.yaml"
_CACHE_FILE = "normalization_cache.json"
_PROMPT_FILE = "normalize.md"

_STRIP_CHARS = string.punctuation + string.whitespace + "«»„“”‘’¿¡。、「」『』"
_RESOLUTION_COLUMNS = ("canonical", "is_valid")
_ANSWERED = pl.col("error").is_null()


class Resolution(BaseModel):
    """The canonical category a raw answer resolves to, None when it names none."""

    model_config = ConfigDict(frozen=True)

    canonical: str | None
    is_valid: bool


type Cache = dict[str, dict[str, Resolution]]


@dataclass(frozen=True)
class NormalizeOutcome:
    """What one normalization produced; parquet_path is None on a dry run."""

    parquet_path: Path | None
    rows: int
    distinct: int
    llm_calls: int


def normalize_question(
    question_id: str,
    *,
    backend: Backend | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> NormalizeOutcome:
    """Resolve a question's raw answers to canonical categories and write them out."""
    spec = spec_for(question_id)
    schema = spec.normalization_schema
    if schema is None:
        raise ValueError(f"Experiment {spec.folder} has no normalization schema.")

    frame = read_results(f"{question_id}__*.parquet")
    if frame.is_empty():
        raise FileNotFoundError(
            f"No raw data for question {question_id} to normalize from. "
        )

    directory = MAPPINGS_DIR / spec.folder
    mapping = _load_mapping(directory, spec, schema)
    cache = _load_cache(directory)
    pairs = _distinct_pairs(frame)

    resolutions: dict[tuple[str, str], Resolution] = {}
    unresolved: list[tuple[str, str]] = []
    for lang, answer in pairs:
        offline = _resolve_offline(lang, answer, spec, mapping, cache)
        if offline is not None:
            resolutions[(lang, answer)] = offline
        else:
            unresolved.append((lang, answer))

    if dry_run:
        return NormalizeOutcome(
            parquet_path=None,
            rows=frame.height,
            distinct=len(pairs),
            llm_calls=len(unresolved),
        )

    if unresolved:
        guard_cost(len(unresolved), force)
        resolutions.update(_resolve_online(unresolved, spec, schema, backend, cache))
        _save_cache(directory, cache)
        _require_all_resolved(unresolved, resolutions)

    normalized = _join_resolutions(frame, resolutions, spec)
    return NormalizeOutcome(
        parquet_path=write_normalized(normalized, question_id),
        rows=frame.height,
        distinct=len(pairs),
        llm_calls=len(unresolved),
    )


def _preprocess(raw: str, spec: ExperimentSpec) -> str:
    """Normalize a raw answer for matching: NFKC, lowercase, strip edge characters."""
    text = unicodedata.normalize("NFKC", raw).lower().strip(_STRIP_CHARS)
    return spec.preprocess(text) if spec.preprocess is not None else text


def _distinct_pairs(frame: pl.DataFrame) -> list[tuple[str, str]]:
    """Return the sorted, deduped (lang, answer) pairs of every row that answered."""
    pairs = (
        frame.filter(_ANSWERED).select("lang", "answer").unique().sort("lang", "answer")
    )
    return [(str(lang), str(answer)) for lang, answer in pairs.iter_rows()]


def _resolve_offline(
    lang: str,
    answer: str,
    spec: ExperimentSpec,
    mapping: dict[str, str],
    cache: Cache,
) -> Resolution | None:
    """Resolve one answer without an LLM: refusal, mapping table, then cache."""
    if not answer.strip():
        return Resolution(canonical=None, is_valid=False)
    canonical = mapping.get(_preprocess(answer, spec))
    if canonical is not None:
        return Resolution(canonical=canonical, is_valid=True)
    return cache.get(lang, {}).get(answer)


def _resolve_online(
    unresolved: list[tuple[str, str]],
    spec: ExperimentSpec,
    schema: type[BaseModel],
    backend: Backend | None,
    cache: Cache,
) -> dict[tuple[str, str], Resolution]:
    """Ask the LLM for what no offline layer resolved, caching what parses."""
    config = load_experiment_config(spec.folder)
    template = _load_prompt(spec.folder)
    backend = backend or backend_for(config.normalize_provider)
    requests = [
        GenRequest(
            model=config.normalize_model,
            prompt=template.replace("{lang}", lang).replace("{raw}", answer),
            response_schema=schema,
            temperature=0.0,
        )
        for lang, answer in unresolved
    ]
    results = backend.generate_many(requests)

    resolved: dict[tuple[str, str], Resolution] = {}
    for (lang, answer), result in zip(unresolved, results, strict=True):
        if result.parsed is None:
            continue
        resolution = Resolution.model_validate(result.parsed, from_attributes=True)
        resolved[(lang, answer)] = resolution
        cache.setdefault(lang, {})[answer] = resolution
    return resolved


def _require_all_resolved(
    unresolved: list[tuple[str, str]],
    resolutions: dict[tuple[str, str], Resolution],
) -> None:
    """Fail rather than let a call that answered nothing become a category."""
    failed = [pair for pair in unresolved if pair not in resolutions]
    if not failed:
        return
    preview = ", ".join(f"{lang}:{answer!r}" for lang, answer in failed[:3])
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
    """Attach the canonical columns to every raw row whose call came back."""
    rows = [
        {
            "lang": lang,
            "answer": answer,
            "canonical": resolution.canonical,
            "is_valid": resolution.is_valid,
        }
        for (lang, answer), resolution in resolutions.items()
    ]
    schema: dict[str, pl.DataType] = {
        "lang": pl.String(),
        "answer": pl.String(),
        "canonical": pl.String(),
        "is_valid": pl.Boolean(),
    }
    joined = frame.join(
        pl.DataFrame(rows, schema=schema), on=["lang", "answer"], how="left"
    ).with_columns(
        **{
            column: pl.when(_ANSWERED).then(pl.col(column)).otherwise(None)
            for column in _RESOLUTION_COLUMNS
        }
    )
    add = spec.extra_normalized_columns
    extra = add(joined) if add is not None else {}
    return _order_columns(joined.with_columns(**extra), list(extra))


def _order_columns(frame: pl.DataFrame, extra: list[str]) -> pl.DataFrame:
    """Move the added columns to sit directly after the answer they describe."""
    added = [*_RESOLUTION_COLUMNS, *extra]
    kept = [column for column in frame.columns if column not in added]
    cut = kept.index("answer") + 1
    return frame.select(kept[:cut] + added + kept[cut:])


def _load_mapping(
    directory: Path, spec: ExperimentSpec, schema: type[BaseModel]
) -> dict[str, str]:
    """Load the deterministic mapping: the experiment's labels, then mapping.yaml."""
    mapping = _seed_mapping(spec)
    path = directory / _MAPPING_FILE
    if path.is_file():
        raw_map: dict[str, str] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mapping.update(
            (_preprocess(key, spec), value) for key, value in raw_map.items()
        )
    invalid = sorted(set(mapping.values()) - canonical_values(schema))
    if invalid:
        raise ValueError(
            f"mapping has values outside the canonical set: {', '.join(invalid)}"
        )
    return mapping


def _seed_mapping(spec: ExperimentSpec) -> dict[str, str]:
    """Preprocess the experiment's label-to-canonical seed, empty when it has none."""
    if spec.mapping_seed is None:
        return {}
    return {
        _preprocess(label, spec): canonical
        for label, canonical in spec.mapping_seed().items()
    }


def _load_cache(directory: Path) -> Cache:
    """Load the promoted LLM resolutions, nested as {lang: {raw: resolution}}."""
    path = directory / _CACHE_FILE
    if not path.is_file():
        return {}
    entries: dict[str, dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    return {
        lang: {
            raw: Resolution.model_validate(fields) for raw, fields in answers.items()
        }
        for lang, answers in entries.items()
    }


def _save_cache(directory: Path, cache: Cache) -> None:
    """Write the promoted LLM resolutions back, sorted so the file stays committable."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = {
        lang: {
            raw: resolution.model_dump(mode="json")
            for raw, resolution in answers.items()
        }
        for lang, answers in cache.items()
    }
    (directory / _CACHE_FILE).write_text(
        json.dumps(entries, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_prompt(folder: str) -> str:
    """Load the experiment's normalization prompt template."""
    path = experiment_dir(folder) / _PROMPT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization prompt: {path}")
    return path.read_text(encoding="utf-8")
