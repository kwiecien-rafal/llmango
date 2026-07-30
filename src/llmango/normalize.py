"""Map one question's raw answers onto canonical categories, cheapest layer first."""

import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict

from llmango.backends import backend_for
from llmango.backends.base import Backend, GenRequest
from llmango.config import NORMALIZE_MODEL, NORMALIZE_PROVIDER, get_experiment_dir
from llmango.experiments import spec_for
from llmango.pricing import guard_cost
from llmango.spec import ExperimentSpec, NormalizationMap, canonical_values
from llmango.storage import read_results, write_normalized

_PROMPT_FILE = "normalize.md"

_STRIP_CHARS = string.punctuation + string.whitespace + "«»„“”‘’¿¡。、「」『』"
_RESOLUTION_COLUMNS = ("canonical", "is_valid")
_ANSWERED = pl.col("error").is_null()


class Resolution(BaseModel):
    """The canonical category a raw answer resolves to, None when it names none."""

    model_config = ConfigDict(frozen=True)

    canonical: str | None
    is_valid: bool


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
            f"No data for question {question_id} to normalize. "
            f"Run 'llmango run {question_id}' first."
        )

    mapping = _load_mapping(spec, schema)
    pairs = _distinct_pairs(frame)

    resolutions: dict[tuple[str, str], Resolution] = {}
    unresolved: list[tuple[str, str]] = []
    for lang, answer in pairs:
        offline = _resolve_offline(answer, spec, mapping)
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
        resolved = _resolve_online(unresolved, spec, schema, backend)
        _promote(resolved, spec)
        resolutions.update(resolved)
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
    answer: str, spec: ExperimentSpec, mapping: NormalizationMap
) -> Resolution | None:
    """Resolve one answer without an LLM: an empty answer, then the mapping table."""
    if not answer.strip():
        return Resolution(canonical=None, is_valid=False)
    key = _preprocess(answer, spec)
    if key not in mapping:
        return None
    canonical = mapping[key]
    return Resolution(canonical=canonical, is_valid=canonical is not None)


def _resolve_online(
    unresolved: list[tuple[str, str]],
    spec: ExperimentSpec,
    schema: type[BaseModel],
    backend: Backend | None,
) -> dict[tuple[str, str], Resolution]:
    """Ask the LLM for what no offline layer resolved, keeping what parses."""
    template = _load_prompt(spec.folder)
    backend = backend or backend_for(NORMALIZE_PROVIDER)
    requests = [
        GenRequest(
            model=NORMALIZE_MODEL,
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
        resolved[(lang, answer)] = _verdict(result.parsed)
    return resolved


def _verdict(parsed: BaseModel) -> Resolution:
    """Read one LLM verdict, dropping the category an invalid answer had to pick."""
    resolution = Resolution.model_validate(parsed, from_attributes=True)
    if resolution.is_valid:
        return resolution
    return Resolution(canonical=None, is_valid=False)


def _promote(resolved: dict[tuple[str, str], Resolution], spec: ExperimentSpec) -> None:
    """Store what was just paid for, keyed as the next run will look it up."""
    promote = spec.promote_normalizations
    if promote is None or not resolved:
        return
    promote(
        {
            _preprocess(answer, spec): resolution.canonical
            for (_, answer), resolution in resolved.items()
        }
    )


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
        f"not written: {preview}. Everything else is in the map, so a rerun "
        f"retries only these."
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


def _load_mapping(spec: ExperimentSpec, schema: type[BaseModel]) -> NormalizationMap:
    """Preprocess the experiment's answer-to-category map, empty when it has none."""
    if spec.normalization_map is None:
        return {}
    mapping = {
        _preprocess(answer, spec): canonical
        for answer, canonical in spec.normalization_map().items()
    }
    named = {canonical for canonical in mapping.values() if canonical is not None}
    invalid = sorted(named - canonical_values(schema))
    if invalid:
        raise ValueError(
            f"{spec.folder} maps answers to values outside the canonical set: "
            f"{', '.join(invalid)}"
        )
    return mapping


def _load_prompt(folder: str) -> str:
    """Load the experiment's normalization prompt template."""
    path = get_experiment_dir(folder) / _PROMPT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization prompt: {path}")
    return path.read_text(encoding="utf-8")
