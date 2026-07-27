"""Post-hoc normalization of free-text answers to canonical categories.

Takes a question id, reads that question's raw answers and maps each onto a
canonical English category in layers, cheapest first. The experiment's own
mapping seed resolves every answer it already has a label for; leftover strings
(off-list or free-text) fall through to the mapping file and then an LLM. The
answer column is never overwritten, so normalization can be re-run with better
methods without regenerating anything.

The mapping file and the LLM cache stay experiment-wide, in the folder the spec
names, so a string one question paid to resolve is free for every sibling
question that ever sees it again.

The canonical, is_valid and multiple columns are the pipeline's own words for
what normalization decides, so they are spelled the same way in the resolution,
in the cache on disk, on the schema the model fills and in the parquet. An
experiment appends whatever else it can compute from the result.
"""

import json
import string
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import get_args

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict

from llmango.backends.base import Backend, GenRequest
from llmango.config import MAPPINGS_DIR, experiment_dir
from llmango.experiments import spec_for
from llmango.questions import SamplingParams, load_experiment_config
from llmango.spec import ExperimentSpec
from llmango.storage import read_results, write_normalized

_MAPPING_FILE = "mapping.yaml"
_CACHE_FILE = "normalization_cache.json"
_PROMPT_FILE = "normalize.md"

_PUNCTUATION = string.punctuation + "«»„“”‘’¿¡… "


class Resolution(BaseModel):
    """The canonical category a raw answer resolves to.

    Reads straight off a normalization response or a cache entry, both of which
    spell these three fields the pipeline's way. Any other field the schema
    carries, such as the echoed answer, is ignored.
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


def normalize_question(
    question_id: str,
    *,
    make_backend: Callable[[], Backend] | None = None,
    model: str | None = None,
    max_llm_calls: int | None = None,
    dry_run: bool = False,
) -> NormalizeOutcome:
    """Add canonical categories to one question's raw answers and write them out.

    Reads every raw run of the question, resolves each distinct answer per
    language through the deterministic layers and then the LLM for the rest, and
    writes a normalized Parquet file that leaves the raw answers untouched. The
    backend is built lazily, so a run resolved entirely offline needs no API key.
    A dry run stops after the offline layers and reports how many answers the LLM
    would resolve, without calling it or writing anything.
    """
    spec = spec_for(question_id)
    normalization_schema = spec.normalization_schema
    if normalization_schema is None:
        raise ValueError(f"Experiment {spec.folder} has no normalization schema.")

    frame = read_results(f"{question_id}__*.parquet")
    if frame.is_empty():
        raise FileNotFoundError(f"No raw results to normalize for {question_id}.")

    directory = MAPPINGS_DIR / spec.folder
    mapping = _load_mapping(directory, spec)
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
    parquet_path = write_normalized(normalized, question_id)
    return NormalizeOutcome(
        parquet_path=parquet_path,
        rows=frame.height,
        distinct=len(pairs),
        llm_calls=len(unresolved),
    )


def _distinct_pairs(frame: pl.DataFrame) -> list[tuple[str, str]]:
    """Return the sorted, deduped (lang, answer) pairs from the raw frame."""
    pairs = frame.select("lang", "answer").unique().sort("lang", "answer")
    return [(str(lang), str(answer)) for lang, answer in pairs.iter_rows()]


def _resolve_offline(
    lang: str,
    answer: str,
    spec: ExperimentSpec,
    mapping: dict[str, str],
    cache: dict[str, dict[str, dict[str, object]]],
) -> Resolution | None:
    """Resolve one answer without an LLM: refusal, mapping table, then cache."""
    if not answer.strip():
        return Resolution(canonical="", is_valid=False, multiple=False)
    canonical = mapping.get(preprocess(answer, spec))
    if canonical is not None:
        return Resolution(canonical=canonical, is_valid=True, multiple=False)
    cached = cache.get(lang, {}).get(answer)
    if cached is not None:
        return Resolution.model_validate(cached)
    return None


def _resolve_online(
    unresolved: list[tuple[str, str]],
    spec: ExperimentSpec,
    response_schema: type[BaseModel],
    make_backend: Callable[[], Backend] | None,
    model: str | None,
    max_llm_calls: int | None,
    cache: dict[str, dict[str, dict[str, object]]],
) -> dict[tuple[str, str], Resolution]:
    """Guard cost, build the backend lazily, and resolve the leftover answers.

    Only a parsed response becomes a resolution. Anything else is left out, so the
    caller can save what was paid for and then report what is missing.
    """
    folder = spec.folder
    if max_llm_calls is not None and len(unresolved) > max_llm_calls:
        raise ValueError(
            f"{len(unresolved)} answers need the LLM layer, above the smoke limit "
            f"of {max_llm_calls}. Re-run with --force to allow the paid calls."
        )
    if make_backend is None:
        raise ValueError(
            f"{len(unresolved)} answers need the LLM layer but no backend given."
        )
    resolved_model = model or _normalize_model(folder)
    if not resolved_model:
        raise ValueError(f"No model given to normalize {folder}.")

    template = _load_prompt(folder)
    requests = [
        GenRequest(
            question_id=folder,
            lang=lang,
            model=resolved_model,
            prompt=template.replace("{lang}", lang).replace("{raw}", answer),
            prompt_sha256="",
            sample_idx=index,
            seed=None,
            sampling=SamplingParams(temperature=0.0),
            response_schema=response_schema,
        )
        for index, (lang, answer) in enumerate(unresolved)
    ]
    results = make_backend().generate_many(requests)

    resolved: dict[tuple[str, str], Resolution] = {}
    for (lang, answer), result in zip(unresolved, results, strict=True):
        if result.parsed is None:
            continue
        resolution = Resolution.model_validate(result.parsed.model_dump(mode="json"))
        resolved[(lang, answer)] = resolution
        cache.setdefault(lang, {})[answer] = resolution.model_dump(mode="json")
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
    """Attach the canonical columns to every raw row via its (lang, answer).

    An answer that resolved to no category at all, such as a refusal, is written
    as a null rather than an empty string, so the column never carries a category
    that does not exist. The experiment appends whatever else it can derive from
    the resolved frame, which the pipeline has no way to compute for it.
    """
    rows = [
        {
            "lang": lang,
            "answer": answer,
            "canonical": resolution.canonical or None,
            "is_valid": resolution.is_valid,
            "multiple": resolution.multiple,
        }
        for (lang, answer), resolution in resolutions.items()
    ]
    schema: dict[str, pl.DataType] = {
        "lang": pl.String(),
        "answer": pl.String(),
        "canonical": pl.String(),
        "is_valid": pl.Boolean(),
        "multiple": pl.Boolean(),
    }
    resolution_frame = pl.DataFrame(rows, schema_overrides=schema)
    joined = frame.join(resolution_frame, on=["lang", "answer"], how="left")
    add = spec.extra_normalized_columns
    extra = add(joined) if add is not None else {}
    return _order_columns(joined.with_columns(**extra), list(extra))


def _order_columns(frame: pl.DataFrame, extra: list[str]) -> pl.DataFrame:
    """Move the added columns to sit directly after the answer they describe."""
    added = ["canonical", "is_valid", "multiple", *extra]
    kept = [column for column in frame.columns if column not in added]
    cut = kept.index("answer") + 1
    return frame.select(kept[:cut] + added + kept[cut:])


def _normalize_model(folder: str) -> str | None:
    """Return the configured normalization model, falling back to the run model."""
    config = load_experiment_config(folder)
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

    The experiment covers each of its own questions, because a question may
    override an input with its own data file while normalization spans every
    question at once.
    """
    if spec.mapping_seed is None:
        return {}
    seed = spec.mapping_seed()
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


def _load_prompt(folder: str) -> str:
    path = experiment_dir(folder) / _PROMPT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization prompt: {path}")
    return path.read_text(encoding="utf-8")
