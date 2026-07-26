"""Tests for the normalization pipeline: layers, dedupe, caching and edge rules.

These run against the real 001_fruit prompt tree (fruits.yaml, experiment.yaml,
normalize.md); only the output directories are redirected into tmp_path. The
fruit table seeds the deterministic mapping, so every in-list answer resolves
offline and only off-list strings reach the LLM layer.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from llmango import normalize as normalize_module
from llmango.backends.base import GenerationBackend, GenRequest, GenResult
from llmango.experiments.fruit import FruitNormalization
from llmango.normalize import normalize_experiment
from llmango.storage import normalized_path, write_results

_EXPERIMENT = "001_fruit"


@pytest.fixture
def env(data_dirs: Path) -> Path:
    """Redirect outputs into tmp_path; the prompt tree stays the real one."""
    (data_dirs / "mappings" / _EXPERIMENT).mkdir(parents=True)
    return data_dirs


_RUN_ID = "001a__en__20260720T101500Z__c3f9a1"

_SHOWN = '["mango", "apple", "banana"]'


def _raw_row(
    lang: str,
    fruit: str,
    sample_idx: int = 0,
    option_order: str = _SHOWN,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "question_id": "001a",
        "lang": lang,
        "schema_variant": "en",
        "schema_name": "FruitChoice",
        "model": "gpt-5.6-luna",
        "backend": "fake",
        "run_id": _RUN_ID,
        "sample_idx": sample_idx,
        "seed": 0,
        "temperature": 1.0,
        "prompt_sha256": "x",
        "option_order": option_order,
        "raw_json": None,
        "fruit_raw": fruit,
        "error": error,
        "created_at": datetime(2026, 7, 20, tzinfo=UTC),
    }


def _write_raw(rows: list[dict[str, object]]) -> None:
    write_results(rows, _RUN_ID, "gpt-5.6-luna")


def _resolved(frame: pl.DataFrame) -> dict[tuple[str, str], str]:
    langs = frame.get_column("lang").to_list()
    raws = frame.get_column("fruit_raw").to_list()
    canonical = frame.get_column("fruit_canonical").to_list()
    return {
        (lang, raw): canon
        for lang, raw, canon in zip(langs, raws, canonical, strict=True)
    }


class ExplodingBackend(GenerationBackend):
    """Backend that fails the test if the LLM layer ever calls it."""

    backend_id = "boom"

    def resolve_model_snapshot(self, model: str) -> str:
        return model

    def generate(self, request: GenRequest) -> GenResult:
        raise AssertionError("the LLM layer should not have been called")


class StubBackend(GenerationBackend):
    """Backend that answers every request with a fixed normalization."""

    backend_id = "stub"

    def __init__(self, result: FruitNormalization) -> None:
        self._result = result
        self.calls = 0

    def resolve_model_snapshot(self, model: str) -> str:
        return model

    def generate(self, request: GenRequest) -> GenResult:
        self.calls += 1
        return GenResult(
            request=request,
            raw_json=self._result.model_dump_json(),
            parsed=self._result,
            model_snapshot="stub",
            finish_reason="stop",
            refusal=None,
            error=None,
            created_at=datetime.now(UTC),
        )


def test_fruit_labels_resolve_offline_and_dedupe(env: Path) -> None:
    _write_raw(
        [
            _raw_row("en", "apple"),
            _raw_row("en", "apple", sample_idx=1),
            _raw_row("en", "Apple", sample_idx=2),
            _raw_row("en", "ＭＡＮＧＯ", sample_idx=3),
            _raw_row("pl", "jabłko"),
        ]
    )

    outcome = normalize_experiment(_EXPERIMENT)

    assert outcome.rows == 5
    assert outcome.distinct == 4
    assert outcome.llm_calls == 0

    frame = pl.read_parquet(normalized_path(_EXPERIMENT))
    resolved = _resolved(frame)
    assert resolved[("en", "apple")] == "apple"
    assert resolved[("en", "Apple")] == "apple"
    assert resolved[("en", "ＭＡＮＧＯ")] == "mango"
    assert resolved[("pl", "jabłko")] == "apple"
    assert frame["is_fruit"].to_list() == [True] * 5


def test_refusal_is_not_a_fruit(env: Path) -> None:
    _write_raw([_raw_row("en", "")])

    outcome = normalize_experiment(_EXPERIMENT)

    frame = pl.read_parquet(normalized_path(_EXPERIMENT))
    assert outcome.llm_calls == 0
    assert frame["is_fruit"].to_list() == [False]
    assert frame["fruit_canonical"].to_list() == [None]
    assert frame["chosen_position"].to_list() == [None]


def test_chosen_position_reports_where_the_answer_was_shown(env: Path) -> None:
    _write_raw(
        [
            _raw_row("en", "mango"),
            _raw_row("pl", "jabłko", sample_idx=1),
            _raw_row("en", "banana", sample_idx=2),
        ]
    )

    normalize_experiment(_EXPERIMENT)

    frame = pl.read_parquet(normalized_path(_EXPERIMENT)).sort("sample_idx")
    assert frame["fruit_canonical"].to_list() == ["mango", "apple", "banana"]
    assert frame["chosen_position"].to_list() == [1, 2, 3]


def test_chosen_position_is_null_when_the_answer_was_not_shown(env: Path) -> None:
    _write_raw([_raw_row("en", "kiwi", option_order='["mango", "apple"]')])

    normalize_experiment(
        _EXPERIMENT,
        make_backend=lambda: StubBackend(
            FruitNormalization(
                raw="kiwi", canonical="kiwi", is_fruit=True, multiple=False
            )
        ),
        model="gpt-5.6-luna",
    )

    frame = pl.read_parquet(normalized_path(_EXPERIMENT))
    assert frame["fruit_canonical"].to_list() == ["kiwi"]
    assert frame["chosen_position"].to_list() == [None]


def test_added_columns_sit_next_to_the_raw_answer(env: Path) -> None:
    _write_raw([_raw_row("en", "apple")])

    normalize_experiment(_EXPERIMENT)

    columns = pl.read_parquet(normalized_path(_EXPERIMENT)).columns
    start = columns.index("fruit_raw")
    assert columns[start : start + 5] == [
        "fruit_raw",
        "fruit_canonical",
        "is_fruit",
        "multiple",
        "chosen_position",
    ]


def test_cache_hit_skips_the_llm(env: Path) -> None:
    cache = {"en": {"kiwi": {"canonical": "kiwi", "is_fruit": True, "multiple": False}}}
    (
        normalize_module.MAPPINGS_DIR / _EXPERIMENT / "normalization_cache.json"
    ).write_text(json.dumps(cache), encoding="utf-8")
    _write_raw([_raw_row("en", "kiwi")])

    outcome = normalize_experiment(_EXPERIMENT, make_backend=ExplodingBackend)

    frame = pl.read_parquet(normalized_path(_EXPERIMENT))
    assert outcome.llm_calls == 0
    assert frame["fruit_canonical"].to_list() == ["kiwi"]


def test_multiple_fruits_take_the_first_and_promote_to_cache(env: Path) -> None:
    result = FruitNormalization(
        raw="banana and apple", canonical="banana", is_fruit=True, multiple=True
    )
    backend = StubBackend(result)
    _write_raw([_raw_row("en", "banana and apple")])

    outcome = normalize_experiment(
        _EXPERIMENT, make_backend=lambda: backend, model="gpt-5.6-luna"
    )

    assert outcome.llm_calls == 1
    assert backend.calls == 1

    frame = pl.read_parquet(normalized_path(_EXPERIMENT))
    assert frame["fruit_canonical"].to_list() == ["banana"]
    assert frame["multiple"].to_list() == [True]
    assert frame["fruit_raw"].to_list() == ["banana and apple"]

    cache_path = (
        normalize_module.MAPPINGS_DIR / _EXPERIMENT / "normalization_cache.json"
    )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache["en"]["banana and apple"]["canonical"] == "banana"


def test_punctuation_and_whitespace_resolve_offline(env: Path) -> None:
    _write_raw([_raw_row("en", "apple!"), _raw_row("en", "  Apple.  ", sample_idx=1)])

    outcome = normalize_experiment(_EXPERIMENT)

    assert outcome.llm_calls == 0
    frame = pl.read_parquet(normalized_path(_EXPERIMENT))
    assert frame["fruit_canonical"].to_list() == ["apple", "apple"]


def test_cost_guard_blocks_a_large_run_without_force(env: Path) -> None:
    _write_raw([_raw_row("en", "starfruit")])

    with pytest.raises(ValueError, match="smoke limit"):
        normalize_experiment(
            _EXPERIMENT, make_backend=ExplodingBackend, max_llm_calls=0
        )


def test_mapping_values_must_be_canonical(env: Path) -> None:
    (normalize_module.MAPPINGS_DIR / _EXPERIMENT / "mapping.yaml").write_text(
        "starfruit: notafruit\n", encoding="utf-8"
    )
    _write_raw([_raw_row("en", "apple")])

    with pytest.raises(ValueError, match="canonical set"):
        normalize_experiment(_EXPERIMENT)


def test_dry_run_counts_llm_work_without_calling_or_writing(env: Path) -> None:
    _write_raw([_raw_row("en", "apple"), _raw_row("en", "starfruit", sample_idx=1)])

    outcome = normalize_experiment(
        _EXPERIMENT, make_backend=ExplodingBackend, dry_run=True
    )

    assert outcome.parquet_path is None
    assert outcome.rows == 2
    assert outcome.distinct == 2
    assert outcome.llm_calls == 1
    assert not normalized_path(_EXPERIMENT).is_file()
