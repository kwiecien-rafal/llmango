"""Tests for the normalization pipeline: layers, dedupe, caching and edge rules."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from llmango import normalize as normalize_module
from llmango import questions as questions_module
from llmango.backends.base import GenerationBackend, GenRequest, GenResult
from llmango.experiments.favorite_fruit import FruitNormalization
from llmango.normalize import normalize_question
from llmango.storage import normalized_path, write_results

_MAPPING = "apple: apple\njabłko: apple\nmango: mango\n"
_SLUG = "001_favorite_fruit"


@pytest.fixture
def env(data_dirs: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    norm_dir = data_dirs / "normalization" / _SLUG
    prompt_dir = data_dirs / "prompts" / _SLUG
    monkeypatch.setattr(questions_module, "PROMPTS_DIR", data_dirs / "prompts")

    norm_dir.mkdir(parents=True)
    (norm_dir / "mapping.yaml").write_text(_MAPPING, encoding="utf-8")
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "normalize.md").write_text(
        "Normalize {raw} written in {lang}.", encoding="utf-8"
    )
    return data_dirs


def _raw_row(lang: str, fruit: str, sample_idx: int = 0) -> dict[str, object]:
    return {
        "question_id": "001_favorite_fruit",
        "lang": lang,
        "model": "gpt-5.6-luna",
        "backend": "fake",
        "run_id": "run-1",
        "sample_idx": sample_idx,
        "seed": 0,
        "temperature": 1.0,
        "prompt_sha256": "x",
        "raw_json": None,
        "fruit_raw": fruit,
        "created_at": datetime(2026, 7, 20, tzinfo=UTC),
    }


def _write_raw(rows: list[dict[str, object]]) -> None:
    write_results(rows, "001_favorite_fruit", "gpt-5.6-luna", "run-1")


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


def test_deterministic_layers_dedupe_and_need_no_backend(env: Path) -> None:
    _write_raw(
        [
            _raw_row("en", "apple"),
            _raw_row("en", "apple", sample_idx=1),
            _raw_row("en", "Apple", sample_idx=2),
            _raw_row("en", "ＭＡＮＧＯ", sample_idx=3),
            _raw_row("pl", "jabłko"),
        ]
    )

    outcome = normalize_question("001_favorite_fruit")

    assert outcome.rows == 5
    assert outcome.distinct == 4
    assert outcome.llm_calls == 0

    frame = pl.read_parquet(normalized_path("001_favorite_fruit"))
    resolved = _resolved(frame)
    assert resolved[("en", "apple")] == "apple"
    assert resolved[("en", "Apple")] == "apple"
    assert resolved[("en", "ＭＡＮＧＯ")] == "mango"
    assert resolved[("pl", "jabłko")] == "apple"
    assert frame["is_fruit"].to_list() == [True] * 5


def test_refusal_is_not_a_fruit(env: Path) -> None:
    _write_raw([_raw_row("en", "")])

    outcome = normalize_question("001_favorite_fruit")

    frame = pl.read_parquet(normalized_path("001_favorite_fruit"))
    assert outcome.llm_calls == 0
    assert frame["is_fruit"].to_list() == [False]
    assert frame["fruit_canonical"].to_list() == [""]


def test_cache_hit_skips_the_llm(env: Path) -> None:
    cache = {"en": {"kiwi": {"canonical": "kiwi", "is_fruit": True, "multiple": False}}}
    (
        normalize_module.NORMALIZATION_DIR / _SLUG / "normalization_cache.json"
    ).write_text(json.dumps(cache), encoding="utf-8")
    _write_raw([_raw_row("en", "kiwi")])

    outcome = normalize_question("001_favorite_fruit", make_backend=ExplodingBackend)

    frame = pl.read_parquet(normalized_path("001_favorite_fruit"))
    assert outcome.llm_calls == 0
    assert frame["fruit_canonical"].to_list() == ["kiwi"]


def test_multiple_fruits_take_the_first_and_promote_to_cache(env: Path) -> None:
    result = FruitNormalization(
        raw="banana and apple", canonical="banana", is_fruit=True, multiple=True
    )
    backend = StubBackend(result)
    _write_raw([_raw_row("en", "banana and apple")])

    outcome = normalize_question(
        "001_favorite_fruit", make_backend=lambda: backend, model="gpt-5.6-luna"
    )

    assert outcome.llm_calls == 1
    assert backend.calls == 1

    frame = pl.read_parquet(normalized_path("001_favorite_fruit"))
    assert frame["fruit_canonical"].to_list() == ["banana"]
    assert frame["multiple"].to_list() == [True]
    assert frame["fruit_raw"].to_list() == ["banana and apple"]

    cache = json.loads(
        (
            normalize_module.NORMALIZATION_DIR / _SLUG / "normalization_cache.json"
        ).read_text(encoding="utf-8")
    )
    assert cache["en"]["banana and apple"]["canonical"] == "banana"


def test_punctuation_and_whitespace_resolve_offline(env: Path) -> None:
    _write_raw([_raw_row("en", "apple!"), _raw_row("en", "  Apple.  ", sample_idx=1)])

    outcome = normalize_question("001_favorite_fruit")

    assert outcome.llm_calls == 0
    frame = pl.read_parquet(normalized_path("001_favorite_fruit"))
    assert frame["fruit_canonical"].to_list() == ["apple", "apple"]


def test_cost_guard_blocks_a_large_run_without_force(env: Path) -> None:
    _write_raw([_raw_row("en", "starfruit")])

    with pytest.raises(ValueError, match="smoke limit"):
        normalize_question(
            "001_favorite_fruit", make_backend=ExplodingBackend, max_llm_calls=0
        )


def test_mapping_values_must_be_canonical(env: Path) -> None:
    (normalize_module.NORMALIZATION_DIR / _SLUG / "mapping.yaml").write_text(
        "apple: aple\n", encoding="utf-8"
    )
    _write_raw([_raw_row("en", "apple")])

    with pytest.raises(ValueError, match="canonical set"):
        normalize_question("001_favorite_fruit")


def test_dry_run_counts_llm_work_without_calling_or_writing(env: Path) -> None:
    _write_raw([_raw_row("en", "apple"), _raw_row("en", "starfruit", sample_idx=1)])

    outcome = normalize_question(
        "001_favorite_fruit", make_backend=ExplodingBackend, dry_run=True
    )

    assert outcome.parquet_path is None
    assert outcome.rows == 2
    assert outcome.distinct == 2
    assert outcome.llm_calls == 1
    assert not normalized_path("001_favorite_fruit").is_file()
