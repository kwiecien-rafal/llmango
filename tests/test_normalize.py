"""Tests for the normalization pipeline: layers, dedupe, promotion and edge rules.

These run against the real e001_fruit prompt tree (fruit_list.yaml, normalize.md);
the output paths and the map normalize writes are redirected into
tmp_path. The experiment's fruit labels resolve every in-list answer offline, so
only off-list strings reach the LLM layer.
"""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from llmango import normalize as normalize_module
from llmango.backends.base import GenRequest, GenResult
from llmango.experiments import spec_for
from llmango.experiments.e001_fruit import experiment as fruit_module
from llmango.experiments.e001_fruit.experiment import FruitNormalization
from llmango.normalize import normalize_question
from llmango.rows import column_dtypes
from llmango.storage import normalized_path, write_results

_QUESTION = "001a"


def _map_path() -> Path:
    """The map the normalizer both reads and promotes its LLM verdicts into."""
    return fruit_module._NORMALIZATION_MAP


def _stored() -> dict[str, str | None]:
    """Read back what normalize promoted into the map."""
    return yaml.safe_load(_map_path().read_text(encoding="utf-8")) or {}


_RUN_ID = "001a__20260720T101500000Z"
_SIBLING_RUN_ID = "001b__20260720T101500000Z"

_SHOWN = '{"fruit_list": ["mango", "apple", "banana"]}'


def _raw_row(
    lang: str,
    fruit: str,
    sample_idx: int = 0,
    prompt_inputs: str = _SHOWN,
    error: str | None = None,
    question_id: str = _QUESTION,
    run_id: str = _RUN_ID,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "lang": lang,
        "model": "gpt-5.6-luna",
        "provider": "fake",
        "run_id": run_id,
        "sample_idx": sample_idx,
        "temperature": 1.0,
        "prompt_sha256": "x",
        "prompt_inputs": prompt_inputs,
        "raw_json": None,
        "answer": fruit,
        "error": error,
        "created_at": datetime(2026, 7, 20, tzinfo=UTC),
    }


def _write_raw(rows: list[dict[str, object]], run_id: str = _RUN_ID) -> None:
    write_results(rows, run_id, column_dtypes({}))


def _resolved(frame: pl.DataFrame) -> dict[tuple[str, str], str]:
    langs = frame.get_column("lang").to_list()
    answers = frame.get_column("answer").to_list()
    canonical = frame.get_column("canonical").to_list()
    return {
        (lang, answer): canon
        for lang, answer, canon in zip(langs, answers, canonical, strict=True)
    }


class ExplodingBackend:
    """Backend that fails the test if the LLM layer ever calls it."""

    def generate_many(self, requests: list[GenRequest]) -> list[GenResult]:
        raise AssertionError("the LLM layer should not have been called")


class StubBackend:
    """Backend that answers every request with a fixed normalization."""

    def __init__(self, result: FruitNormalization) -> None:
        self._result = result
        self.calls = 0

    def generate_many(self, requests: list[GenRequest]) -> list[GenResult]:
        return [self._generate(request) for request in requests]

    def _generate(self, request: GenRequest) -> GenResult:
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


class FlakyBackend(StubBackend):
    """Backend that answers every request except the one naming a given answer."""

    def __init__(self, result: FruitNormalization, failing: str) -> None:
        super().__init__(result)
        self._failing = failing

    def _generate(self, request: GenRequest) -> GenResult:
        if self._failing in request.prompt:
            return GenResult.failed(request, "rate limited", datetime.now(UTC))
        return super()._generate(request)


def test_fruit_labels_resolve_offline_and_dedupe(data_dirs: Path) -> None:
    _write_raw(
        [
            _raw_row("en", "apple"),
            _raw_row("en", "apple", sample_idx=1),
            _raw_row("en", "Apple", sample_idx=2),
            _raw_row("en", "ＭＡＮＧＯ", sample_idx=3),
            _raw_row("pl", "jabłko"),
        ]
    )

    outcome = normalize_question(_QUESTION)

    assert outcome.rows == 5
    assert outcome.distinct == 4
    assert outcome.llm_calls == 0

    frame = pl.read_parquet(normalized_path(_QUESTION))
    resolved = _resolved(frame)
    assert resolved[("en", "apple")] == "apple"
    assert resolved[("en", "Apple")] == "apple"
    assert resolved[("en", "ＭＡＮＧＯ")] == "mango"
    assert resolved[("pl", "jabłko")] == "apple"
    assert frame["is_valid"].to_list() == [True] * 5


def test_only_the_question_asked_for_is_read(data_dirs: Path) -> None:
    """A sibling question's answers belong to its own normalized file, not this one."""
    _write_raw([_raw_row("en", "apple")])
    _write_raw(
        [_raw_row("en", "banana", question_id="001b", run_id=_SIBLING_RUN_ID)],
        run_id=_SIBLING_RUN_ID,
    )

    outcome = normalize_question(_QUESTION)

    assert outcome.rows == 1
    assert pl.read_parquet(normalized_path(_QUESTION))["answer"].to_list() == ["apple"]
    assert not normalized_path("001b").is_file()


def test_a_question_with_no_raw_results_says_so(data_dirs: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No data for question 001a"):
        normalize_question(_QUESTION)


def test_a_refusal_names_no_category(data_dirs: Path) -> None:
    _write_raw([_raw_row("en", "")])

    outcome = normalize_question(_QUESTION)

    frame = pl.read_parquet(normalized_path(_QUESTION))
    assert outcome.llm_calls == 0
    assert frame["is_valid"].to_list() == [False]
    assert frame["canonical"].to_list() == [None]


def test_an_errored_call_is_never_adjudicated(data_dirs: Path) -> None:
    """A call that failed carries no answer, so it is not a refusal either."""
    _write_raw(
        [
            _raw_row("en", "", error="rate limited"),
            _raw_row("en", "", sample_idx=1),
            _raw_row("en", "apple", sample_idx=2),
        ]
    )

    outcome = normalize_question(_QUESTION, backend=ExplodingBackend())

    frame = pl.read_parquet(normalized_path(_QUESTION)).sort("sample_idx")
    assert outcome.rows == 3
    assert outcome.distinct == 2
    assert frame["is_valid"].to_list() == [None, False, True]
    assert frame["canonical"].to_list() == [None, None, "apple"]


def test_added_columns_sit_next_to_the_answer(data_dirs: Path) -> None:
    """The pipeline's two columns first, then whatever the experiment appends."""
    _write_raw([_raw_row("en", "apple")])

    normalize_question(_QUESTION)

    columns = pl.read_parquet(normalized_path(_QUESTION)).columns
    start = columns.index("answer")
    assert columns[start : start + 4] == [
        "answer",
        "canonical",
        "is_valid",
        "chosen_position",
    ]


def test_a_stored_answer_skips_the_llm(data_dirs: Path) -> None:
    _map_path().write_text("kiwi: other\n", encoding="utf-8")
    _write_raw([_raw_row("en", "kiwi")])

    outcome = normalize_question(_QUESTION, backend=ExplodingBackend())

    frame = pl.read_parquet(normalized_path(_QUESTION))
    assert outcome.llm_calls == 0
    assert frame["canonical"].to_list() == ["other"]


def test_a_stored_null_means_the_answer_named_no_fruit(data_dirs: Path) -> None:
    """A null value is how the map records an answer that named nothing countable."""
    _map_path().write_text("nie wiem:\n", encoding="utf-8")
    _write_raw([_raw_row("pl", "nie wiem")])

    outcome = normalize_question(_QUESTION, backend=ExplodingBackend())

    frame = pl.read_parquet(normalized_path(_QUESTION))
    assert outcome.llm_calls == 0
    assert frame["canonical"].to_list() == [None]
    assert frame["is_valid"].to_list() == [False]


def test_multiple_fruits_take_the_first_and_promote_to_the_map(
    data_dirs: Path,
) -> None:
    result = FruitNormalization(canonical="banana", is_valid=True)
    backend = StubBackend(result)
    _write_raw([_raw_row("en", "banana and apple")])

    outcome = normalize_question(_QUESTION, backend=backend)

    assert outcome.llm_calls == 1
    assert backend.calls == 1

    frame = pl.read_parquet(normalized_path(_QUESTION))
    assert frame["canonical"].to_list() == ["banana"]
    assert frame["answer"].to_list() == ["banana and apple"]

    assert _stored()["banana and apple"] == "banana"


def test_an_answer_naming_no_fruit_is_promoted_as_null(data_dirs: Path) -> None:
    """The schema forces a category on a refusal; the map must not record it."""
    backend = StubBackend(FruitNormalization(canonical="other", is_valid=False))
    _write_raw([_raw_row("pl", "nie mam zdania")])

    normalize_question(_QUESTION, backend=backend)

    assert _stored()["nie mam zdania"] is None
    frame = pl.read_parquet(normalized_path(_QUESTION))
    assert frame["canonical"].to_list() == [None]
    assert frame["is_valid"].to_list() == [False]


def test_an_unparsed_answer_fails_the_run_but_keeps_the_paid_results(
    data_dirs: Path,
) -> None:
    result = FruitNormalization(canonical="other", is_valid=True)
    backend = FlakyBackend(result, failing="durian")
    _write_raw([_raw_row("en", "starfruit"), _raw_row("en", "durian", sample_idx=1)])

    with pytest.raises(ValueError, match="unparsed"):
        normalize_question(_QUESTION, backend=backend)

    assert not normalized_path(_QUESTION).is_file()
    stored = _stored()
    assert "starfruit" in stored
    assert "durian" not in stored


def test_punctuation_and_whitespace_resolve_offline(data_dirs: Path) -> None:
    _write_raw([_raw_row("en", "apple!"), _raw_row("en", "  Apple.  ", sample_idx=1)])

    outcome = normalize_question(_QUESTION)

    assert outcome.llm_calls == 0
    frame = pl.read_parquet(normalized_path(_QUESTION))
    assert frame["canonical"].to_list() == ["apple", "apple"]


def test_cost_guard_blocks_a_large_run_without_force(data_dirs: Path) -> None:
    """Only the answers no offline layer resolved count against the limit."""
    off_list = [_raw_row("en", f"starfruit {index}", index) for index in range(101)]
    _write_raw(off_list)

    with pytest.raises(ValueError, match="unforced limit"):
        normalize_question(_QUESTION, backend=ExplodingBackend())


def test_normalization_map_values_must_be_canonical(
    data_dirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An experiment mapping an answer onto a category its schema omits is an error."""
    spec = replace(
        spec_for(_QUESTION), normalization_map=lambda: {"starfruit": "notafruit"}
    )
    monkeypatch.setattr(normalize_module, "spec_for", lambda _: spec)
    _write_raw([_raw_row("en", "apple")])

    with pytest.raises(ValueError, match="canonical set"):
        normalize_question(_QUESTION)


def test_dry_run_counts_llm_work_without_calling_or_writing(data_dirs: Path) -> None:
    _write_raw([_raw_row("en", "apple"), _raw_row("en", "starfruit", sample_idx=1)])

    outcome = normalize_question(_QUESTION, backend=ExplodingBackend(), dry_run=True)

    assert outcome.parquet_path is None
    assert outcome.rows == 2
    assert outcome.distinct == 2
    assert outcome.llm_calls == 1
    assert not normalized_path(_QUESTION).is_file()
