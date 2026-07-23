"""Shared test fixtures: a fake, offline OpenAI client and a fake, offline backend."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from llmango import analyze as analyze_module
from llmango import manifest as manifest_module
from llmango import normalize as normalize_module
from llmango import storage as storage_module
from llmango.backends.base import GenerationBackend, GenRequest, GenResult
from llmango.experiments.favorite_fruit import FruitChoice


@dataclass
class FakeMessage:
    content: str | None
    parsed: BaseModel | None
    refusal: str | None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]
    model: str


@dataclass
class FakeModelInfo:
    id: str


@dataclass
class FakeCompletions:
    completion: FakeCompletion
    calls: list[dict[str, object]]

    def parse(self, **kwargs: object) -> FakeCompletion:
        self.calls.append(kwargs)
        return self.completion


@dataclass
class FakeChat:
    completions: FakeCompletions


@dataclass
class FakeModels:
    model_id: str

    def retrieve(self, model: str) -> FakeModelInfo:
        return FakeModelInfo(id=self.model_id)


@dataclass
class FakeOpenAIClient:
    chat: FakeChat
    models: FakeModels
    calls: list[dict[str, object]]


def build_fake_openai_client(
    *,
    parsed: BaseModel | None = None,
    content: str | None = None,
    refusal: str | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-5.6-luna-2026-01-01",
) -> FakeOpenAIClient:
    """Build a fake OpenAI client whose parse call returns a canned completion."""
    message = FakeMessage(content=content, parsed=parsed, refusal=refusal)
    choice = FakeChoice(message=message, finish_reason=finish_reason)
    completion = FakeCompletion(choices=[choice], model=model)
    calls: list[dict[str, object]] = []
    return FakeOpenAIClient(
        chat=FakeChat(completions=FakeCompletions(completion=completion, calls=calls)),
        models=FakeModels(model_id=model),
        calls=calls,
    )


class FakeBackend(GenerationBackend):
    """Deterministic backend that answers with a scripted fruit per lang and sample.

    Answers are read as answers[lang][sample_idx]; an unscripted language falls
    back to "apple", so the zero-argument default still answers every request.
    """

    backend_id = "fake"

    def __init__(self, answers: dict[str, list[str]] | None = None) -> None:
        self._answers = answers or {}

    def resolve_model_snapshot(self, model: str) -> str:
        return f"{model}-fake"

    def generate(self, request: GenRequest) -> GenResult:
        scripted = self._answers.get(request.lang)
        fruit = scripted[request.sample_idx] if scripted else "apple"
        parsed = FruitChoice(fruit=fruit)
        return GenResult(
            request=request,
            raw_json=parsed.model_dump_json(),
            parsed=parsed,
            model_snapshot=self.resolve_model_snapshot(request.model),
            finish_reason="stop",
            refusal=None,
            error=None,
            created_at=datetime.now(UTC),
        )


@pytest.fixture
def make_openai_client() -> Callable[..., FakeOpenAIClient]:
    return build_fake_openai_client


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def make_fake_backend() -> Callable[..., FakeBackend]:
    return FakeBackend


@pytest.fixture
def data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every pipeline output directory into tmp_path."""
    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(storage_module, "NORMALIZED_DIR", tmp_path / "normalized")
    monkeypatch.setattr(manifest_module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        normalize_module, "NORMALIZATION_DIR", tmp_path / "normalization"
    )
    monkeypatch.setattr(analyze_module, "AGG_DIR", tmp_path / "aggregated")
    return tmp_path
