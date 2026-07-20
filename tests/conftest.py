"""Shared test fixtures: a fake, offline OpenAI client and a fake, offline backend."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

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
    """Deterministic backend that answers with a fixed fruit per language."""

    backend_id = "fake"

    def __init__(self, fruits: dict[str, str] | None = None) -> None:
        self._fruits = fruits or {}

    def resolve_model_snapshot(self, model: str) -> str:
        return f"{model}-fake"

    def generate(self, request: GenRequest) -> GenResult:
        fruit = self._fruits.get(request.lang, "apple")
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
