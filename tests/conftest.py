"""Shared test fixtures: a fake, offline OpenAI client and a fake, offline backend."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from llmango import aggregate as aggregate_module
from llmango import analyze as analyze_module
from llmango import manifest as manifest_module
from llmango import pricing as pricing_module
from llmango import storage as storage_module
from llmango.aggregate import Distribution, _distribution
from llmango.backends.base import Backend, GenRequest, GenResult, Usage
from llmango.experiments.e001_fruit import experiment as fruit_module
from llmango.experiments.e001_fruit.experiment import FruitChoice
from llmango.pricing import PricingEntry, PricingTable

SUPPORT = 10


def build_distribution(
    counts: dict[str, int], n_invalid: int = 0, support: int = SUPPORT
) -> Distribution:
    """Build one arm's numbers the way aggregate builds them, from plain counts.

    Hand-rolling the dict here would let every chart test keep passing against a
    shape nothing produces, so the real builder makes it and the tests inherit
    every statistic and interval aggregate stores.
    """
    answers = [name for name, count in counts.items() for _ in range(count)]

    return _distribution(answers, support, n_invalid)


@dataclass
class FakeTokenDetails:
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: FakeTokenDetails | None
    completion_tokens_details: FakeTokenDetails | None


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
    id: str = "chatcmpl-fake"
    service_tier: str | None = "default"
    created: int = 1_700_000_000
    usage: FakeUsage | None = None

    def response_body(self) -> str:
        """The verbatim body a provider returns, which raw.text yields.

        It carries no parsed field, because the provider sends none: parsing is
        the SDK's, and the envelope records only what came back over the wire.
        """
        choice = self.choices[0]
        return json.dumps(
            {
                "id": self.id,
                "object": "chat.completion",
                "created": self.created,
                "model": self.model,
                "service_tier": self.service_tier,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": choice.finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": choice.message.content,
                            "refusal": choice.message.refusal,
                        },
                    }
                ],
            }
        )


@dataclass
class FakeRawResponse:
    """The raw-response wrapper, holding both the body text and the parsed model."""

    completion: FakeCompletion

    @property
    def text(self) -> str:
        return self.completion.response_body()

    def parse(self) -> FakeCompletion:
        return self.completion


@dataclass
class FakeRawCompletions:
    completion: FakeCompletion
    calls: list[dict[str, object]]

    def parse(self, **kwargs: object) -> FakeRawResponse:
        self.calls.append(kwargs)
        return FakeRawResponse(completion=self.completion)

    def create(self, **kwargs: object) -> FakeRawResponse:
        self.calls.append(kwargs)
        return FakeRawResponse(completion=self.completion)


@dataclass
class FakeCompletions:
    completion: FakeCompletion
    calls: list[dict[str, object]]

    @property
    def with_raw_response(self) -> FakeRawCompletions:
        return FakeRawCompletions(completion=self.completion, calls=self.calls)


@dataclass
class FakeChat:
    completions: FakeCompletions


@dataclass
class FakeOpenAIClient:
    chat: FakeChat
    calls: list[dict[str, object]]


def _default_usage() -> FakeUsage:
    """A populated usage object so tests exercise the token and cost columns."""
    return FakeUsage(
        prompt_tokens=12,
        completion_tokens=3,
        total_tokens=15,
        prompt_tokens_details=FakeTokenDetails(cached_tokens=4),
        completion_tokens_details=FakeTokenDetails(reasoning_tokens=1),
    )


def build_fake_openai_client(
    *,
    parsed: BaseModel | None = None,
    content: str | None = None,
    refusal: str | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-5.6-luna-2026-01-01",
    usage: FakeUsage | None = None,
) -> FakeOpenAIClient:
    """Build a fake OpenAI client whose parse call returns a canned completion."""
    message = FakeMessage(content=content, parsed=parsed, refusal=refusal)
    choice = FakeChoice(message=message, finish_reason=finish_reason)
    completion = FakeCompletion(
        choices=[choice], model=model, usage=usage or _default_usage()
    )
    calls: list[dict[str, object]] = []
    return FakeOpenAIClient(
        chat=FakeChat(completions=FakeCompletions(completion=completion, calls=calls)),
        calls=calls,
    )


class FakeBackend(Backend):
    """Deterministic backend answering a scripted list in call order, or "apple"."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self._answers = list(answers or [])
        self.calls = 0

    def generate(self, request: GenRequest) -> GenResult:
        fruit = self._scripted(self.calls)
        self.calls += 1

        return self._result(request, fruit)

    def _scripted(self, index: int) -> str:
        return self._answers[index] if index < len(self._answers) else "apple"

    def _result(self, request: GenRequest, fruit: str) -> GenResult:
        parsed = FruitChoice(fruit=fruit)
        now = datetime.now(UTC)
        return GenResult(
            request=request,
            raw_json=parsed.model_dump_json(),
            parsed=parsed,
            model_snapshot=f"{request.model}-fake",
            finish_reason="stop",
            refusal=None,
            error=None,
            created_at=now,
            response_id="chatcmpl-fake",
            service_tier="default",
            provider_created_at=now,
            response_envelope='{"id": "chatcmpl-fake"}',
            usage=Usage(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
                cached_tokens=4,
                reasoning_tokens=1,
            ),
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


PRICING_TABLE = PricingTable(
    currency="USD",
    unit="per_1m_tokens",
    models={
        "gpt-5.6-luna": PricingEntry(
            input=0.05,
            cached_input=0.005,
            output=0.4,
            last_updated="2026-07-24",
        )
    },
)


@pytest.fixture
def data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every pipeline path into tmp_path, the map normalize writes included."""
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(PRICING_TABLE.model_dump_json(), encoding="utf-8")
    normalization_map = tmp_path / "normalization_map.yaml"
    normalization_map.write_text("", encoding="utf-8")

    monkeypatch.setattr(storage_module, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(storage_module, "NORMALIZED_DIR", tmp_path / "normalized")
    monkeypatch.setattr(manifest_module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(fruit_module, "_NORMALIZATION_MAP", normalization_map)
    monkeypatch.setattr(aggregate_module, "AGG_DIR", tmp_path / "aggregated")
    monkeypatch.setattr(analyze_module, "AGG_DIR", tmp_path / "aggregated")
    monkeypatch.setattr(analyze_module, "CHARTS_DIR", tmp_path / "charts")
    monkeypatch.setattr(pricing_module, "PRICING_FILE", pricing_file)
    return tmp_path
