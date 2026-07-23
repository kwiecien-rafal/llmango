"""Tests for the OpenAI backend, with the client faked so nothing hits network."""

from collections.abc import Callable
from datetime import datetime
from typing import Protocol, cast

from openai import OpenAI

from llmango.backends.base import GenRequest
from llmango.backends.openai_backend import OpenAIBackend
from llmango.experiments.favorite_fruit import FruitChoice
from llmango.questions import SamplingParams


class FakeClient(Protocol):
    """Structural type for the faked OpenAI client the tests inspect."""

    calls: list[dict[str, object]]


FakeClientFactory = Callable[..., FakeClient]


def _request() -> GenRequest:
    return GenRequest(
        question_id="favorite_fruit",
        lang="en",
        model="gpt-5.6-luna",
        prompt="What is your favorite fruit?",
        prompt_sha256="deadbeef",
        sample_idx=0,
        seed=7,
        sampling=SamplingParams(temperature=0.5, seed=7),
        response_schema=FruitChoice,
    )


def test_generate_parses_the_structured_response(
    make_openai_client: FakeClientFactory,
) -> None:
    parsed = FruitChoice(fruit="mango")
    client = make_openai_client(
        parsed=parsed,
        content=parsed.model_dump_json(),
        model="gpt-5.6-luna-2026-01-01",
    )
    backend = OpenAIBackend(client=cast(OpenAI, client))

    result = backend.generate(_request())

    assert result.parsed == parsed
    assert result.raw_json == parsed.model_dump_json()
    assert result.model_snapshot == "gpt-5.6-luna-2026-01-01"
    assert result.finish_reason == "stop"
    assert result.refusal is None
    assert result.error is None
    assert isinstance(result.created_at, datetime)


def test_generate_captures_a_refusal(make_openai_client: FakeClientFactory) -> None:
    client = make_openai_client(
        parsed=None,
        content=None,
        refusal="I can't help with that.",
    )
    backend = OpenAIBackend(client=cast(OpenAI, client))

    result = backend.generate(_request())

    assert result.parsed is None
    assert result.refusal == "I can't help with that."
    assert result.raw_json is None
    assert result.error is None


def test_generate_forwards_the_sampling_params(
    make_openai_client: FakeClientFactory,
) -> None:
    parsed = FruitChoice(fruit="apple")
    client = make_openai_client(parsed=parsed, content=parsed.model_dump_json())
    backend = OpenAIBackend(client=cast(OpenAI, client))

    backend.generate(_request())

    call = client.calls[0]
    assert call["model"] == "gpt-5.6-luna"
    assert call["temperature"] == 0.5
    assert call["seed"] == 7
    assert call["response_format"] is FruitChoice


def test_resolve_model_snapshot_reads_the_client(
    make_openai_client: FakeClientFactory,
) -> None:
    client = make_openai_client(model="gpt-5.6-luna-2026-01-01")
    backend = OpenAIBackend(client=cast(OpenAI, client))

    assert backend.resolve_model_snapshot("gpt-5.6-luna") == "gpt-5.6-luna-2026-01-01"
