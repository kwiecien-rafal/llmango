"""Tests for the OpenAI backend, with the client faked so nothing hits network."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Protocol, cast

from openai import OpenAI

from llmango.backends.base import GenRequest
from llmango.backends.openai_backend import OpenAIBackend
from llmango.experiments.fruit import FruitChoice
from llmango.questions import SamplingParams


class FakeClient(Protocol):
    """Structural type for the faked OpenAI client the tests inspect."""

    calls: list[dict[str, object]]


FakeClientFactory = Callable[..., FakeClient]


def _request() -> GenRequest:
    return GenRequest(
        question_id="001a",
        lang="en",
        model="gpt-5.6-luna",
        prompt="Pick one random fruit from this list: apple, mango",
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


def test_generate_captures_provenance_and_usage(
    make_openai_client: FakeClientFactory,
) -> None:
    parsed = FruitChoice(fruit="mango")
    client = make_openai_client(parsed=parsed, content=parsed.model_dump_json())
    backend = OpenAIBackend(client=cast(OpenAI, client))

    result = backend.generate(_request())

    assert result.response_id == "chatcmpl-fake"
    assert result.system_fingerprint == "fp_fake"
    assert result.service_tier == "default"
    assert result.provider_created_at is not None
    assert result.response_envelope is not None
    assert "chatcmpl-fake" in result.response_envelope

    assert result.request_envelope is not None
    assert "Pick one random fruit from this list" in result.request_envelope

    assert result.usage is not None
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 15
    assert result.usage.cached_tokens == 4
    assert result.usage.reasoning_tokens == 1


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
    assert call["response_format"] is FruitChoice


def test_generate_never_sends_the_seed(
    make_openai_client: FakeClientFactory,
) -> None:
    """The seed keys the option order only; sending it would ask for repeatable
    answers and flatten the distribution the run measures."""
    parsed = FruitChoice(fruit="apple")
    client = make_openai_client(parsed=parsed, content=parsed.model_dump_json())
    backend = OpenAIBackend(client=cast(OpenAI, client))

    backend.generate(_request())

    assert _request().seed == 7
    assert "seed" not in client.calls[0]


def test_generate_free_text_sends_no_response_format(
    make_openai_client: FakeClientFactory,
) -> None:
    client = make_openai_client(parsed=None, content="banana")
    backend = OpenAIBackend(client=cast(OpenAI, client))

    result = backend.generate(replace(_request(), response_schema=None))

    assert result.parsed is None
    assert result.raw_json == "banana"
    assert "response_format" not in client.calls[0]


def test_resolve_model_snapshot_reads_the_client(
    make_openai_client: FakeClientFactory,
) -> None:
    client = make_openai_client(model="gpt-5.6-luna-2026-01-01")
    backend = OpenAIBackend(client=cast(OpenAI, client))

    assert backend.resolve_model_snapshot("gpt-5.6-luna") == "gpt-5.6-luna-2026-01-01"
