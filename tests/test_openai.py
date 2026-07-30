"""Tests for the OpenAI backend, with the client faked so nothing hits network."""

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Protocol, cast

import pytest
from openai import OpenAI

from llmango.backends import openai as openai_module
from llmango.backends.base import GenRequest
from llmango.backends.openai import OpenAIBackend
from llmango.experiments.e001_fruit.experiment import FruitChoice


class FakeClient(Protocol):
    """Structural type for the faked OpenAI client the sync tests inspect."""

    calls: list[dict[str, object]]


FakeClientFactory = Callable[..., FakeClient]


_EN_PROMPT = "Pick one random fruit (en)"


def _request(prompt: str = _EN_PROMPT) -> GenRequest:
    return GenRequest(
        model="gpt-5.6-luna",
        prompt=prompt,
        response_schema=FruitChoice,
        temperature=0.5,
    )


def test_require_openai_key_returns_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_module, "load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert openai_module.require_openai_key() == "sk-test"


def test_require_openai_key_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_module, "load_env", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        openai_module.require_openai_key()


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
    assert result.service_tier == "default"
    assert result.provider_created_at is not None
    assert result.response_envelope is not None
    assert "chatcmpl-fake" in result.response_envelope

    assert result.request_envelope is not None
    assert "Pick one random fruit" in result.request_envelope

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


def test_generate_forwards_the_model_and_temperature(
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


def test_generate_free_text_sends_no_response_format(
    make_openai_client: FakeClientFactory,
) -> None:
    client = make_openai_client(parsed=None, content="banana")
    backend = OpenAIBackend(client=cast(OpenAI, client))

    result = backend.generate(replace(_request(), response_schema=None))

    assert result.parsed is None
    assert result.raw_json == "banana"
    assert "response_format" not in client.calls[0]


def test_generate_stores_the_verbatim_response_body(
    make_openai_client: FakeClientFactory,
) -> None:
    """The envelope is the body the provider returned, not a re-serialization of
    the SDK model, so it records what came back over the wire."""
    parsed = FruitChoice(fruit="mango")
    client = make_openai_client(parsed=parsed, content=parsed.model_dump_json())
    backend = OpenAIBackend(client=cast(OpenAI, client))

    result = backend.generate(_request())

    assert result.response_envelope is not None
    body = json.loads(result.response_envelope)
    assert body["id"] == "chatcmpl-fake"
    assert body["choices"][0]["message"]["content"] == parsed.model_dump_json()
    assert "parsed" not in body["choices"][0]["message"]


def test_the_request_envelope_matches_what_the_call_sends(
    make_openai_client: FakeClientFactory,
) -> None:
    """The envelope is provenance, so it must record the call that was made."""
    parsed = FruitChoice(fruit="mango")
    client = make_openai_client(parsed=parsed, content=parsed.model_dump_json())
    backend = OpenAIBackend(client=cast(OpenAI, client))
    request = _request()

    result = backend.generate(request)

    assert result.request_envelope is not None
    envelope = json.loads(result.request_envelope)
    call = client.calls[0]
    assert envelope["model"] == call["model"]
    assert envelope["temperature"] == call["temperature"]
    assert envelope["messages"] == [{"role": "user", "content": request.prompt}]
    schema = envelope["response_format"]["json_schema"]
    assert schema["name"] == "FruitChoice"
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False


def test_the_request_envelope_omits_response_format_for_free_text(
    make_openai_client: FakeClientFactory,
) -> None:
    client = make_openai_client(parsed=None, content="banana")
    backend = OpenAIBackend(client=cast(OpenAI, client))

    result = backend.generate(replace(_request(), response_schema=None))

    assert result.request_envelope is not None
    assert "response_format" not in json.loads(result.request_envelope)
