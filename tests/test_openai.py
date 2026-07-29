"""Tests for the OpenAI backend, with the client faked so nothing hits network.

Both transports are exercised here because both are the same backend: the sync
path through generate, the batch path through submit and fetch.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol, cast

import pytest
from openai import OpenAI

from llmango.backends import openai as openai_module
from llmango.backends.base import GenRequest
from llmango.backends.openai import OpenAIBackend, build_jsonl
from llmango.experiments.e001_fruit.experiment import FruitChoice


class FakeClient(Protocol):
    """Structural type for the faked OpenAI client the sync tests inspect."""

    calls: list[dict[str, object]]


FakeClientFactory = Callable[..., FakeClient]


_EN_PROMPT = "Pick one random fruit (en)"
_PL_PROMPT = "Pick one random fruit (pl)"


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


def test_generate_many_answers_every_request_in_order(
    make_openai_client: FakeClientFactory,
) -> None:
    parsed = FruitChoice(fruit="mango")
    client = make_openai_client(parsed=parsed, content=parsed.model_dump_json())
    backend = OpenAIBackend(client=cast(OpenAI, client))

    requests = [_request(), _request(_PL_PROMPT)]
    results = backend.generate_many(requests)

    assert [result.request for result in results] == requests
    assert len(client.calls) == 2


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
    the SDK model, so the sync and batch transports record the same shape."""
    parsed = FruitChoice(fruit="mango")
    client = make_openai_client(parsed=parsed, content=parsed.model_dump_json())
    backend = OpenAIBackend(client=cast(OpenAI, client))

    result = backend.generate(_request())

    assert result.response_envelope is not None
    body = json.loads(result.response_envelope)
    assert body["id"] == "chatcmpl-fake"
    assert body["choices"][0]["message"]["content"] == parsed.model_dump_json()
    assert "parsed" not in body["choices"][0]["message"]


@dataclass
class FakeFile:
    id: str


@dataclass
class FakeBinaryContent:
    content: bytes


@dataclass
class FakeFiles:
    content_text: str
    created: list[dict[str, object]] = field(default_factory=list)

    def create(self, *, file: object, purpose: str) -> FakeFile:
        self.created.append({"file": file, "purpose": purpose})
        return FakeFile(id="file-input")

    def content(self, file_id: str) -> FakeBinaryContent:
        return FakeBinaryContent(content=self.content_text.encode("utf-8"))


@dataclass
class FakeBatch:
    id: str
    status: str
    output_file_id: str | None
    error_file_id: str | None = None


@dataclass
class FakeBatches:
    batch: FakeBatch
    created: list[dict[str, object]] = field(default_factory=list)

    def create(
        self, *, input_file_id: str, endpoint: str, completion_window: str
    ) -> FakeBatch:
        self.created.append(
            {
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
            }
        )
        return self.batch

    def retrieve(self, batch_id: str) -> FakeBatch:
        return self.batch


@dataclass
class FakeBatchClient:
    files: FakeFiles
    batches: FakeBatches


def _output_line(custom_id: str, fruit: str) -> str:
    completion = {
        "model": "gpt-5.6-luna-2026-01-01",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": FruitChoice(fruit=fruit).model_dump_json(),
                    "refusal": None,
                },
            }
        ],
    }
    record = {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": completion},
        "error": None,
    }
    return json.dumps(record)


def _batch_backend(
    *,
    content_text: str = "",
    status: str = "completed",
    output_file_id: str | None = "file-output",
    error_file_id: str | None = None,
) -> tuple[OpenAIBackend, FakeBatchClient]:
    """Build a backend over a fake batch client, returning both."""
    client = FakeBatchClient(
        files=FakeFiles(content_text=content_text),
        batches=FakeBatches(
            batch=FakeBatch(
                id="batch-1",
                status=status,
                output_file_id=output_file_id,
                error_file_id=error_file_id,
            )
        ),
    )
    return OpenAIBackend(client=cast(OpenAI, client)), client


def test_build_jsonl_encodes_each_request() -> None:
    requests = [_request(), _request(_PL_PROMPT)]

    lines = build_jsonl(requests).splitlines()

    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["custom_id"] == "0"
    assert first["method"] == "POST"
    assert first["url"] == "/v1/chat/completions"
    assert first["body"]["model"] == "gpt-5.6-luna"
    assert first["body"]["temperature"] == 0.5
    assert first["body"]["messages"] == [{"role": "user", "content": _EN_PROMPT}]
    schema = first["body"]["response_format"]["json_schema"]
    assert schema["name"] == "FruitChoice"
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False

    assert json.loads(lines[1])["custom_id"] == "1"


def test_build_jsonl_omits_response_format_for_free_text() -> None:
    body = json.loads(build_jsonl([replace(_request(), response_schema=None)]))["body"]

    assert "response_format" not in body


def test_the_batch_body_matches_what_the_sync_call_sends(
    make_openai_client: FakeClientFactory,
) -> None:
    """One request body serves both transports, so they cannot drift apart."""
    parsed = FruitChoice(fruit="mango")
    client = make_openai_client(parsed=parsed, content=parsed.model_dump_json())
    backend = OpenAIBackend(client=cast(OpenAI, client))
    request = _request()

    result = backend.generate(request)

    assert result.request_envelope is not None
    assert (
        json.loads(result.request_envelope)
        == json.loads(build_jsonl([request]))["body"]
    )


def test_submit_uploads_the_jsonl_and_creates_a_batch() -> None:
    backend, client = _batch_backend()

    batch_id = backend.submit([_request()])

    assert batch_id == "batch-1"
    assert client.files.created[0]["purpose"] == "batch"
    created_batch = client.batches.created[0]
    assert created_batch["input_file_id"] == "file-input"
    assert created_batch["endpoint"] == "/v1/chat/completions"
    assert created_batch["completion_window"] == "24h"


def test_fetch_parses_output_lines_back_to_requests() -> None:
    """Lines come back in any order, so each is matched to its own request."""
    requests = [_request(), _request(_PL_PROMPT)]
    content = "\n".join([_output_line("1", "banan"), _output_line("0", "mango")])
    backend, _ = _batch_backend(content_text=content)

    results = backend.fetch("batch-1", requests)

    assert [result.request.prompt for result in results] == [_EN_PROMPT, _PL_PROMPT]
    assert cast(FruitChoice, results[0].parsed).fruit == "mango"
    assert cast(FruitChoice, results[1].parsed).fruit == "banan"
    assert results[0].model_snapshot == "gpt-5.6-luna-2026-01-01"
    assert all(result.error is None for result in results)


def test_fetch_captures_provenance_and_usage() -> None:
    record = {
        "custom_id": "0",
        "response": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-batch",
                "model": "gpt-5.6-luna-2026-01-01",
                "service_tier": "flex",
                "created": 1_700_000_000,
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "total_tokens": 24,
                    "prompt_tokens_details": {"cached_tokens": 8},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": FruitChoice(fruit="mango").model_dump_json(),
                            "refusal": None,
                        },
                    }
                ],
            },
        },
        "error": None,
    }
    backend, _ = _batch_backend(content_text=json.dumps(record))

    result = backend.fetch("batch-1", [_request()])[0]

    assert result.response_id == "chatcmpl-batch"
    assert result.service_tier == "flex"
    assert result.provider_created_at is not None
    assert result.response_envelope is not None
    assert "chatcmpl-batch" in result.response_envelope
    assert result.request_envelope is not None
    assert "Pick one random fruit" in result.request_envelope
    assert result.usage is not None
    assert result.usage.prompt_tokens == 20
    assert result.usage.cached_tokens == 8
    assert result.usage.reasoning_tokens == 2


def test_fetch_captures_a_refusal() -> None:
    record = {
        "custom_id": "0",
        "response": {
            "status_code": 200,
            "body": {
                "model": "gpt-5.6-luna-2026-01-01",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": None, "refusal": "no"},
                    }
                ],
            },
        },
        "error": None,
    }
    backend, _ = _batch_backend(content_text=json.dumps(record))

    result = backend.fetch("batch-1", [_request()])[0]

    assert result.parsed is None
    assert result.refusal == "no"
    assert result.error is None


def test_fetch_marks_missing_lines_as_errors() -> None:
    backend, _ = _batch_backend(content_text=_output_line("0", "mango"))

    results = backend.fetch("batch-1", [_request(), _request(_PL_PROMPT)])

    assert results[0].parsed is not None
    assert results[1].parsed is None
    assert results[1].error == "missing from batch output"


def test_fetch_refuses_a_batch_it_did_not_send() -> None:
    """A line is matched by position, so an id this run never sent is not guesswork."""
    stale = _output_line("FruitChoice::en::0", "mango")
    backend, _ = _batch_backend(content_text=stale)

    with pytest.raises(RuntimeError, match="did not send"):
        backend.fetch("batch-1", [_request()])


def test_fetch_raises_when_the_batch_is_not_complete() -> None:
    backend, _ = _batch_backend(status="in_progress", output_file_id=None)

    with pytest.raises(RuntimeError, match="in_progress"):
        backend.fetch("batch-1", [_request()])


def test_fetch_captures_unparseable_content_without_aborting() -> None:
    truncated = {
        "custom_id": "0",
        "response": {
            "status_code": 200,
            "body": {
                "model": "gpt-5.6-luna-2026-01-01",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": '{"fruit": "man', "refusal": None},
                    }
                ],
            },
        },
        "error": None,
    }
    content = "\n".join([json.dumps(truncated), _output_line("1", "mango")])
    backend, _ = _batch_backend(content_text=content)

    results = backend.fetch("batch-1", [_request(), _request(_PL_PROMPT)])

    assert results[0].parsed is None
    assert results[0].error is not None
    assert cast(FruitChoice, results[1].parsed).fruit == "mango"


def test_fetch_reads_errored_requests_from_the_error_file() -> None:
    error_record = {
        "custom_id": "0",
        "response": None,
        "error": {"code": "rate_limit_exceeded", "message": "slow down"},
    }
    backend, _ = _batch_backend(
        content_text=json.dumps(error_record),
        output_file_id=None,
        error_file_id="file-error",
    )

    result = backend.fetch("batch-1", [_request()])[0]

    assert result.parsed is None
    assert result.error is not None
    assert "rate_limit_exceeded" in result.error
