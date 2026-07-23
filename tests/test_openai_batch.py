"""Tests for the OpenAI Batch backend, with the client faked so nothing hits network."""

import json
from dataclasses import dataclass, field
from typing import cast

from openai import OpenAI

from llmango.backends.base import GenRequest
from llmango.backends.openai_batch import OpenAIBatchBackend, build_jsonl
from llmango.experiments.favorite_fruit import FruitChoice
from llmango.questions import SamplingParams


def _request(lang: str = "en", sample_idx: int = 0, seed: int | None = 7) -> GenRequest:
    return GenRequest(
        question_id="001_favorite_fruit",
        lang=lang,
        model="gpt-5.6-luna",
        prompt=f"What is your favorite fruit? ({lang})",
        prompt_sha256="deadbeef",
        sample_idx=sample_idx,
        seed=seed,
        sampling=SamplingParams(temperature=0.5, seed=seed),
        response_schema=FruitChoice,
    )


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


def _client(
    *,
    content_text: str = "",
    status: str = "completed",
    output_file_id: str | None = "file-output",
    error_file_id: str | None = None,
) -> FakeBatchClient:
    return FakeBatchClient(
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


def test_build_jsonl_encodes_each_request() -> None:
    requests = [_request(lang="en", sample_idx=0), _request(lang="pl", sample_idx=1)]

    lines = build_jsonl(requests).splitlines()

    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["custom_id"] == "en::0"
    assert first["method"] == "POST"
    assert first["url"] == "/v1/chat/completions"
    assert first["body"]["model"] == "gpt-5.6-luna"
    assert first["body"]["temperature"] == 0.5
    assert first["body"]["seed"] == 7
    assert first["body"]["messages"] == [
        {"role": "user", "content": "What is your favorite fruit? (en)"}
    ]
    schema = first["body"]["response_format"]["json_schema"]
    assert schema["name"] == "FruitChoice"
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False

    assert json.loads(lines[1])["custom_id"] == "pl::1"


def test_build_jsonl_omits_unset_sampling_params() -> None:
    request = GenRequest(
        question_id="001_favorite_fruit",
        lang="en",
        model="gpt-5.6-luna",
        prompt="prompt",
        prompt_sha256="deadbeef",
        sample_idx=0,
        seed=None,
        sampling=SamplingParams(temperature=1.0),
        response_schema=FruitChoice,
    )

    body = json.loads(build_jsonl([request]))["body"]

    assert "seed" not in body
    assert "top_p" not in body
    assert "max_tokens" not in body


def test_submit_uploads_the_jsonl_and_creates_a_batch() -> None:
    client = _client()
    backend = OpenAIBatchBackend(client=cast(OpenAI, client))

    batch_id = backend.submit([_request()])

    assert batch_id == "batch-1"
    assert client.files.created[0]["purpose"] == "batch"
    created_batch = client.batches.created[0]
    assert created_batch["input_file_id"] == "file-input"
    assert created_batch["endpoint"] == "/v1/chat/completions"
    assert created_batch["completion_window"] == "24h"


def test_fetch_parses_output_lines_back_to_requests() -> None:
    requests = [_request(lang="en", sample_idx=0), _request(lang="pl", sample_idx=1)]
    content = "\n".join(
        [_output_line("pl::1", "banan"), _output_line("en::0", "mango")]
    )
    client = _client(content_text=content)
    backend = OpenAIBatchBackend(client=cast(OpenAI, client))

    results = backend.fetch("batch-1", requests)

    assert [result.request.lang for result in results] == ["en", "pl"]
    assert cast(FruitChoice, results[0].parsed).fruit == "mango"
    assert cast(FruitChoice, results[1].parsed).fruit == "banan"
    assert results[0].model_snapshot == "gpt-5.6-luna-2026-01-01"
    assert all(result.error is None for result in results)


def test_fetch_captures_a_refusal() -> None:
    record = {
        "custom_id": "en::0",
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
    client = _client(content_text=json.dumps(record))
    backend = OpenAIBatchBackend(client=cast(OpenAI, client))

    result = backend.fetch("batch-1", [_request()])[0]

    assert result.parsed is None
    assert result.refusal == "no"
    assert result.error is None


def test_fetch_marks_missing_lines_as_errors() -> None:
    client = _client(content_text=_output_line("en::0", "mango"))
    backend = OpenAIBatchBackend(client=cast(OpenAI, client))

    results = backend.fetch(
        "batch-1",
        [_request(lang="en", sample_idx=0), _request(lang="pl", sample_idx=1)],
    )

    assert results[0].parsed is not None
    assert results[1].parsed is None
    assert results[1].error == "missing from batch output"


def test_fetch_raises_when_the_batch_is_not_complete() -> None:
    client = _client(status="in_progress", output_file_id=None)
    backend = OpenAIBatchBackend(client=cast(OpenAI, client))

    try:
        backend.fetch("batch-1", [_request()])
    except RuntimeError as error:
        assert "in_progress" in str(error)
    else:
        raise AssertionError("fetch should raise when the batch is not complete")


def test_fetch_captures_unparseable_content_without_aborting() -> None:
    truncated = {
        "custom_id": "en::0",
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
    content = "\n".join([json.dumps(truncated), _output_line("pl::1", "mango")])
    client = _client(content_text=content)
    backend = OpenAIBatchBackend(client=cast(OpenAI, client))

    results = backend.fetch(
        "batch-1",
        [_request(lang="en", sample_idx=0), _request(lang="pl", sample_idx=1)],
    )

    assert results[0].parsed is None
    assert results[0].error is not None
    assert cast(FruitChoice, results[1].parsed).fruit == "mango"


def test_fetch_reads_errored_requests_from_the_error_file() -> None:
    error_record = {
        "custom_id": "en::0",
        "response": None,
        "error": {"code": "rate_limit_exceeded", "message": "slow down"},
    }
    client = _client(
        content_text=json.dumps(error_record),
        output_file_id=None,
        error_file_id="file-error",
    )
    backend = OpenAIBatchBackend(client=cast(OpenAI, client))

    result = backend.fetch("batch-1", [_request()])[0]

    assert result.parsed is None
    assert result.error is not None
    assert "rate_limit_exceeded" in result.error
