"""OpenAI Batch API path for mass generation.

Builds a JSONL of chat-completions requests, submits it as one batch job, and
parses the downloaded output back into GenResults. Each request carries a
custom_id encoding its language and sample index, so responses can be matched
back to their request even though the batch returns them in any order.
"""

import json
from datetime import UTC, datetime
from functools import cache
from typing import Any

from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ValidationError

from llmango.backends.base import BatchBackend, GenRequest, GenResult
from llmango.config import require_openai_key

_ENDPOINT = "/v1/chat/completions"
_COMPLETION_WINDOW = "24h"


class _BatchResponse(BaseModel):
    """The response half of one batch output line."""

    status_code: int
    body: dict[str, Any]


class _BatchLine(BaseModel):
    """One validated line of an OpenAI batch output or error file."""

    custom_id: str
    response: _BatchResponse | None = None
    error: dict[str, Any] | None = None


def _custom_id(lang: str, sample_idx: int) -> str:
    """Build the custom_id that ties one request to its batched response."""
    return f"{lang}::{sample_idx}"


@cache
def _response_format(model: type[BaseModel]) -> dict[str, Any]:
    """Build the strict json_schema response_format for a Pydantic model."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": to_strict_json_schema(model),
            "strict": True,
        },
    }


def _body(request: GenRequest) -> dict[str, Any]:
    """Build the chat-completions request body for one generation."""
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "response_format": _response_format(request.response_model),
        "temperature": request.sampling.temperature,
    }
    if request.sampling.top_p is not None:
        body["top_p"] = request.sampling.top_p
    if request.sampling.max_tokens is not None:
        body["max_tokens"] = request.sampling.max_tokens
    if request.seed is not None:
        body["seed"] = request.seed
    return body


def build_jsonl(requests: list[GenRequest]) -> str:
    """Serialize requests to the batch input JSONL, one line per request."""
    lines = [
        json.dumps(
            {
                "custom_id": _custom_id(request.lang, request.sample_idx),
                "method": "POST",
                "url": _ENDPOINT,
                "body": _body(request),
            },
            ensure_ascii=False,
        )
        for request in requests
    ]
    return "\n".join(lines)


def _parse_output(line: _BatchLine, request: GenRequest) -> GenResult:
    """Parse one validated batch line into a GenResult, capturing failures.

    A malformed or schema-invalid response becomes a failed result so one bad
    line never aborts the whole fetch.
    """
    created_at = datetime.now(UTC)
    if line.error is not None:
        return GenResult.failed(request, f"batch error: {line.error}", created_at)
    if line.response is None:
        return GenResult.failed(request, "no response", created_at)
    if line.response.status_code != 200:
        return GenResult.failed(
            request, f"batch status {line.response.status_code}", created_at
        )

    try:
        body = line.response.body
        choice = body["choices"][0]
        message = choice["message"]
        refusal = message.get("refusal")
        content = message.get("content")
        parsed = (
            request.response_model.model_validate_json(content)
            if refusal is None and content is not None
            else None
        )
        return GenResult(
            request=request,
            raw_json=content,
            parsed=parsed,
            model_snapshot=body.get("model"),
            finish_reason=choice.get("finish_reason"),
            refusal=refusal,
            error=None,
            created_at=created_at,
        )
    except (KeyError, IndexError, ValidationError) as error:
        return GenResult.failed(request, f"unparseable response: {error}", created_at)


class OpenAIBatchBackend(BatchBackend):
    """Mass generation through the OpenAI Batch API: submit then fetch."""

    backend_id = "openai-batch"

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=require_openai_key())

    def resolve_model_snapshot(self, model: str) -> str:
        return self._client.models.retrieve(model).id

    def submit(self, requests: list[GenRequest]) -> str:
        """Upload the requests as a JSONL file and start a batch job."""
        upload = self._client.files.create(
            file=("batch.jsonl", build_jsonl(requests).encode("utf-8")),
            purpose="batch",
        )
        batch = self._client.batches.create(
            input_file_id=upload.id,
            endpoint=_ENDPOINT,
            completion_window=_COMPLETION_WINDOW,
        )
        return batch.id

    def fetch(self, batch_id: str, requests: list[GenRequest]) -> list[GenResult]:
        """Download a completed batch's output and parse it into GenResults.

        Raises if the batch has not completed yet. Successful and errored
        requests are read from the output and error files, and each result is
        matched back to its request by custom_id.
        """
        batch = self._client.batches.retrieve(batch_id)
        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} is not ready to fetch (status: {batch.status})."
            )

        lines: dict[str, _BatchLine] = {}
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id is not None:
                lines.update(self._read_lines(file_id))

        results: list[GenResult] = []
        for request in requests:
            line = lines.get(_custom_id(request.lang, request.sample_idx))
            if line is None:
                result = GenResult.failed(
                    request, "missing from batch output", datetime.now(UTC)
                )
            else:
                result = _parse_output(line, request)
            results.append(result)
        return results

    def _read_lines(self, file_id: str) -> dict[str, _BatchLine]:
        """Download a batch result file and validate its lines by custom_id."""
        content = self._client.files.content(file_id).content.decode("utf-8")
        lines: dict[str, _BatchLine] = {}
        for raw in content.splitlines():
            if raw.strip():
                line = _BatchLine.model_validate_json(raw)
                lines[line.custom_id] = line
        return lines
