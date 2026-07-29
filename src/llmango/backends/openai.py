"""The OpenAI provider module."""

import json
import os
from datetime import UTC, datetime
from functools import cache
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError

from llmango.backends.base import Backend, GenRequest, GenResult, Usage
from llmango.config import REPO_ROOT

_ENDPOINT = "/v1/chat/completions"
_COMPLETION_WINDOW = "24h"


def load_env() -> None:
    """Load environment variables from the repo-root .env file."""
    load_dotenv(REPO_ROOT / ".env")


def require_openai_key() -> str:
    """Load .env and return the OpenAI API key."""
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file at the repo root "
            "or export it in your environment."
        )
    return key


@cache
def _response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build the strict json_schema response_format for a Pydantic schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": to_strict_json_schema(schema),
            "strict": True,
        },
    }


def _request_body(request: GenRequest) -> dict[str, Any]:
    """Build the request body, never sending a seed: it would suppress randomness."""
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": request.temperature,
    }
    if request.response_schema is not None:
        body["response_format"] = _response_format(request.response_schema)
    return body


def _request_envelope(request: GenRequest) -> str:
    """Serialize the request body to the verbatim JSON string sent to the API."""
    return json.dumps(_request_body(request), ensure_ascii=False)


def _custom_id(index: int) -> str:
    """Tie one request to its batched response by its place in the submitted list."""
    return str(index)


def build_jsonl(requests: list[GenRequest]) -> str:
    """Serialize requests to the batch input JSONL, one line per request."""
    lines = [
        json.dumps(
            {
                "custom_id": _custom_id(index),
                "method": "POST",
                "url": _ENDPOINT,
                "body": _request_body(request),
            },
            ensure_ascii=False,
        )
        for index, request in enumerate(requests)
    ]
    return "\n".join(lines)


def _build_usage(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cached_tokens: int | None,
    reasoning_tokens: int | None,
) -> Usage:
    """Build a Usage from the five token counts either transport reports."""
    return Usage(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        total_tokens=total_tokens or 0,
        cached_tokens=cached_tokens or 0,
        reasoning_tokens=reasoning_tokens or 0,
    )


def _usage_from_sdk(usage: CompletionUsage | None) -> Usage | None:
    """Map the SDK usage object onto our Usage, flattening the token details."""
    if usage is None:
        return None
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return _build_usage(
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        prompt_details.cached_tokens if prompt_details else 0,
        completion_details.reasoning_tokens if completion_details else 0,
    )


def _usage_from_body(usage: dict[str, Any] | None) -> Usage | None:
    """Map a batch response body's usage dict onto our Usage."""
    if usage is None:
        return None
    prompt_details: dict[str, Any] = usage.get("prompt_tokens_details") or {}
    completion_details: dict[str, Any] = usage.get("completion_tokens_details") or {}
    return _build_usage(
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        usage.get("total_tokens", 0),
        prompt_details.get("cached_tokens", 0),
        completion_details.get("reasoning_tokens", 0),
    )


def _provider_created_at(created: object) -> datetime | None:
    """Convert a batch body's unix created timestamp to a UTC datetime."""
    if isinstance(created, int):
        return datetime.fromtimestamp(created, UTC)
    return None


class _BatchResponse(BaseModel):
    """The response half of one batch output line."""

    status_code: int
    body: dict[str, Any]


class _BatchLine(BaseModel):
    """One validated line of an OpenAI batch output or error file."""

    custom_id: str
    response: _BatchResponse | None = None
    error: dict[str, Any] | None = None


def _parse_line(line: _BatchLine, request: GenRequest) -> GenResult:
    """Parse one batch line, capturing a bad one rather than aborting the fetch."""
    created_at = datetime.now(UTC)
    envelope = _request_envelope(request)
    if line.error is not None:
        return GenResult.failed(
            request, f"batch error: {line.error}", created_at, envelope
        )
    if line.response is None:
        return GenResult.failed(request, "no response", created_at, envelope)
    if line.response.status_code != 200:
        return GenResult.failed(
            request, f"batch status {line.response.status_code}", created_at, envelope
        )

    try:
        body = line.response.body
        choice = body["choices"][0]
        message = choice["message"]
        refusal = message.get("refusal")
        content = message.get("content")
        schema = request.response_schema
        parsed = (
            schema.model_validate_json(content)
            if schema is not None and refusal is None and content is not None
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
            response_id=body.get("id"),
            service_tier=body.get("service_tier"),
            provider_created_at=_provider_created_at(body.get("created")),
            request_envelope=envelope,
            response_envelope=json.dumps(body, ensure_ascii=False),
            usage=_usage_from_body(body.get("usage")),
        )
    except (KeyError, IndexError, ValidationError) as error:
        return GenResult.failed(
            request, f"unparseable response: {error}", created_at, envelope
        )


class OpenAIBackend(Backend):
    """The OpenAI provider, synchronous by default, batched on request."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=require_openai_key())

    def generate(self, request: GenRequest) -> GenResult:
        """Generate one response inline, recording the verbatim bodies both ways."""
        created_at = datetime.now(UTC)
        envelope = _request_envelope(request)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": request.prompt},
        ]
        try:
            if request.response_schema is not None:
                raw = self._client.chat.completions.with_raw_response.parse(
                    model=request.model,
                    messages=messages,
                    response_format=request.response_schema,
                    temperature=request.temperature,
                )
                completion = raw.parse()
                parsed = completion.choices[0].message.parsed
            else:
                raw = self._client.chat.completions.with_raw_response.create(
                    model=request.model,
                    messages=messages,
                    temperature=request.temperature,
                )
                completion = raw.parse()
                parsed = None
            response_envelope = raw.text
        except Exception as error:
            return GenResult.failed(request, str(error), created_at, envelope)

        choice = completion.choices[0]
        message = choice.message
        return GenResult(
            request=request,
            raw_json=message.content,
            parsed=parsed,
            model_snapshot=completion.model,
            finish_reason=choice.finish_reason,
            refusal=message.refusal,
            error=None,
            created_at=created_at,
            response_id=completion.id,
            service_tier=completion.service_tier,
            provider_created_at=datetime.fromtimestamp(completion.created, UTC),
            request_envelope=envelope,
            response_envelope=response_envelope,
            usage=_usage_from_sdk(completion.usage),
        )

    def generate_many(self, requests: list[GenRequest]) -> list[GenResult]:
        return [self.generate(request) for request in requests]

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
        """Download a completed batch and match every line back to its request."""
        batch = self._client.batches.retrieve(batch_id)
        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} is not ready to fetch (status: {batch.status})."
            )

        lines: dict[str, _BatchLine] = {}
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id is not None:
                lines.update(self._read_lines(file_id))

        expected = {_custom_id(index) for index in range(len(requests))}
        unknown = sorted(set(lines) - expected)
        if unknown:
            raise RuntimeError(
                f"Batch {batch_id} returned {len(unknown)} ids this run did not "
                f"send ({', '.join(unknown[:3])}). A line is matched to its request "
                f"by position, so the two must be the same list."
            )

        results: list[GenResult] = []
        for index, request in enumerate(requests):
            line = lines.get(_custom_id(index))
            if line is None:
                result = GenResult.failed(
                    request,
                    "missing from batch output",
                    datetime.now(UTC),
                    _request_envelope(request),
                )
            else:
                result = _parse_line(line, request)
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
