"""Shared request-body construction for the OpenAI sync and batch backends.

Both backends send the same chat-completions body for a given request. Building
it in one place lets the sync path record the exact bytes it sent as a
request_envelope that mirrors the response_envelope, and guarantees the batch
JSONL and the sync request never drift apart.
"""

import json
from functools import cache
from typing import Any

from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from llmango.backends.base import GenRequest, Usage


def build_usage(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cached_tokens: int | None,
    reasoning_tokens: int | None,
) -> Usage:
    """Build a Usage from the five token counts, coalescing nulls to zero.

    Both backends extract these from different shapes (a typed SDK object vs a
    JSON body); this keeps the null-guarding and field mapping in one place.
    """
    return Usage(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        total_tokens=total_tokens or 0,
        cached_tokens=cached_tokens or 0,
        reasoning_tokens=reasoning_tokens or 0,
    )


@cache
def response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build the strict json_schema response_format for a Pydantic schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": to_strict_json_schema(schema),
            "strict": True,
        },
    }


def request_body(request: GenRequest) -> dict[str, Any]:
    """Build the chat-completions request body for one generation.

    Optional sampling params are included only when set, so the body matches
    exactly what each backend puts on the wire.

    The request seed is deliberately never sent. It keys the option order only.
    The provider treats a seed as a request for repeatable sampling, so sending
    it would ask for the same answer on every sample of a fixed-order question
    and suppress the very randomness these runs measure.
    """
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": request.sampling.temperature,
    }
    if request.response_schema is not None:
        body["response_format"] = response_format(request.response_schema)
    if request.sampling.top_p is not None:
        body["top_p"] = request.sampling.top_p
    if request.sampling.max_tokens is not None:
        body["max_tokens"] = request.sampling.max_tokens
    return body


def request_envelope(request: GenRequest) -> str:
    """Serialize the request body to the verbatim JSON string sent to the API."""
    return json.dumps(request_body(request), ensure_ascii=False)
