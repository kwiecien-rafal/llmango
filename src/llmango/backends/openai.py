"""The OpenAI provider module."""

import json
import os
from datetime import UTC, datetime
from functools import cache
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from llmango.backends.base import Backend, GenRequest, GenResult, Usage
from llmango.config import REPO_ROOT


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


def _request_envelope(request: GenRequest) -> str:
    """Serialize what is sent, never a seed: it would suppress randomness."""
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": request.temperature,
    }

    if request.response_schema is not None:
        body["response_format"] = _response_format(request.response_schema)

    return json.dumps(body, ensure_ascii=False)


def _elapsed(start: float) -> float:
    """How long a call took, to the millisecond, off a clock nothing can adjust."""
    return round(perf_counter() - start, 3)


def _usage_from_sdk(usage: CompletionUsage | None) -> Usage | None:
    """Map the SDK usage object onto our Usage, flattening the token details."""
    if usage is None:
        return None

    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details

    return Usage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
        cached_tokens=(prompt_details.cached_tokens or 0) if prompt_details else 0,
        reasoning_tokens=(
            (completion_details.reasoning_tokens or 0) if completion_details else 0
        ),
    )


class OpenAIBackend(Backend):
    """The OpenAI provider, generating one response per request."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=require_openai_key())

    def generate(self, request: GenRequest) -> GenResult:
        """Generate one response, recording the verbatim bodies both ways."""
        created_at = datetime.now(UTC)
        envelope = _request_envelope(request)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": request.prompt},
        ]
        start = perf_counter()
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
            return GenResult.failed(
                request, str(error), created_at, _elapsed(start), envelope
            )

        generation_seconds = _elapsed(start)
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
            generation_seconds=generation_seconds,
            response_id=completion.id,
            service_tier=completion.service_tier,
            provider_created_at=datetime.fromtimestamp(completion.created, UTC),
            request_envelope=envelope,
            response_envelope=response_envelope,
            usage=_usage_from_sdk(completion.usage),
        )
