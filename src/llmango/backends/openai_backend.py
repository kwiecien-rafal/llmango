"""OpenAI generation backend using the structured-outputs parse API."""

from datetime import UTC, datetime

from openai import Omit, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage

from llmango.backends.base import GenerationBackend, GenRequest, GenResult, Usage
from llmango.backends.openai_common import build_usage, request_envelope
from llmango.config import require_openai_key


def _given[T](value: T | None) -> T | Omit:
    """Map an unset sampling param onto the SDK's omit sentinel."""
    return value if value is not None else omit


def _usage(usage: CompletionUsage | None) -> Usage | None:
    """Map the SDK usage object onto our Usage, flattening the token details."""
    if usage is None:
        return None
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return build_usage(
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        prompt_details.cached_tokens if prompt_details else 0,
        completion_details.reasoning_tokens if completion_details else 0,
    )


class OpenAIBackend(GenerationBackend):
    """Generation backend using the OpenAI structured-outputs parse API."""

    backend_id = "openai"

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=require_openai_key())

    def resolve_model_snapshot(self, model: str) -> str:
        return self._client.models.retrieve(model).id

    def generate(self, request: GenRequest) -> GenResult:
        created_at = datetime.now(UTC)
        envelope = request_envelope(request)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": request.prompt},
        ]
        try:
            if request.response_schema is not None:
                completion = self._client.chat.completions.parse(
                    model=request.model,
                    messages=messages,
                    response_format=request.response_schema,
                    temperature=request.sampling.temperature,
                    top_p=_given(request.sampling.top_p),
                    max_tokens=_given(request.sampling.max_tokens),
                    seed=_given(request.seed),
                )
                parsed = completion.choices[0].message.parsed
            else:
                completion = self._client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    temperature=request.sampling.temperature,
                    top_p=_given(request.sampling.top_p),
                    max_tokens=_given(request.sampling.max_tokens),
                    seed=_given(request.seed),
                )
                parsed = None
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
            system_fingerprint=completion.system_fingerprint,
            service_tier=completion.service_tier,
            provider_created_at=datetime.fromtimestamp(completion.created, UTC),
            request_envelope=envelope,
            response_envelope=completion.model_dump_json(),
            usage=_usage(completion.usage),
        )
