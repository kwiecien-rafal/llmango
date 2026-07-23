"""OpenAI generation backend using the structured-outputs parse API."""

from datetime import UTC, datetime

from openai import Omit, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam

from llmango.backends.base import GenerationBackend, GenRequest, GenResult
from llmango.config import require_openai_key


def _given[T](value: T | None) -> T | Omit:
    """Map an unset sampling param onto the SDK's omit sentinel."""
    return value if value is not None else omit


class OpenAIBackend(GenerationBackend):
    """Generation backend using the OpenAI structured-outputs parse API."""

    backend_id = "openai"

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=require_openai_key())

    def resolve_model_snapshot(self, model: str) -> str:
        return self._client.models.retrieve(model).id

    def generate(self, request: GenRequest) -> GenResult:
        created_at = datetime.now(UTC)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": request.prompt},
        ]
        try:
            completion = self._client.chat.completions.parse(
                model=request.model,
                messages=messages,
                response_format=request.response_schema,
                temperature=request.sampling.temperature,
                top_p=_given(request.sampling.top_p),
                max_tokens=_given(request.sampling.max_tokens),
                seed=_given(request.seed),
            )
        except Exception as error:
            return GenResult.failed(request, str(error), created_at)

        choice = completion.choices[0]
        message = choice.message
        return GenResult(
            request=request,
            raw_json=message.content,
            parsed=message.parsed,
            model_snapshot=completion.model,
            finish_reason=choice.finish_reason,
            refusal=message.refusal,
            error=None,
            created_at=created_at,
        )
