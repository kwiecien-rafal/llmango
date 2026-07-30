"""Generation backend interface and its request/result value types."""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Self

from pydantic import BaseModel


@dataclass(frozen=True)
class GenRequest:
    """One prompt to generate one response for, free text when it has no schema."""

    model: str
    prompt: str
    response_schema: type[BaseModel] | None
    temperature: float = 1.0


@dataclass(frozen=True)
class Usage:
    """Provider token counts; reasoning_tokens are already part of completion_tokens."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class GenResult:
    """The outcome of one generation."""

    request: GenRequest
    raw_json: str | None
    parsed: BaseModel | None
    model_snapshot: str | None
    finish_reason: str | None
    refusal: str | None
    error: str | None
    created_at: datetime
    response_id: str | None = None
    service_tier: str | None = None
    provider_created_at: datetime | None = None
    request_envelope: str | None = None
    response_envelope: str | None = None
    usage: Usage | None = None

    @classmethod
    def failed(
        cls,
        request: GenRequest,
        error: str,
        created_at: datetime,
        request_envelope: str | None = None,
    ) -> Self:
        """Build a result carrying an error, and what was sent, and no response."""
        return cls(
            request=request,
            raw_json=None,
            parsed=None,
            model_snapshot=None,
            finish_reason=None,
            refusal=None,
            error=error,
            created_at=created_at,
            request_envelope=request_envelope,
        )


class Backend(Protocol):
    """One provider, turning one request at a time into a result."""

    @abstractmethod
    def generate(self, request: GenRequest) -> GenResult:
        """Turn one request into a validated result."""
        ...
