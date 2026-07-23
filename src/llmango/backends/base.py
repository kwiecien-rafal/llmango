"""Generation backend interface and its request/result value types.

Every backend turns a GenRequest into a GenResult. The runner, storage and
analysis code depend only on this interface, so adding a backend never requires
touching them.
"""

from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Self

from pydantic import BaseModel

from llmango.questions import SamplingParams


@dataclass(frozen=True)
class GenRequest:
    """One prompt to generate one structured response for."""

    question_id: str
    lang: str
    model: str
    prompt: str
    prompt_sha256: str
    sample_idx: int
    seed: int | None
    sampling: SamplingParams
    response_schema: type[BaseModel]


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

    @classmethod
    def failed(cls, request: GenRequest, error: str, created_at: datetime) -> Self:
        """Build a result carrying an error and no parsed response."""
        return cls(
            request=request,
            raw_json=None,
            parsed=None,
            model_snapshot=None,
            finish_reason=None,
            refusal=None,
            error=error,
            created_at=created_at,
        )


class GenerationBackend(Protocol):
    """The single interface every generation backend implements."""

    backend_id: str

    @abstractmethod
    def resolve_model_snapshot(self, model: str) -> str:
        """Return the exact model snapshot or revision id that will be used."""
        ...

    @abstractmethod
    def generate(self, request: GenRequest) -> GenResult:
        """Turn one request into one validated result."""
        ...

    def generate_many(self, requests: Iterable[GenRequest]) -> list[GenResult]:
        """Generate results for many requests, sequentially by default."""
        return [self.generate(request) for request in requests]


class BatchBackend(Protocol):
    """Interface for a backend that generates through an asynchronous batch job."""

    backend_id: str

    @abstractmethod
    def resolve_model_snapshot(self, model: str) -> str:
        """Return the exact model snapshot or revision id that will be used."""
        ...

    @abstractmethod
    def submit(self, requests: list[GenRequest]) -> str:
        """Submit requests as one batch job and return its id."""
        ...

    @abstractmethod
    def fetch(self, batch_id: str, requests: list[GenRequest]) -> list[GenResult]:
        """Fetch the batch's results, parsed and matched back to the requests."""
        ...
