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
    """One prompt to generate one response for.

    response_schema is None for a free-text request, which sends no structured
    output. prompt_inputs and schema_variant are carried through unused by
    backends so the runner can record them as provenance columns. prompt_inputs
    is the JSON of what each of the question's prompt inputs resolved to for this
    sample. Both are empty for a request that is not an arm of a question, such as
    a normalization call, whose results never reach a raw parquet.

    seed keys the per-sample prompt inputs and is provenance only. A backend must
    never forward it to a provider as a sampling seed, which would ask for repeatable
    answers and flatten the distribution the run exists to measure.
    """

    question_id: str
    lang: str
    model: str
    prompt: str
    prompt_sha256: str
    sample_idx: int
    seed: int | None
    sampling: SamplingParams
    response_schema: type[BaseModel] | None
    prompt_inputs: str = "{}"
    schema_variant: str = ""


@dataclass(frozen=True)
class Usage:
    """Token counts reported by the provider for one generation.

    reasoning_tokens are already part of completion_tokens; they are kept for
    information and never added to cost a second time.
    """

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
    system_fingerprint: str | None = None
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
        """Build a result carrying an error and no parsed response.

        The request envelope is still recorded when known, so a failed call
        remains traceable to exactly what was sent.
        """
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
