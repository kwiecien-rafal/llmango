"""Generation backends behind one interface, and the registry naming them."""

from collections.abc import Callable
from functools import cache

from llmango.backends.base import (
    Backend,
    GenRequest,
    GenResult,
)
from llmango.backends.openai import OpenAIBackend

_PROVIDERS: dict[str, Callable[[], Backend]] = {"openai": OpenAIBackend}

__all__ = [
    "Backend",
    "GenRequest",
    "GenResult",
    "OpenAIBackend",
    "backend_for",
]


@cache
def backend_for(provider: str) -> Backend:
    """Build the backend a question's provider names, once per process."""
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Known: {', '.join(sorted(_PROVIDERS))}."
        )
    return _PROVIDERS[provider]()
