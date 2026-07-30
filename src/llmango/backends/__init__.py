"""Generation backends behind one interface, and the registry naming them."""

from collections.abc import Callable
from functools import cache

from llmango.backends.base import Backend, GenRequest, GenResult


def _openai() -> Backend:
    """Build the OpenAI backend, importing its SDK only when one is asked for."""
    from llmango.backends.openai import OpenAIBackend

    return OpenAIBackend()


_PROVIDERS: dict[str, Callable[[], Backend]] = {"openai": _openai}

__all__ = [
    "Backend",
    "GenRequest",
    "GenResult",
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
