"""Generation backends behind one interface."""

from llmango.backends.base import (
    Backend,
    GenRequest,
    GenResult,
)
from llmango.backends.openai import OpenAIBackend

__all__ = [
    "Backend",
    "GenRequest",
    "GenResult",
    "OpenAIBackend",
]
