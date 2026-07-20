"""Generation backends behind one interface."""

from llmango.backends.base import (
    GenerationBackend,
    GenRequest,
    GenResult,
)
from llmango.backends.openai_backend import OpenAIBackend

__all__ = [
    "GenRequest",
    "GenResult",
    "GenerationBackend",
    "OpenAIBackend",
]
