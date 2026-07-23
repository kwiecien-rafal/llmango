"""Generic Pydantic base for structured LLM responses.

Experiment-specific response schemas live in the experiments package and inherit
from this base.
"""

from pydantic import BaseModel, ConfigDict


class LLMResponse(BaseModel):
    """Base class for structured LLM response schemas."""

    model_config = ConfigDict(extra="forbid")
