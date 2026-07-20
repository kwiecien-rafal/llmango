"""Generic Pydantic base for structured LLM responses.

Experiment-specific response models live in the experiments package and inherit
from this base.
"""

from pydantic import BaseModel, ConfigDict


class LLMResponse(BaseModel):
    """Base class for structured LLM response models."""

    model_config = ConfigDict(extra="forbid")
