"""Experiment 001: favorite_fruit.

Asks to name favorite fruit.
"""

from pydantic import BaseModel

from llmango.registry import ExperimentSpec, register_experiment
from llmango.schemas import LLMResponse

QUESTION_ID = "favorite_fruit"


class FruitChoice(LLMResponse):
    """A model's favorite fruit"""

    fruit: str


def to_row(parsed: BaseModel | None) -> dict[str, object]:
    """Map a parsed response to its parsed columns, empty on refusal or error."""
    fruit = parsed.fruit if isinstance(parsed, FruitChoice) else ""
    return {"fruit_raw": fruit}


register_experiment(
    ExperimentSpec(
        question_id=QUESTION_ID,
        response_model=FruitChoice,
        to_row=to_row,
    )
)
