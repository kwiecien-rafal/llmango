"""Experiment 001: favorite_fruit.

Asks to name favorite fruit.
"""

from llmango.registry import ExperimentSpec, register_experiment
from llmango.schemas import LLMResponse

QUESTION_ID = "favorite_fruit"


class FruitChoice(LLMResponse):
    """A model's favorite fruit"""

    fruit: str


register_experiment(ExperimentSpec(question_id=QUESTION_ID, response_model=FruitChoice))
