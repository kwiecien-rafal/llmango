"""Every experiment there is, and which question ids each one owns."""

from llmango.experiments.e001_fruit import FRUIT
from llmango.spec import ExperimentSpec

SPECS: dict[str, ExperimentSpec] = {question: FRUIT for question in FRUIT.questions}


def spec_for(question_id: str) -> ExperimentSpec:
    """Return the question's spec.

    Specs are defined per-experiment in
    src\\llmango\\experiments\\<experiment_id>\\experiment.py"""

    try:
        return SPECS[question_id]
    except KeyError:
        known = ", ".join(sorted(SPECS))
        raise ValueError(
            f"Unknown question: {question_id!r}. Known questions: {known}."
        ) from None
