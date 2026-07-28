"""Every experiment there is, and which question ids each one owns."""

from llmango.experiments.fruit import FRUIT
from llmango.spec import ExperimentSpec

SPECS: dict[str, ExperimentSpec] = {question: FRUIT for question in FRUIT.questions}


def spec_for(question_id: str) -> ExperimentSpec:
    """Return the spec owning a question id, or raise listing the ids that exist."""
    try:
        return SPECS[question_id]
    except KeyError:
        known = ", ".join(sorted(SPECS))
        raise ValueError(
            f"Unknown question: {question_id!r}. Known questions: {known}."
        ) from None
