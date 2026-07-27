"""Every experiment there is, and which question ids each one owns.

A question id is the only identifier the pipeline takes, so this file is the one
place that says what exists. Each experiment declares its questions and SPECS
maps every one of them onto the spec that owns it. Declaring the ids rather than
globbing the prompt tree means a question folder no experiment claims fails
loudly instead of being silently skipped.
"""

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
