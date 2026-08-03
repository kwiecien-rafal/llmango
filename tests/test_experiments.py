"""Tests for the question id to spec table every command looks through."""

import pytest

from llmango.config import get_experiment_dir
from llmango.experiments import EXPERIMENTS, SPECS, spec_for
from llmango.experiments.e001_fruit import FRUIT


def test_every_declared_question_resolves_to_its_spec() -> None:
    for question_id in FRUIT.questions:
        assert spec_for(question_id) is FRUIT


def test_the_lookup_table_covers_every_experiment_there_is() -> None:
    """SPECS is derived from EXPERIMENTS, which is the one list analyze walks."""
    assert set(SPECS) == {
        question for spec in EXPERIMENTS for question in spec.questions
    }


def test_no_two_experiments_claim_the_same_question() -> None:
    """A collision would silently hand one experiment's question to the other."""
    declared = [question for spec in EXPERIMENTS for question in spec.questions]

    assert len(declared) == len(set(declared))


def test_only_a_question_id_is_an_identifier() -> None:
    """A number, a bare digit and a folder name all resolve to nothing."""
    for ref in ("001", "1", "e001_fruit"):
        with pytest.raises(ValueError, match="Unknown question"):
            spec_for(ref)


def test_an_unknown_question_lists_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="Known questions: 001a, 001b, 001c, 001d"):
        spec_for("001z")


def test_every_question_folder_is_declared() -> None:
    """A folder no spec claims would be silently skipped by every stage."""
    for spec in EXPERIMENTS:
        found = {
            child.name
            for child in get_experiment_dir(spec.folder).iterdir()
            if (child / "question.yaml").is_file()
        }
        assert found == set(spec.questions)
