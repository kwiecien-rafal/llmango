"""Tests for the question id to spec table every command looks through."""

import pytest

from llmango.config import experiment_dir
from llmango.experiments import SPECS, spec_for
from llmango.experiments.fruit import FRUIT


def test_every_declared_question_resolves_to_its_spec() -> None:
    for question_id in FRUIT.questions:
        assert spec_for(question_id) is FRUIT


def test_only_a_question_id_is_an_identifier() -> None:
    """A number, a bare digit and a folder name all resolve to nothing."""
    for ref in ("001", "1", "001_fruit"):
        with pytest.raises(ValueError, match="Unknown question"):
            spec_for(ref)


def test_an_unknown_question_lists_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="Known questions: 001a, 001b, 001c, 001d"):
        spec_for("001z")


def test_every_question_folder_is_declared() -> None:
    """A folder no spec claims would be silently skipped by every stage."""
    specs = {spec.folder: spec for spec in SPECS.values()}
    for folder, spec in specs.items():
        found = {
            child.name
            for child in experiment_dir(folder).iterdir()
            if (child / "meta.yaml").is_file()
        }
        assert found == set(spec.questions)
