"""Tests for the publish stage: what it carries, what it skips, and what it commits."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest
import yaml

from llmango.publish import REPO_TYPE, PublishOutcome, publish_all
from llmango.storage import write_normalized

_CARD = "README.md"


@dataclass
class FakeCommitInfo:
    commit_url: str


@dataclass
class FakeHfApi:
    """A hub client recording what publish sends, over a network it never touches."""

    created: dict[str, Any] = field(default_factory=dict[str, Any])
    operations: list[Any] = field(default_factory=list[Any])
    commit_message: str = ""

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.created = {"repo_id": repo_id, **kwargs}

    def create_commit(
        self, *, repo_id: str, operations: list[Any], commit_message: str, **_: Any
    ) -> FakeCommitInfo:
        self.operations = operations
        self.commit_message = commit_message
        return FakeCommitInfo(f"https://huggingface.co/datasets/{repo_id}/commit/f00")


def _normalize(question_id: str) -> None:
    """Write a question's normalized Parquet through the writer normalize uses."""
    write_normalized(
        pl.DataFrame({"question_id": [question_id], "canonical": ["apple"]}),
        question_id,
    )


def _publish(api: FakeHfApi | None = None, dry_run: bool = False) -> PublishOutcome:
    return publish_all(api=cast(Any, api), dry_run=dry_run)


def _card_of(api: FakeHfApi) -> str:
    """Read back the card the commit carried, as the hub would receive it."""
    operation = next(
        operation for operation in api.operations if operation.path_in_repo == _CARD
    )
    return cast(bytes, operation.path_or_fileobj).decode("utf-8")


def _frontmatter(card: str) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(card.split("---\n")[1]))


@pytest.fixture
def normalized(data_dirs: Path) -> Path:
    """Every question of e001_fruit normalized, which is a whole experiment to ship."""
    for question_id in ("001a", "001b", "001c", "001d"):
        _normalize(question_id)
    return data_dirs


def test_every_question_is_filed_under_its_experiments_folder(normalized: Path) -> None:
    outcome = _publish(dry_run=True)

    assert [upload.path_in_repo for upload in outcome.uploads] == [
        "e001_fruit/001a.parquet",
        "e001_fruit/001b.parquet",
        "e001_fruit/001c.parquet",
        "e001_fruit/001d.parquet",
    ]


def test_an_experiment_is_a_config_and_a_question_is_a_split(normalized: Path) -> None:
    """The two levels the viewer offers, mapped onto the two the pipeline has."""
    api = FakeHfApi()

    _publish(api)

    configs = _frontmatter(_card_of(api))["configs"]
    assert configs == [
        {
            "config_name": "e001_fruit",
            "default": True,
            "data_files": [
                {"split": "001a", "path": "e001_fruit/001a.parquet"},
                {"split": "001b", "path": "e001_fruit/001b.parquet"},
                {"split": "001c", "path": "e001_fruit/001c.parquet"},
                {"split": "001d", "path": "e001_fruit/001d.parquet"},
            ],
        }
    ]


def test_the_card_declares_the_languages_its_questions_are_asked_in(
    normalized: Path,
) -> None:
    """Read off the questions themselves, so a new language cannot go undeclared."""
    api = FakeHfApi()

    _publish(api)

    assert _frontmatter(_card_of(api))["language"] == ["en", "pl", "ja"]


def test_the_card_carries_the_written_body_under_its_frontmatter(
    normalized: Path,
) -> None:
    api = FakeHfApi()

    _publish(api)

    card = _card_of(api)
    assert card.startswith("---\n")
    assert "# LLMango responses" in card.split("---\n", 2)[2]


def test_one_commit_carries_every_parquet_and_the_card(normalized: Path) -> None:
    """A card describing files that landed in a later commit would misread the data."""
    api = FakeHfApi()

    outcome = _publish(api)

    assert [operation.path_in_repo for operation in api.operations] == [
        "e001_fruit/001a.parquet",
        "e001_fruit/001b.parquet",
        "e001_fruit/001c.parquet",
        "e001_fruit/001d.parquet",
        _CARD,
    ]
    assert outcome.commit_url == (
        f"https://huggingface.co/datasets/{outcome.repo_id}/commit/f00"
    )


def test_the_dataset_is_created_private_and_only_once(normalized: Path) -> None:
    """A first publish must not make the data public before it has been looked at."""
    api = FakeHfApi()

    outcome = _publish(api)

    assert api.created == {
        "repo_id": outcome.repo_id,
        "repo_type": REPO_TYPE,
        "private": True,
        "exist_ok": True,
    }


def test_a_question_without_normalized_data_is_skipped_and_named(
    data_dirs: Path,
) -> None:
    _normalize("001a")
    api = FakeHfApi()

    outcome = _publish(api)

    assert [upload.path_in_repo for upload in outcome.uploads] == [
        "e001_fruit/001a.parquet"
    ]
    assert outcome.skipped == ["001b", "001c", "001d"]
    assert _frontmatter(_card_of(api))["configs"][0]["data_files"] == [
        {"split": "001a", "path": "e001_fruit/001a.parquet"}
    ]


def test_a_dry_run_uploads_nothing(normalized: Path) -> None:
    api = FakeHfApi()

    outcome = _publish(api, dry_run=True)

    assert outcome.commit_url is None
    assert api.created == {}
    assert api.operations == []


def test_nothing_normalized_points_at_the_normalize_command(data_dirs: Path) -> None:
    with pytest.raises(FileNotFoundError, match="llmango normalize"):
        _publish(FakeHfApi())
