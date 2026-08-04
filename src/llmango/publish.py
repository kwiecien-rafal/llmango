"""Publish every normalized question to the HuggingFace dataset, in one commit."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from llmango.config import (
    DATASET_CARD_FILE,
    HF_DATASET_REPO,
    get_normalized_path,
    load_env,
)
from llmango.experiments import EXPERIMENTS
from llmango.questions import load_question

if TYPE_CHECKING:
    from huggingface_hub import HfApi

REPO_TYPE = "dataset"

_CARD_IN_REPO = "README.md"
_COMMIT_MESSAGE = "Publish normalized responses"
_LICENSE = "cc-by-4.0"
_PRETTY_NAME = "LLMango: visualizing AI behaviour"
_TAGS = ("llm", "multilingual", "structured-outputs", "randomness", "evaluation")


@dataclass(frozen=True)
class Upload:
    """One local file and the path it takes inside the dataset repo."""

    local_path: Path
    path_in_repo: str


@dataclass(frozen=True)
class PublishOutcome:
    """What a publish carried, what it had nothing for, and where it landed."""

    repo_id: str
    uploads: list[Upload]
    skipped: list[str]
    commit_url: str | None


def publish_all(*, api: "HfApi | None" = None, dry_run: bool = False) -> PublishOutcome:
    """Upload every normalized question and the card describing them, in one commit."""
    published, skipped = _publishable()

    if not published:
        raise FileNotFoundError(
            "No normalized data to publish. Run 'llmango normalize <question_id>' "
            "first."
        )

    uploads = _uploads(published)
    card = _card(published)

    if dry_run:
        return PublishOutcome(HF_DATASET_REPO, uploads, skipped, commit_url=None)

    return PublishOutcome(
        repo_id=HF_DATASET_REPO,
        uploads=uploads,
        skipped=skipped,
        commit_url=_commit(uploads, card, api),
    )


def _publishable() -> tuple[dict[str, list[str]], list[str]]:
    """Group the questions holding a normalized Parquet by experiment, name the rest."""
    published: dict[str, list[str]] = {}
    skipped: list[str] = []
    for experiment in EXPERIMENTS:
        for question_id in experiment.questions:
            if get_normalized_path(experiment.folder, question_id).is_file():
                published.setdefault(experiment.folder, []).append(question_id)
            else:
                skipped.append(question_id)

    return published, skipped


def _uploads(published: dict[str, list[str]]) -> list[Upload]:
    """The Parquet of every published question, filed under its experiment's folder."""
    return [
        Upload(
            local_path=get_normalized_path(folder, question_id),
            path_in_repo=f"{folder}/{question_id}.parquet",
        )
        for folder, question_ids in published.items()
        for question_id in question_ids
    ]


def _card(published: dict[str, list[str]]) -> str:
    """Render the card: generated frontmatter over the body DATASET_CARD.md holds."""
    frontmatter = yaml.safe_dump(
        _metadata(published), allow_unicode=True, sort_keys=False
    )
    body = DATASET_CARD_FILE.read_text(encoding="utf-8")

    return f"---\n{frontmatter}---\n\n{body}"


def _metadata(published: dict[str, list[str]]) -> dict[str, object]:
    """The frontmatter the viewer reads: the editorial fields, and what was uploaded."""
    return {
        "pretty_name": _PRETTY_NAME,
        "license": _LICENSE,
        "language": _languages(published),
        "tags": list(_TAGS),
        "configs": [
            _config(folder, question_ids, default=index == 0)
            for index, (folder, question_ids) in enumerate(published.items())
        ],
    }


def _config(
    folder: str, question_ids: list[str], *, default: bool
) -> dict[str, object]:
    """One experiment as a config, each of its questions a split the viewer lists."""
    return {
        "config_name": folder,
        **({"default": True} if default else {}),
        "data_files": [
            {"split": question_id, "path": f"{folder}/{question_id}.parquet"}
            for question_id in question_ids
        ],
    }


def _languages(published: dict[str, list[str]]) -> list[str]:
    """Every language the published questions are asked in, as each one declares it."""
    languages = [
        lang
        for question_ids in published.values()
        for question_id in question_ids
        for lang in load_question(question_id).languages
    ]

    return list(dict.fromkeys(languages))


def _commit(uploads: list[Upload], card: str, api: "HfApi | None") -> str:
    """Create the dataset if it is new, then land every file in one commit."""
    from huggingface_hub import CommitOperationAdd

    hub = api or _api()
    hub.create_repo(HF_DATASET_REPO, repo_type=REPO_TYPE, private=True, exist_ok=True)
    operations = [
        CommitOperationAdd(
            path_in_repo=upload.path_in_repo, path_or_fileobj=upload.local_path
        )
        for upload in uploads
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=_CARD_IN_REPO, path_or_fileobj=card.encode("utf-8")
        )
    )
    commit = hub.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message=_COMMIT_MESSAGE,
    )

    return commit.commit_url


def _api() -> "HfApi":
    """Build the hub client only now, so a dry run needs neither it nor a token."""
    from huggingface_hub import HfApi

    load_env()

    return HfApi()
