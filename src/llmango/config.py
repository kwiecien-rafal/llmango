"""Repo-root-anchored paths, pipeline settings and content hashing."""

import hashlib
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

PROMPTS_DIR = REPO_ROOT / "prompts"
DATA_DIR = REPO_ROOT / "data"
CHARTS_DIR = REPO_ROOT / "site" / "public" / "charts"
DATASET_CARD_FILE = REPO_ROOT / "DATASET_CARD.md"

NORMALIZE_PROVIDER = "openai"
NORMALIZE_MODEL = "gpt-5.6-luna"

HF_DATASET_REPO = "rafalkwiecien/llmango"


def load_env() -> None:
    """Load environment variables from the repo-root .env file."""
    load_dotenv(REPO_ROOT / ".env")


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of a piece of content, for manifests and provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_experiment_prompt_dir(folder: str) -> Path:
    """Return an experiment's prompt folder, holding its shared files and questions."""
    return PROMPTS_DIR / folder


def get_question_prompt_dir(folder: str, question_id: str) -> Path:
    """Return one question's prompt folder under its experiment's."""
    return get_experiment_prompt_dir(folder) / question_id


def get_experiment_data_dir(folder: str) -> Path:
    """Return an experiment's data folder, holding every stage's output for it."""
    return DATA_DIR / folder


def get_raw_dir(folder: str) -> Path:
    """Return the folder an experiment's runs append their raw JSONL to."""
    return get_experiment_data_dir(folder) / "raw"


def get_raw_results_path(folder: str, run_id: str) -> Path:
    """Return the JSONL path one run appends its results to."""
    return get_raw_dir(folder) / f"{run_id}.jsonl"


def get_manifest_path(folder: str, run_id: str) -> Path:
    """Return the manifest path recording how one run was configured."""
    return get_experiment_data_dir(folder) / "manifests" / f"{run_id}.json"


def get_normalized_path(folder: str, question_id: str) -> Path:
    """Return the Parquet path one question's normalized answers are written to."""
    return get_experiment_data_dir(folder) / "normalized" / f"{question_id}.parquet"


def get_aggregate_path(folder: str, question_id: str) -> Path:
    """Return the JSON path one question's committed numbers are written to."""
    return get_experiment_data_dir(folder) / "aggregated" / f"{question_id}.json"


def get_charts_dir(folder: str) -> Path:
    """Return the served folder an experiment's charts are drawn into."""
    return CHARTS_DIR / folder
