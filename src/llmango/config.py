"""Repo-root-anchored paths, content hashing, environment loading and API key access.

The prompt tree's shape and the way any piece of content is hashed live here so
that every stage agrees on both without importing each other.
"""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

PROMPTS_DIR = REPO_ROOT / "prompts"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
AGG_DIR = DATA_DIR / "aggregated"
MAPPINGS_DIR = DATA_DIR / "mappings"
RUNS_DIR = REPO_ROOT / "runs"
SITE_DIR = REPO_ROOT / "site"
CHARTS_DIR = SITE_DIR / "public" / "charts"
PRICING_FILE = DATA_DIR / "pricing.json"


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of a piece of content, for manifests and provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def experiment_dir(folder: str) -> Path:
    """Return an experiment's prompt folder, holding its shared files and questions."""
    return PROMPTS_DIR / folder


def question_dir(folder: str, question_id: str) -> Path:
    """Return one question's folder under its experiment's."""
    return experiment_dir(folder) / question_id


def load_env() -> None:
    """Load environment variables from the repo-root .env file."""
    load_dotenv(REPO_ROOT / ".env")


def require_openai_key() -> str:
    """Load .env and return the OpenAI API key."""
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file at the repo root "
            "or export it in your environment."
        )
    return key
