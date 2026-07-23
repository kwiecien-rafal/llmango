"""Repo-root-anchored paths, environment loading, and API key access."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

PROMPTS_DIR = REPO_ROOT / "prompts"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
AGG_DIR = DATA_DIR / "aggregated"
NORMALIZATION_DIR = DATA_DIR / "normalization"
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"


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
