"""Repo-root-anchored paths and content hashing. No secrets: those are a backend's."""

import hashlib
from pathlib import Path

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
