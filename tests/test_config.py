"""Tests for repo-root-anchored paths and API key access."""

import pytest

from llmango import config


def test_repo_root_is_the_project_directory() -> None:
    assert config.REPO_ROOT.is_dir()
    assert (config.REPO_ROOT / "pyproject.toml").is_file()


def test_paths_live_under_repo_root_and_exist() -> None:
    paths = [
        config.PROMPTS_DIR,
        config.RAW_DIR,
        config.AGG_DIR,
        config.NORMALIZATION_DIR,
        config.RUNS_DIR,
        config.RESULTS_DIR,
    ]
    for path in paths:
        assert config.REPO_ROOT in path.parents
        assert path.is_dir()


def test_require_openai_key_returns_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert config.require_openai_key() == "sk-test"


def test_require_openai_key_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        config.require_openai_key()
