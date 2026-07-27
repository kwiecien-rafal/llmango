"""Tests for repo-root-anchored paths and content hashing."""

from llmango import config


def test_repo_root_is_the_project_directory() -> None:
    assert config.REPO_ROOT.is_dir()
    assert (config.REPO_ROOT / "pyproject.toml").is_file()


def test_paths_live_under_repo_root_and_exist() -> None:
    paths = [
        config.PROMPTS_DIR,
        config.RAW_DIR,
        config.AGG_DIR,
        config.MAPPINGS_DIR,
        config.RUNS_DIR,
        config.CHARTS_DIR,
    ]
    for path in paths:
        assert config.REPO_ROOT in path.parents
        assert path.is_dir()


def test_sha256_text_is_deterministic() -> None:
    assert config.sha256_text("hello") == config.sha256_text("hello")
    assert config.sha256_text("hello") != config.sha256_text("world")


def test_prompt_tree_helpers_nest_a_question_under_its_experiment() -> None:
    experiment = config.experiment_dir("001_fruit")
    assert experiment == config.PROMPTS_DIR / "001_fruit"
    assert config.question_dir("001_fruit", "001a") == experiment / "001a"
