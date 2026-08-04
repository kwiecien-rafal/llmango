"""Tests for repo-root-anchored paths, pipeline settings and content hashing."""

from pathlib import Path

from llmango import config


def _stage_paths(folder: str) -> list[Path]:
    """Every path a pipeline stage writes for one experiment."""
    return [
        config.get_raw_results_path(folder, "001a__20260803T090154398Z"),
        config.get_manifest_path(folder, "001a__20260803T090154398Z"),
        config.get_normalized_path(folder, "001a"),
        config.get_aggregate_path(folder, "001a"),
    ]


def test_repo_root_is_the_project_directory() -> None:
    assert config.REPO_ROOT.is_dir()
    assert (config.REPO_ROOT / "pyproject.toml").is_file()


def test_sha256_text_is_deterministic() -> None:
    assert config.sha256_text("hello") == config.sha256_text("hello")
    assert config.sha256_text("hello") != config.sha256_text("world")


def test_every_stage_writes_under_its_own_experiment() -> None:
    """A path that dropped its folder still round-trips, so nothing else catches it."""
    for folder in ("e001_fruit", "e002_other"):
        experiment_dir = config.get_experiment_data_dir(folder)
        for stage_path in _stage_paths(folder):
            assert experiment_dir in stage_path.parents
