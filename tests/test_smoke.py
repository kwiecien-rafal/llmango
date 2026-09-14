"""Smoke test: the package imports."""

import llmango


def test_package_imports() -> None:
    assert llmango.__name__ == "llmango"
