"""Smoke test: the package imports and exposes a version string."""

import llmango


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(llmango.__version__, str)
    assert llmango.__version__
