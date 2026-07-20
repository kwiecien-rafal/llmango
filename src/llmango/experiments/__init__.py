"""Experiment modules. Importing this package registers every experiment."""

from llmango.experiments import favorite_fruit as favorite_fruit


def ensure_registered() -> None:
    """Guarantee every experiment module has been imported and registered."""
