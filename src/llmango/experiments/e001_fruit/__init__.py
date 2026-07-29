"""Experiment 001: how a model picks one fruit, and how language shifts the pick.

Deliberately does not import .charts, which is what keeps matplotlib off the
path of every command but analyze.
"""

from llmango.experiments.e001_fruit.experiment import FRUIT

__all__ = ["FRUIT"]
