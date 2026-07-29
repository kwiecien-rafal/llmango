"""Tests for experiment 001's declared charts: what each one names and reads."""

import subprocess
import sys

from llmango.aggregate import Aggregate, Distribution
from llmango.experiments.e001_fruit.charts import CHARTS, order_effect


def _aggregate(question_id: str, langs: dict[str, Distribution]) -> Aggregate:
    return {"question_id": question_id, "distributions": {"FruitChoice": langs}}


def _cell(counts: dict[str, int]) -> Distribution:
    total = sum(counts.values())
    return {"n": total, "counts": counts, "other_share": 0.0}


def test_every_chart_is_named_once_and_declares_what_it_reads() -> None:
    declared = {chart.name: chart.questions for chart in CHARTS}

    assert declared == {
        "language_drift": ("001a",),
        "order_effect": ("001a", "001b"),
        "schema_effect": ("001d",),
    }
    assert len(declared) == len(CHARTS)


def test_the_order_comparison_reads_one_language_from_two_questions() -> None:
    """001b exists only to be read against 001a, so the chart labels by question."""
    drawn = order_effect(
        {
            "001a": _aggregate("001a", {"en": _cell({"apple": 3, "banana": 1})}),
            "001b": _aggregate("001b", {"en": _cell({"apple": 1, "banana": 3})}),
        }
    )

    assert drawn.columns == ["001a order", "001b order"]
    assert "option order" in drawn.title


def test_the_experiment_package_does_not_pull_in_matplotlib() -> None:
    """Importing an experiment must not cost every command matplotlib's import.

    Run in a subprocess because the test session has already imported it. The
    charts module is where matplotlib is reached, and analyze imports that lazily.
    """
    probe = (
        "import sys; import llmango.experiments.e001_fruit; "
        "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0
