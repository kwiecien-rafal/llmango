"""Tests for the statistics layer: entropy, spread, intervals and the shift test."""

import math

import pytest

from llmango.stats import (
    _entropy_bits,
    distance_from_uniform,
    effective_choices,
    homogeneity_pvalue,
    jensen_shannon,
    normalized_entropy,
    total_variation,
    wilson_interval,
)

_SUPPORT = 10


def test_an_even_spread_reaches_the_entropy_of_a_fair_die() -> None:
    assert normalized_entropy([100] * _SUPPORT, _SUPPORT) == 1.0


def test_one_answer_every_time_carries_no_entropy() -> None:
    assert normalized_entropy([250], _SUPPORT) == 0.0


def test_entropy_falls_as_an_arm_concentrates() -> None:
    even = normalized_entropy([50] * 4, _SUPPORT)
    lopsided = normalized_entropy([170, 10, 10, 10], _SUPPORT)

    assert 0.0 < lopsided < even < 1.0


def test_miller_madow_lifts_the_plugin_estimate() -> None:
    """The plug-in estimator is biased low at small n, which would overstate
    how deterministic a model looks. The correction is what the charts plot."""
    counts = [3, 3, 2, 2]
    total = sum(counts)
    plugin = -sum(
        (count / total) * math.log2(count / total) for count in counts if count
    )

    assert plugin == pytest.approx(1.9710, abs=1e-3)
    assert _entropy_bits(counts) == pytest.approx(
        plugin + 3 / (2 * total * math.log(2))
    )


def test_effective_choices_reads_entropy_as_a_number_of_options() -> None:
    assert effective_choices([100] * _SUPPORT, _SUPPORT) == _SUPPORT
    assert effective_choices([250], _SUPPORT) == 1.0


def test_effective_choices_never_exceeds_the_options_offered() -> None:
    assert effective_choices([1] * _SUPPORT, _SUPPORT) == _SUPPORT


def test_distance_from_uniform_spans_an_even_spread_to_one_answer() -> None:
    assert distance_from_uniform([100] * _SUPPORT, _SUPPORT) == 0.0
    assert distance_from_uniform([250], _SUPPORT) == 0.9


def test_distance_from_uniform_counts_the_categories_never_picked() -> None:
    """Six unpicked fruits are six categories' worth of missing mass, not zero
    terms to skip; dropping them would report an arm as more even than it is."""
    assert distance_from_uniform([25] * 4, _SUPPORT) == pytest.approx(0.6)


def test_total_variation_is_symmetric_and_bounded() -> None:
    left, right = [3, 1, 0], [0, 1, 3]

    assert total_variation(left, right) == total_variation(right, left)
    assert total_variation(left, left) == 0.0
    assert total_variation([4, 0], [0, 4]) == 1.0


def test_jensen_shannon_spans_identical_to_disjoint_arms() -> None:
    assert jensen_shannon([3, 1], [3, 1]) == 0.0
    assert jensen_shannon([4, 0], [0, 4]) == 1.0


def test_an_arm_that_answered_nothing_has_no_shape_to_compare() -> None:
    assert total_variation([0, 0], [3, 1]) == pytest.approx(0.5)
    assert normalized_entropy([], _SUPPORT) == 0.0


def test_a_wilson_interval_brackets_the_share_it_is_drawn_around() -> None:
    low, high = wilson_interval(30, 100)

    assert low < 0.3 < high


def test_a_wilson_interval_stays_inside_zero_and_one() -> None:
    """A normal approximation would put the bound on an unpicked category below
    zero and draw a cap hanging under the axis; Wilson is why these are Wilson."""
    assert wilson_interval(0, 20)[0] == 0.0
    assert wilson_interval(20, 20)[1] == 1.0
    assert wilson_interval(1, 1) == (pytest.approx(0.2065, abs=1e-3), 1.0)


def test_a_wilson_interval_narrows_as_the_sample_grows() -> None:
    small = wilson_interval(30, 100)
    large = wilson_interval(300, 1000)

    assert (large[1] - large[0]) < (small[1] - small[0])


def test_an_empty_arm_has_no_interval() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_arms_drawn_from_one_distribution_are_not_called_different() -> None:
    assert homogeneity_pvalue([[30, 30, 40], [30, 30, 40]], seed=1) > 0.9


def test_arms_that_disagree_sharply_are_called_different() -> None:
    assert homogeneity_pvalue([[90, 5, 5], [5, 5, 90]], seed=1) < 0.01


def test_the_shift_test_repeats_under_one_seed() -> None:
    """A p-value that moved every run would put a fresh number in every commit."""
    arms = [[20, 30, 50], [30, 30, 40]]

    assert homogeneity_pvalue(arms, seed=7) == homogeneity_pvalue(arms, seed=7)


def test_the_shift_test_never_returns_an_impossible_certainty() -> None:
    """The +1 is Davison and Hinkley's: 0 of 20,000 resamples is evidence that
    p is small, not evidence that it is zero."""
    assert homogeneity_pvalue([[500, 0], [0, 500]], seed=1, permutations=200) > 0.0


def test_a_single_arm_has_nothing_to_be_homogeneous_with() -> None:
    assert homogeneity_pvalue([[3, 1, 0]], seed=1) == 1.0
