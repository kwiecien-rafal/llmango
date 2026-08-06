"""The statistics every stage derives from, computed over category counts alone."""

from collections.abc import Callable, Sequence

import numpy as np

CONFIDENCE_Z = 1.959963984540054
PERMUTATIONS = 20_000
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 1

_LN2 = float(np.log(2.0))


def normalized_entropy(counts: Sequence[int], support: int) -> float:
    """Share of a uniform choice's entropy an arm reached, 1.0 being a fair die."""
    ceiling = _log2(support)
    if ceiling <= 0.0:
        return 0.0

    return round(min(_entropy_bits(counts) / ceiling, 1.0), 4)


def effective_choices(counts: Sequence[int], support: int) -> float:
    """How many equally likely options an arm behaved as though it chose among."""
    return round(min(2.0 ** _entropy_bits(counts), float(support)), 4)


def distance_from_uniform(counts: Sequence[int], support: int) -> float:
    """Total variation between an arm's answers and an even spread over support."""
    total = sum(counts)
    if not total or support <= 0:
        return 0.0

    shares = [count / total for count in counts]
    unpicked = max(support - len(shares), 0)
    spread = 1.0 / support
    distance = sum(abs(share - spread) for share in shares) + unpicked * spread

    return round(distance / 2.0, 4)


def jensen_shannon(left: Sequence[int], right: Sequence[int]) -> float:
    """Jensen-Shannon divergence in bits between two arms over one category order."""
    first, second = _shares(left), _shares(right)
    middle = [(one + other) / 2.0 for one, other in zip(first, second, strict=True)]
    divergence = (
        _relative_entropy(first, middle) + _relative_entropy(second, middle)
    ) / 2.0

    return round(max(divergence, 0.0), 4)


def wilson_interval(count: int, total: int) -> tuple[float, float]:
    """The 95% Wilson score interval around one category's share of an arm."""
    if total <= 0:
        return (0.0, 0.0)

    share = count / total
    z_squared = CONFIDENCE_Z**2
    denominator = 1.0 + z_squared / total
    centre = (share + z_squared / (2 * total)) / denominator
    spread = CONFIDENCE_Z * np.sqrt(
        share * (1.0 - share) / total + z_squared / (4 * total**2)
    )
    half_width = float(spread) / denominator

    return (
        round(max(centre - half_width, 0.0), 4),
        round(min(centre + half_width, 1.0), 4),
    )


def entropy_interval(
    counts: Sequence[int], support: int, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    """The 95% bootstrap interval around one arm's share of uniform entropy."""
    return _bootstrap(
        [counts], lambda arms: normalized_entropy(arms[0], support), draws
    )


def effective_choices_interval(
    counts: Sequence[int], support: int, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    """The 95% bootstrap interval around how many options an arm chose among."""
    return _bootstrap([counts], lambda arms: effective_choices(arms[0], support), draws)


def homogeneity_pvalue(
    arms: Sequence[Sequence[int]], seed: int, permutations: int = PERMUTATIONS
) -> float:
    """Monte-Carlo probability that arms this different arose from one distribution."""
    observed = np.asarray(arms, dtype=np.int64)
    if observed.ndim != 2 or observed.shape[0] < 2 or not observed.sum():
        return 1.0

    sizes = observed.sum(axis=1)
    pooled = observed.sum(axis=0) / observed.sum()
    generator = np.random.default_rng(seed)
    resampled = np.stack(
        [generator.multinomial(size, pooled, size=permutations) for size in sizes],
        axis=1,
    )
    extreme = int((_chi_square(resampled) >= _chi_square(observed[None, :, :])).sum())

    return round((1 + extreme) / (1 + permutations), 4)


def _bootstrap(
    arms: Sequence[Sequence[int]],
    statistic: Callable[[list[list[int]]], float],
    draws: int,
) -> tuple[float, float]:
    """Re-run every arm from its own counts and spread the statistic over the draws."""
    if any(sum(arm) <= 0 for arm in arms):
        return (0.0, 0.0)

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = [
        generator.multinomial(sum(arm), _shares(arm), size=draws) for arm in arms
    ]
    values = [
        statistic([arm[draw].tolist() for arm in resampled]) for draw in range(draws)
    ]
    low, high = np.percentile(values, [2.5, 97.5])

    return (round(float(low), 4), round(float(high), 4))


def _entropy_bits(counts: Sequence[int]) -> float:
    """Shannon entropy in bits, Miller-Madow corrected for small-sample bias."""
    total = sum(counts)
    if total <= 0:
        return 0.0

    shares = [count / total for count in counts if count > 0]
    plugin = -sum(share * _log2(share) for share in shares)
    correction = (len(shares) - 1) / (2 * total * _LN2)

    return plugin + correction


def _shares(counts: Sequence[int]) -> list[float]:
    """One arm's counts as shares, all zero when the arm answered nothing."""
    total = sum(counts)
    if not total:
        return [0.0] * len(counts)

    return [count / total for count in counts]


def _relative_entropy(shares: Sequence[float], reference: Sequence[float]) -> float:
    """Kullback-Leibler divergence in bits, skipping the terms that vanish."""
    paired = zip(shares, reference, strict=True)

    return sum(
        share * _log2(share / other)
        for share, other in paired
        if share > 0 and other > 0
    )


def _chi_square(tables: np.ndarray) -> np.ndarray:
    """Pearson's statistic for a stack of arm-by-category tables, one per table."""
    rows = tables.sum(axis=2)[:, :, None]
    columns = tables.sum(axis=1)[:, None, :]
    totals = tables.sum(axis=(1, 2))[:, None, None]
    expected = np.divide(
        rows * columns,
        totals,
        out=np.zeros(tables.shape, dtype=float),
        where=totals > 0,
    )
    residuals = np.divide(
        (tables - expected) ** 2,
        expected,
        out=np.zeros_like(expected),
        where=expected > 0,
    )

    return residuals.sum(axis=(1, 2))


def _log2(value: float) -> float:
    """Base-2 logarithm as a plain float, zero for a value that cannot have one."""
    if value <= 0.0:
        return 0.0

    return float(np.log2(value))
