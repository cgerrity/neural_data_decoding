"""Distributional convergence metrics for seed-ensemble parity checks.

Bit-exact reproducibility across the MATLAB/Python boundary is explicitly **not**
a parity goal (ADR 001): weight initialization, RNG stream ordering, and the
order of floating-point reductions all differ between the two runtimes. Two runs
that are both *correct* will still land on different final accuracies.

What *is* checkable is that the two pipelines converge to the same *distribution*
of final accuracies over a seed ensemble. Run K seeds in each runtime, then ask
whether the two samples are statistically consistent. This module provides the
pure comparators for that check — no I/O, no MATLAB, no training — so they unit
test in microseconds and compose into the MATLAB-gated end-to-end harness:

- :func:`summarize_runs` — reduce a list of final accuracies to summary stats.
- :func:`within_n_sigma` — is a single reference value inside ``mean ± n·std`` of
  a sample? (e.g. is the MATLAB mean inside the Python ensemble's 2σ band?)
- :func:`means_within_n_sigma` — are two ensembles' means consistent to within
  ``n`` standard errors of their difference? (symmetric ensemble-vs-ensemble)
- :func:`ks_2sample` — two-sample Kolmogorov–Smirnov distance and p-value.

The 2σ convention follows the migration spec (``docs/PLAN.md``: "within 2σ across
5 seeds").

Examples
--------
>>> from neural_data_decoding.training.convergence_metrics import within_n_sigma
>>> within_n_sigma(0.71, [0.70, 0.72, 0.68, 0.74, 0.71], n_sigma=2.0)
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class EnsembleSummary:
    """Summary statistics of a seed ensemble of final accuracies.

    Parameters
    ----------
    n
        Number of runs in the ensemble.
    mean
        Sample mean of the final accuracies.
    std
        Sample standard deviation (``ddof=1``; ``0.0`` when ``n == 1``).
    sem
        Standard error of the mean (``std / sqrt(n)``; ``0.0`` when ``n == 1``).
    minimum
        Smallest final accuracy observed.
    maximum
        Largest final accuracy observed.
    """

    n: int
    mean: float
    std: float
    sem: float
    minimum: float
    maximum: float


def summarize_runs(values: Sequence[float]) -> EnsembleSummary:
    """Reduce a seed ensemble of final accuracies to summary statistics.

    Parameters
    ----------
    values
        Final accuracies (or any scalar convergence metric), one per seed.
        Must contain at least one value.

    Returns
    -------
    EnsembleSummary
        Count, mean, sample standard deviation (``ddof=1``), standard error of
        the mean, minimum, and maximum. For a single value the ``std`` and
        ``sem`` are ``0.0`` (an ensemble of one has no spread).

    Raises
    ------
    ValueError
        If ``values`` is empty.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("summarize_runs requires at least one value")
    n = int(arr.size)
    # ddof=1 is undefined for n == 1 (division by zero); an ensemble of one has
    # no measurable spread, so report zero rather than NaN.
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / float(np.sqrt(n)) if n > 1 else 0.0
    return EnsembleSummary(
        n=n,
        mean=float(arr.mean()),
        std=std,
        sem=sem,
        minimum=float(arr.min()),
        maximum=float(arr.max()),
    )


def within_n_sigma(
    reference: float,
    values: Sequence[float],
    *,
    n_sigma: float = 2.0,
    zero_std_atol: float = 1e-9,
) -> bool:
    """Test whether a reference value lies inside a sample's ``mean ± n·std`` band.

    This is the asymmetric check: treat ``values`` as the empirical distribution
    (e.g. the Python seed ensemble) and ask whether a single ``reference`` (e.g.
    the MATLAB ensemble mean, or a published number) is a plausible draw from it.

    Parameters
    ----------
    reference
        The single value to test for membership in the band.
    values
        The sample defining the band. Must contain at least one value.
    n_sigma
        Half-width of the acceptance band in sample standard deviations. Default
        ``2.0`` (the migration spec's convention).
    zero_std_atol
        When the sample has zero spread (all values equal, or ``n == 1``), the
        band collapses to a point; ``reference`` passes only if it is within this
        absolute tolerance of the mean. Default ``1e-9``.

    Returns
    -------
    bool
        ``True`` if ``reference`` is inside the band, else ``False``.

    Raises
    ------
    ValueError
        If ``values`` is empty.
    """
    summary = summarize_runs(values)
    if summary.std <= zero_std_atol:
        return abs(reference - summary.mean) <= zero_std_atol
    return abs(reference - summary.mean) <= n_sigma * summary.std


def means_within_n_sigma(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    n_sigma: float = 2.0,
    zero_std_atol: float = 1e-9,
) -> bool:
    """Test whether two ensembles' means agree to within ``n`` standard errors.

    Symmetric counterpart to :func:`within_n_sigma`: given two seed ensembles
    (e.g. Python vs MATLAB), ask whether their means differ by no more than
    ``n_sigma`` times the standard error of the difference,
    ``sqrt(sem_a**2 + sem_b**2)``. This is the natural test when *both* sides are
    stochastic and neither is a fixed ground truth.

    Parameters
    ----------
    sample_a, sample_b
        The two ensembles to compare. Each must contain at least one value.
    n_sigma
        Acceptance half-width in standard errors of the difference. Default
        ``2.0``.
    zero_std_atol
        When the combined standard error is zero (both ensembles have no spread),
        the means must agree to within this absolute tolerance. Default ``1e-9``.

    Returns
    -------
    bool
        ``True`` if the means are consistent, else ``False``.

    Raises
    ------
    ValueError
        If either sample is empty.
    """
    a = summarize_runs(sample_a)
    b = summarize_runs(sample_b)
    combined_sem = float(np.hypot(a.sem, b.sem))
    mean_gap = abs(a.mean - b.mean)
    if combined_sem <= zero_std_atol:
        return mean_gap <= zero_std_atol
    return mean_gap <= n_sigma * combined_sem


@dataclass(frozen=True)
class KSResult:
    """Result of a two-sample Kolmogorov–Smirnov test.

    Parameters
    ----------
    statistic
        The KS statistic: the maximum absolute difference between the two
        empirical CDFs (in ``[0, 1]``; smaller means more similar).
    pvalue
        The two-sided p-value. A *large* p-value means the samples are consistent
        with being drawn from the same distribution; a small one (``< alpha``)
        rejects that null.
    """

    statistic: float
    pvalue: float


def ks_2sample(sample_a: Sequence[float], sample_b: Sequence[float]) -> KSResult:
    """Two-sample Kolmogorov–Smirnov test between two seed ensembles.

    A distribution-shape check that complements the mean-based
    :func:`means_within_n_sigma`: it is sensitive to differences in spread and
    shape, not just location.

    Parameters
    ----------
    sample_a, sample_b
        The two ensembles to compare. Each must contain at least one value.

    Returns
    -------
    KSResult
        The KS ``statistic`` and two-sided ``pvalue``.

    Raises
    ------
    ValueError
        If either sample is empty.
    """
    if len(sample_a) == 0 or len(sample_b) == 0:
        raise ValueError("ks_2sample requires non-empty samples")
    # scipy types ks_2samp's return as an opaque ``_`` class; it is in fact a
    # named tuple exposing .statistic / .pvalue at runtime.
    result = cast(
        Any,
        ks_2samp(np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)),
    )
    return KSResult(statistic=float(result.statistic), pvalue=float(result.pvalue))


__all__ = [
    "EnsembleSummary",
    "KSResult",
    "ks_2sample",
    "means_within_n_sigma",
    "summarize_runs",
    "within_n_sigma",
]
