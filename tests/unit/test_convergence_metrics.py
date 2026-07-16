"""Unit tests for the distributional convergence comparators.

These are pure-function tests — no training, no MATLAB — so they pin the
statistical behavior the seed-ensemble parity harness relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_data_decoding.training.convergence_metrics import (
    EnsembleSummary,
    KSResult,
    ks_2sample,
    means_within_n_sigma,
    summarize_runs,
    within_n_sigma,
)


def test_summarize_runs_matches_numpy() -> None:
    """Summary stats agree with NumPy (ddof=1 std) on a small ensemble."""
    values = [0.70, 0.72, 0.68, 0.74, 0.71]
    s = summarize_runs(values)
    assert isinstance(s, EnsembleSummary)
    assert s.n == 5
    assert s.mean == pytest.approx(float(np.mean(values)))
    assert s.std == pytest.approx(float(np.std(values, ddof=1)))
    assert s.sem == pytest.approx(s.std / np.sqrt(5))
    assert s.minimum == pytest.approx(0.68)
    assert s.maximum == pytest.approx(0.74)


def test_summarize_runs_single_value_has_zero_spread() -> None:
    """An ensemble of one reports zero std/sem rather than NaN."""
    s = summarize_runs([0.5])
    assert s.n == 1
    assert s.std == 0.0
    assert s.sem == 0.0
    assert s.mean == 0.5


def test_summarize_runs_rejects_empty() -> None:
    """Empty input is a programming error, not a silent zero."""
    with pytest.raises(ValueError):
        summarize_runs([])


def test_within_n_sigma_accepts_value_inside_band() -> None:
    """A reference near the mean falls inside the 2σ band."""
    values = [0.70, 0.72, 0.68, 0.74, 0.71]
    assert within_n_sigma(0.71, values, n_sigma=2.0)


def test_within_n_sigma_rejects_far_value() -> None:
    """A reference many sigma away is rejected."""
    values = [0.70, 0.72, 0.68, 0.74, 0.71]
    assert not within_n_sigma(0.95, values, n_sigma=2.0)


def test_within_n_sigma_zero_spread_requires_exact_match() -> None:
    """With no spread the band is a point at the mean (within tolerance)."""
    assert within_n_sigma(0.5, [0.5, 0.5, 0.5])
    assert not within_n_sigma(0.5001, [0.5, 0.5, 0.5])


def test_within_n_sigma_boundary_is_inclusive() -> None:
    """A reference exactly n_sigma away is accepted (band is closed)."""
    values = [0.0, 2.0]  # mean 1.0, std (ddof=1) = sqrt(2)
    std = float(np.std(values, ddof=1))
    assert within_n_sigma(1.0 + std, values, n_sigma=1.0)
    assert not within_n_sigma(1.0 + 1.0001 * std, values, n_sigma=1.0)


def test_means_within_n_sigma_consistent_ensembles() -> None:
    """Two overlapping ensembles have consistent means."""
    a = [0.70, 0.72, 0.68, 0.74, 0.71]
    b = [0.71, 0.69, 0.73, 0.70, 0.72]
    assert means_within_n_sigma(a, b, n_sigma=2.0)


def test_means_within_n_sigma_rejects_separated_ensembles() -> None:
    """Well-separated, tight ensembles fail the mean-consistency test."""
    a = [0.10, 0.11, 0.09, 0.10, 0.10]
    b = [0.90, 0.91, 0.89, 0.90, 0.90]
    assert not means_within_n_sigma(a, b, n_sigma=2.0)


def test_means_within_n_sigma_is_symmetric() -> None:
    """Swapping the arguments does not change the verdict."""
    a = [0.70, 0.72, 0.68, 0.74, 0.71]
    b = [0.60, 0.62, 0.58, 0.64, 0.61]
    assert means_within_n_sigma(a, b, n_sigma=2.0) == means_within_n_sigma(
        b, a, n_sigma=2.0
    )


def test_ks_2sample_same_distribution_has_high_pvalue() -> None:
    """Identical samples yield statistic 0 and p-value 1."""
    sample = [0.1, 0.2, 0.3, 0.4, 0.5]
    r = ks_2sample(sample, sample)
    assert isinstance(r, KSResult)
    assert r.statistic == pytest.approx(0.0)
    assert r.pvalue == pytest.approx(1.0)


def test_ks_2sample_disjoint_distributions_reject() -> None:
    """Disjoint supports give statistic 1 and a small p-value."""
    a = [0.0, 0.01, 0.02, 0.03]
    b = [0.97, 0.98, 0.99, 1.0]
    r = ks_2sample(a, b)
    assert r.statistic == pytest.approx(1.0)
    assert r.pvalue < 0.05


def test_ks_2sample_rejects_empty() -> None:
    """Empty samples are rejected explicitly."""
    with pytest.raises(ValueError):
        ks_2sample([], [0.1])
