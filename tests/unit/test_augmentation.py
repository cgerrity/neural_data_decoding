"""Tests for :mod:`neural_data_decoding.data.augmentation`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neural_data_decoding.data.augmentation import (
    additive_augmentation_signal,
    generate_time_shift_samples,
)


# ───────────────────────── additive_augmentation_signal ─────────────────────────


def test_additive_signal_returns_correct_shape() -> None:
    """Output shape always matches the requested input shape."""
    rng = np.random.default_rng(0)
    out = additive_augmentation_signal(
        shape=(4, 100, 2),
        std_channel_offset=0.1,
        std_white_noise=0.05,
        std_random_walk=0.01,
        rng=rng,
    )
    assert out.shape == (4, 100, 2)
    assert out.dtype == np.float64


def test_all_disabled_returns_zero_tensor() -> None:
    """When every component is None/NaN, output is all zeros."""
    rng = np.random.default_rng(0)
    out = additive_augmentation_signal(
        shape=(3, 50, 1),
        std_channel_offset=None,
        std_white_noise=float("nan"),
        std_random_walk=None,
        rng=rng,
    )
    np.testing.assert_array_equal(out, np.zeros((3, 50, 1)))


def test_seeded_calls_are_reproducible() -> None:
    """Two calls with identically-seeded RNGs yield bit-identical output."""
    a = additive_augmentation_signal(
        shape=(4, 80, 3),
        std_channel_offset=0.1,
        std_white_noise=0.05,
        std_random_walk=0.01,
        rng=np.random.default_rng(42),
    )
    b = additive_augmentation_signal(
        shape=(4, 80, 3),
        std_channel_offset=0.1,
        std_white_noise=0.05,
        std_random_walk=0.01,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(a, b)


def test_consecutive_calls_with_same_rng_differ() -> None:
    """Re-randomization: consecutive calls on the same RNG give different output.

    This is the critical contract from Critical Note #7 — the Dataset must
    re-roll noise on every ``__getitem__``, not cache it.
    """
    rng = np.random.default_rng(0)
    first = additive_augmentation_signal(
        shape=(4, 80, 3),
        std_channel_offset=0.1,
        std_white_noise=0.05,
        std_random_walk=0.01,
        rng=rng,
    )
    second = additive_augmentation_signal(
        shape=(4, 80, 3),
        std_channel_offset=0.1,
        std_white_noise=0.05,
        std_random_walk=0.01,
        rng=rng,
    )
    assert not np.array_equal(first, second)


def test_channel_offset_is_constant_along_sample_axis() -> None:
    """Per-channel offset (alone) must be identical at every time sample."""
    out = additive_augmentation_signal(
        shape=(4, 80, 2),
        std_channel_offset=1.0,
        std_white_noise=None,
        std_random_walk=None,
        rng=np.random.default_rng(0),
        want_low_pass=False,
    )
    # Every sample within (channel, probe) should be identical.
    for channel in range(4):
        for probe in range(2):
            values = out[channel, :, probe]
            assert np.allclose(values, values[0])


def test_white_noise_has_expected_std_magnitude() -> None:
    """White noise (alone) has empirical std close to the requested value."""
    sigma = 0.5
    out = additive_augmentation_signal(
        shape=(10, 10_000, 1),
        std_channel_offset=None,
        std_white_noise=sigma,
        std_random_walk=None,
        rng=np.random.default_rng(0),
        want_low_pass=False,   # smoothing would reduce the std
    )
    # With 100k samples per realization, the empirical std should be very
    # close to sigma. Loose tolerance because Critical Note #7 says we
    # only need statistical, not bit-exact, parity.
    assert math.isclose(float(out.std()), sigma, rel_tol=0.05)


def test_random_walk_is_cumulative() -> None:
    """Random walk (alone) is the running sum of i.i.d. Gaussian increments."""
    rng_a = np.random.default_rng(0)
    out = additive_augmentation_signal(
        shape=(2, 100, 1),
        std_channel_offset=None,
        std_white_noise=None,
        std_random_walk=0.01,
        rng=rng_a,
        want_low_pass=False,   # smoothing breaks the strict cumsum check
    )
    # diff along sample axis should be i.i.d. Gaussian, NOT all the same
    diffs = np.diff(out, axis=1)
    assert not np.allclose(diffs, diffs[:, 0:1, :])  # not constant
    # The std of the diffs should match std_random_walk approximately.
    assert math.isclose(float(diffs.std()), 0.01, rel_tol=0.2)


def test_low_pass_smoothing_reduces_high_freq_noise() -> None:
    """With ``want_low_pass=True`` the smoothed signal has smaller std than raw."""
    rng = np.random.default_rng(0)
    raw = additive_augmentation_signal(
        shape=(4, 200, 1),
        std_channel_offset=None,
        std_white_noise=0.5,
        std_random_walk=None,
        rng=rng,
        want_low_pass=False,
    )
    rng_smooth = np.random.default_rng(0)
    smoothed = additive_augmentation_signal(
        shape=(4, 200, 1),
        std_channel_offset=None,
        std_white_noise=0.5,
        std_random_walk=None,
        rng=rng_smooth,
        want_low_pass=True,
    )
    assert smoothed.std() < raw.std()


def test_invalid_shape_raises() -> None:
    """A non-3D shape raises :class:`ValueError`."""
    with pytest.raises(ValueError, match="channels, samples, probes"):
        additive_augmentation_signal(
            shape=(4, 100),  # type: ignore[arg-type]
            std_channel_offset=None,
            std_white_noise=None,
            std_random_walk=None,
            rng=np.random.default_rng(0),
        )


# ───────────────────────── generate_time_shift_samples ─────────────────────────


def test_time_shift_disabled_returns_zeros() -> None:
    """``std_time_shift=None`` returns an all-zero shift tensor."""
    shifts = generate_time_shift_samples(
        num_channels=4,
        num_probes=2,
        num_windows=5,
        std_time_shift=None,
        sampling_frequency=1000.0,
        want_separate=True,
        rng=np.random.default_rng(0),
    )
    assert shifts.shape == (4, 2, 5)
    np.testing.assert_array_equal(shifts, np.zeros_like(shifts))


def test_time_shift_at_1khz_returns_ms_equivalent_samples() -> None:
    """At 1 kHz sampling, a ``std_time_shift=100`` yields shifts in ``[-100, 100]``."""
    shifts = generate_time_shift_samples(
        num_channels=4,
        num_probes=2,
        num_windows=20,
        std_time_shift=100.0,
        sampling_frequency=1000.0,
        want_separate=True,
        rng=np.random.default_rng(0),
    )
    assert shifts.dtype == np.int64
    assert shifts.min() >= -100
    assert shifts.max() <= 100


def test_separate_vs_shared_time_shift() -> None:
    """``want_separate=False`` produces a single shift broadcast everywhere."""
    rng = np.random.default_rng(0)
    shared = generate_time_shift_samples(
        num_channels=4,
        num_probes=2,
        num_windows=5,
        std_time_shift=100.0,
        sampling_frequency=1000.0,
        want_separate=False,
        rng=rng,
    )
    assert shared.shape == (4, 2, 5)
    # Every cell must be identical when ``want_separate=False``.
    assert np.all(shared == shared.flat[0])

    # Separate version should have variation across cells (with high probability).
    separate = generate_time_shift_samples(
        num_channels=4,
        num_probes=2,
        num_windows=5,
        std_time_shift=100.0,
        sampling_frequency=1000.0,
        want_separate=True,
        rng=rng,
    )
    assert len(np.unique(separate)) > 1


def test_time_shift_scales_with_sampling_frequency() -> None:
    """Halving the sampling rate halves the integer-sample shifts."""
    shifts_1k = generate_time_shift_samples(
        num_channels=2,
        num_probes=1,
        num_windows=1000,
        std_time_shift=100.0,
        sampling_frequency=1000.0,
        want_separate=True,
        rng=np.random.default_rng(0),
    )
    shifts_2k = generate_time_shift_samples(
        num_channels=2,
        num_probes=1,
        num_windows=1000,
        std_time_shift=100.0,
        sampling_frequency=2000.0,
        want_separate=True,
        rng=np.random.default_rng(0),
    )
    # At 2 kHz the same time-in-ms maps to half as many samples.
    assert abs(shifts_2k).max() <= abs(shifts_1k).max() // 2 + 1
