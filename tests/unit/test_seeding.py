"""Tests for :mod:`neural_data_decoding.utils.seeding`."""

from __future__ import annotations

import random

import numpy as np

from neural_data_decoding.utils.seeding import set_global_seed


def test_set_global_seed_returns_seed() -> None:
    """The function returns the seed it applied."""
    assert set_global_seed(42) == 42


def test_set_global_seed_makes_random_deterministic() -> None:
    """Two calls to ``set_global_seed(s)`` yield identical ``random`` sequences."""
    set_global_seed(7)
    a = [random.random() for _ in range(5)]
    set_global_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_global_seed_makes_numpy_deterministic() -> None:
    """Two calls to ``set_global_seed(s)`` yield identical NumPy sequences."""
    set_global_seed(7)
    a = np.random.rand(5)
    set_global_seed(7)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_diverge() -> None:
    """Different seeds produce different sequences."""
    set_global_seed(1)
    a = np.random.rand(20)
    set_global_seed(2)
    b = np.random.rand(20)
    assert not np.array_equal(a, b)
