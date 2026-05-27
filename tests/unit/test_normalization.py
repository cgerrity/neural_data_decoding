"""Tests for :mod:`neural_data_decoding.data.normalization`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_data_decoding.data.normalization import (
    list_recipes,
    register,
    select_normalization,
)


# ───────────────────────── Fixtures ─────────────────────────


@pytest.fixture()
def simple_table() -> pd.DataFrame:
    """A normalization table with two channels in one area.

    Channel/Area indices are 1-based to match MATLAB convention.
    """
    return pd.DataFrame(
        {
            "Area": [1, 1],
            "Channel": [1, 2],
            "Mean": [0.0, 1.0],
            "STD": [1.0, 2.0],
            "Min": [-3.0, -5.0],
            "Max": [3.0, 7.0],
        }
    )


@pytest.fixture()
def simple_data() -> np.ndarray:
    """A ``(channels, samples, areas)`` tensor for the simple_table fixture."""
    return np.array(
        [
            [[0.0], [1.0], [2.0], [-1.0]],   # channel 1
            [[1.0], [3.0], [5.0], [-1.0]],   # channel 2
        ],
        dtype=np.float64,
    )


# ───────────────────────── Registry ─────────────────────────


def test_list_recipes_includes_none_and_optimal() -> None:
    """The registry always contains 'None' and the Optimal recipe."""
    recipes = list_recipes()
    assert "None" in recipes
    assert (
        "Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered - Range 0.5"
        in recipes
    )


def test_register_rejects_duplicate_names() -> None:
    """Re-registering an existing recipe raises ``ValueError``."""
    with pytest.raises(ValueError, match="already registered"):

        @register("None")
        def _dup(_data: np.ndarray, _table: pd.DataFrame) -> np.ndarray:
            return _data


# ───────────────────────── Passthrough ─────────────────────────


def test_none_recipe_is_passthrough(simple_data: np.ndarray, simple_table: pd.DataFrame) -> None:
    """The 'None' recipe returns the input unchanged."""
    out = select_normalization(simple_data, simple_table, "None")
    np.testing.assert_array_equal(out, simple_data)


def test_missing_table_falls_back_to_passthrough(simple_data: np.ndarray) -> None:
    """An empty or None normalization table degrades to passthrough."""
    out_none = select_normalization(simple_data, None, "Channel - Z-Score")
    np.testing.assert_array_equal(out_none, simple_data)

    out_empty = select_normalization(simple_data, pd.DataFrame(), "Channel - Z-Score")
    np.testing.assert_array_equal(out_empty, simple_data)


# ───────────────────────── Optimal recipe ─────────────────────────


def test_optimal_recipe_returns_correct_shape(
    simple_data: np.ndarray, simple_table: pd.DataFrame
) -> None:
    """Applying the Optimal recipe preserves tensor shape."""
    out = select_normalization(
        simple_data,
        simple_table,
        "Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered - Range 0.5",
    )
    assert out.shape == simple_data.shape
    assert out.dtype == np.float64


def test_optimal_recipe_is_deterministic(
    simple_data: np.ndarray, simple_table: pd.DataFrame
) -> None:
    """The Optimal recipe is a deterministic function of (data, table)."""
    recipe = "Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered - Range 0.5"
    a = select_normalization(simple_data, simple_table, recipe)
    b = select_normalization(simple_data, simple_table, recipe)
    np.testing.assert_array_equal(a, b)


def test_optimal_recipe_rejects_wrong_shape(simple_table: pd.DataFrame) -> None:
    """Passing data without 3 axes raises ``ValueError``."""
    bad = np.zeros((2, 4))  # only 2 axes
    with pytest.raises(ValueError, match="channels, samples, areas"):
        select_normalization(
            bad,
            simple_table,
            "Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered - Range 0.5",
        )


def test_optimal_recipe_rejects_missing_columns(simple_data: np.ndarray) -> None:
    """Tables missing required columns raise ``ValueError``."""
    incomplete = pd.DataFrame({"Area": [1], "Channel": [1], "Mean": [0.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        select_normalization(
            simple_data,
            incomplete,
            "Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered - Range 0.5",
        )


# ───────────────────────── Unknown recipes ─────────────────────────


def test_unknown_recipe_raises_keyerror(simple_data: np.ndarray, simple_table: pd.DataFrame) -> None:
    """An unregistered recipe name raises ``KeyError`` with a useful message."""
    with pytest.raises(KeyError, match="not registered"):
        select_normalization(simple_data, simple_table, "Pumpkin Spice Recipe")


def test_stub_recipes_raise_notimplemented(
    simple_data: np.ndarray, simple_table: pd.DataFrame
) -> None:
    """Recipes registered as Milestone-CC stubs raise ``NotImplementedError``."""
    with pytest.raises(NotImplementedError, match="Milestone CC"):
        select_normalization(simple_data, simple_table, "Channel - Z-Score")
