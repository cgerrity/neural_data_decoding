"""Tests for :mod:`neural_data_decoding.data.stratification`.

These are pure unit tests on the Python implementation. The MATLAB-parity
tests that compare strata IDs against an actual ``cgg_getKFoldPartitions``
output live under ``tests/parity/test_stratification.py`` and are skipped
until ``scripts/prepare_golden_fixtures.py`` is run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_data_decoding.data.stratification import (
    DEFAULT_DATA_NUMBER_COLUMN,
    assign_folds,
    stratify,
)


# ───────────────────────── Fixtures ─────────────────────────


@pytest.fixture()
def small_identifiers() -> pd.DataFrame:
    """A 10-trial table covering two dimensions and a binary correctness flag."""
    return pd.DataFrame(
        {
            "DataNumber":    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Dimension 1":   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "Correct Trial": [1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
            "Gain":          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


# ───────────────────────── stratify — validation ─────────────────────────


def test_missing_data_number_column_raises(small_identifiers: pd.DataFrame) -> None:
    """Identifiers without the data-number column are rejected."""
    bad = small_identifiers.drop(columns=["DataNumber"])
    with pytest.raises(ValueError, match="DataNumber"):
        stratify(bad, all_split_names=[["Dimension 1"]], num_folds=5)


def test_duplicate_data_numbers_raise(small_identifiers: pd.DataFrame) -> None:
    """A duplicated trial ID is a programming error and must be flagged."""
    bad = small_identifiers.copy()
    bad.loc[0, "DataNumber"] = bad.loc[1, "DataNumber"]
    with pytest.raises(ValueError, match="unique per trial"):
        stratify(bad, all_split_names=[["Dimension 1"]], num_folds=5)


def test_missing_split_column_raises(small_identifiers: pd.DataFrame) -> None:
    """Stratification levels referencing absent columns are rejected."""
    with pytest.raises(KeyError, match="missing stratification columns"):
        stratify(
            small_identifiers,
            all_split_names=[["Dimension 1"], ["NotARealColumn"]],
            num_folds=5,
        )


# ───────────────────────── stratify — behaviour ─────────────────────────


def test_returns_one_group_id_per_trial(small_identifiers: pd.DataFrame) -> None:
    """Output array has the same length as the identifier DataFrame."""
    groups = stratify(small_identifiers, all_split_names=[["Dimension 1"]], num_folds=5)
    assert groups.shape == (len(small_identifiers),)
    assert groups.dtype == np.int64


def test_empty_split_hierarchy_yields_single_group(small_identifiers: pd.DataFrame) -> None:
    """With no levels, every trial is assigned to a single group (ID 1)."""
    groups = stratify(small_identifiers, all_split_names=[], num_folds=5)
    np.testing.assert_array_equal(groups, np.ones(len(small_identifiers), dtype=np.int64))


def test_small_categories_collapse_into_maintain_group(small_identifiers: pd.DataFrame) -> None:
    """Categories with ``<= num_folds`` trials are merged into a single leaf group.

    Mirrors the MATLAB behavior at ``cgg_procSplitIntoGroups.m`` line 62
    where ``MaintainSplit`` concatenates all small-category DataNumbers.
    """
    # With num_folds=5, both Dimension-1 categories have 5 trials each
    # (counts 5 and 5) — both fall into the "maintain" branch and collapse
    # into a single leaf.
    groups = stratify(small_identifiers, all_split_names=[["Dimension 1"]], num_folds=5)
    assert len(np.unique(groups)) == 1


def test_large_categories_recurse(small_identifiers: pd.DataFrame) -> None:
    """Categories larger than ``num_folds`` recurse into the next level."""
    # With num_folds=3, both Dimension-1 categories (5 trials each) recurse
    # to the next level. Each Dimension-1 group is then partitioned by
    # Correct Trial; the resulting per-(dim, correct) buckets are all small
    # enough to be leaves.
    groups = stratify(
        small_identifiers,
        all_split_names=[["Dimension 1"], ["Correct Trial"]],
        num_folds=3,
    )
    # Two dimensions × variable correct-trial counts → multiple distinct groups.
    assert len(np.unique(groups)) > 1


def test_stratification_is_deterministic(small_identifiers: pd.DataFrame) -> None:
    """Running twice on the same input yields bitwise-identical group IDs."""
    levels = [["Dimension 1"], ["Correct Trial"]]
    a = stratify(small_identifiers, all_split_names=levels, num_folds=3)
    b = stratify(small_identifiers, all_split_names=levels, num_folds=3)
    np.testing.assert_array_equal(a, b)


def test_row_order_does_not_affect_group_assignment(small_identifiers: pd.DataFrame) -> None:
    """Permuting input rows must not change which trial belongs to which group.

    The stratification depends on the trial's ``DataNumber`` value, not its
    row position. Two identical tables (one shuffled) must yield identical
    per-DataNumber group assignments.
    """
    levels = [["Dimension 1"], ["Correct Trial"]]
    shuffled = small_identifiers.sample(frac=1, random_state=0).reset_index(drop=True)

    groups_original = stratify(small_identifiers, all_split_names=levels, num_folds=3)
    groups_shuffled = stratify(shuffled, all_split_names=levels, num_folds=3)

    # Compare via (DataNumber → GroupID) mapping.
    map_original = dict(zip(small_identifiers["DataNumber"], groups_original))
    map_shuffled = dict(zip(shuffled["DataNumber"], groups_shuffled))
    assert map_original == map_shuffled


def test_custom_data_number_column(small_identifiers: pd.DataFrame) -> None:
    """The ``data_number_column`` argument lets callers use a different column name."""
    renamed = small_identifiers.rename(columns={"DataNumber": "TrialID"})
    groups = stratify(
        renamed,
        all_split_names=[["Dimension 1"]],
        num_folds=5,
        data_number_column="TrialID",
    )
    assert groups.shape == (len(renamed),)


def test_default_data_number_column_constant() -> None:
    """The module exposes the expected default column name."""
    assert DEFAULT_DATA_NUMBER_COLUMN == "DataNumber"


# ───────────────────────── assign_folds ─────────────────────────


def test_assign_folds_returns_one_id_per_trial(small_identifiers: pd.DataFrame) -> None:
    """The output array length matches the trial count."""
    groups = stratify(small_identifiers, all_split_names=[["Dimension 1"]], num_folds=5)
    folds = assign_folds(groups, num_folds=5)
    assert folds.shape == groups.shape
    assert folds.dtype == np.int64


def test_assign_folds_uses_one_based_indexing(small_identifiers: pd.DataFrame) -> None:
    """Fold IDs are in ``1..num_folds`` (matches MATLAB convention)."""
    groups = stratify(
        small_identifiers,
        all_split_names=[["Dimension 1"], ["Correct Trial"]],
        num_folds=3,
    )
    folds = assign_folds(groups, num_folds=3)
    assert folds.min() >= 1
    assert folds.max() <= 3


def test_assign_folds_is_deterministic_with_same_seed(small_identifiers: pd.DataFrame) -> None:
    """Two calls with the same seed produce identical fold assignments."""
    groups = stratify(
        small_identifiers,
        all_split_names=[["Dimension 1"], ["Correct Trial"]],
        num_folds=3,
    )
    a = assign_folds(groups, num_folds=3, seed=42)
    b = assign_folds(groups, num_folds=3, seed=42)
    np.testing.assert_array_equal(a, b)


def test_assign_folds_differs_across_seeds() -> None:
    """Different seeds typically produce different fold assignments.

    With small toy data and few folds we could in principle get the same
    permutation by chance, so this test uses a larger group count where
    collisions are exceedingly unlikely.
    """
    rng = np.random.default_rng(0)
    big_groups = rng.integers(low=1, high=50, size=200)
    a = assign_folds(big_groups, num_folds=5, seed=1)
    b = assign_folds(big_groups, num_folds=5, seed=2)
    assert not np.array_equal(a, b)


def test_assign_folds_keeps_groups_intact(small_identifiers: pd.DataFrame) -> None:
    """Every trial in a given group lands in the same fold."""
    groups = stratify(
        small_identifiers,
        all_split_names=[["Dimension 1"], ["Correct Trial"]],
        num_folds=3,
    )
    folds = assign_folds(groups, num_folds=3, seed=0)

    for group_id in np.unique(groups):
        member_folds = folds[groups == group_id]
        assert len(np.unique(member_folds)) == 1, (
            f"Group {group_id} was split across folds {np.unique(member_folds)}"
        )


def test_assign_folds_rejects_invalid_num_folds() -> None:
    """``num_folds=0`` (or negative) is a programming error."""
    with pytest.raises(ValueError, match="num_folds"):
        assign_folds(np.array([1, 2, 3]), num_folds=0)


def test_assign_folds_rejects_negative_num_folds() -> None:
    """Negative ``num_folds`` is also rejected."""
    with pytest.raises(ValueError, match="num_folds"):
        assign_folds(np.array([1, 2, 3]), num_folds=-1)
