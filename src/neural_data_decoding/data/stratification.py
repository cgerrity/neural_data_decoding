"""Hierarchical stratified K-fold partitioning.

Ports the recursive splitter from ``cgg_procAssignGroups.m``,
``cgg_procAssignGroupsBySplit.m``, and ``cgg_procSplitIntoGroups.m``. The
algorithm walks a tree of stratification levels: at each level, trials are
partitioned by the cross-product of the level's columns. Partitions with
``<= num_folds`` trials become leaf groups; partitions with more recurse
to the next level. When the levels are exhausted, the remaining trial set
becomes a single leaf group.

The result is a flat list of trial groups (one per leaf) plus a per-trial
group-ID array. The group IDs are then mapped to K folds by
:func:`assign_folds`.

**Parity contract.** For identical inputs (same identifier DataFrame, same
``all_split_names`` hierarchy, same ``num_folds``), this implementation
must produce the **exact same per-trial group IDs** as the MATLAB
reference. Fold assignment is a separate concern with its own parity
boundary — see :func:`assign_folds`.

Examples
--------
>>> import pandas as pd
>>> identifiers = pd.DataFrame({
...     "DataNumber":   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
...     "Dimension 1":  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
...     "Correct Trial":[1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
... })
>>> levels = [["Dimension 1"], ["Correct Trial"]]
>>> groups = stratify(identifiers, all_split_names=levels, num_folds=5)
>>> groups.shape
(10,)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Default name of the column that uniquely identifies a trial.
DEFAULT_DATA_NUMBER_COLUMN = "DataNumber"


def stratify(
    identifiers: pd.DataFrame,
    *,
    all_split_names: Sequence[Sequence[str]],
    num_folds: int,
    data_number_column: str = DEFAULT_DATA_NUMBER_COLUMN,
) -> np.ndarray:
    """Assign per-trial stratification-group IDs via the MATLAB recursive splitter.

    Parameters
    ----------
    identifiers
        DataFrame with one row per trial. Must contain
        ``data_number_column`` (a unique-per-trial integer identifier) plus
        every column referenced in ``all_split_names``.
    all_split_names
        Ordered sequence of stratification levels. Each inner sequence
        names the columns whose Cartesian product defines the partitioning
        at that level. The MATLAB default is the 7-level hierarchy in
        ``PARAMETERS_cgg_procSimpleDecoders_v2.m``.
    num_folds
        The K in K-fold. A category with ``<= num_folds`` trials becomes a
        leaf (small categories can't be safely split further); ``> num_folds``
        categories recurse to the next level.
    data_number_column
        Column name carrying the unique per-trial integer ID.

    Returns
    -------
    numpy.ndarray
        Integer group ID per trial, aligned with ``identifiers``'s row
        order. Group IDs start at 1 (matches the MATLAB convention).

    Raises
    ------
    KeyError
        If a column referenced in ``all_split_names`` is missing from
        ``identifiers``.
    ValueError
        If ``data_number_column`` is missing or contains duplicate values.

    Notes
    -----
    Group IDs are assigned in the order leaves are produced by the recursive
    walk. This ordering is deterministic and matches MATLAB's traversal order
    so per-trial group IDs are identical across the two implementations.
    """
    _validate(identifiers, all_split_names, data_number_column)

    data_numbers = identifiers[data_number_column].to_numpy()

    # Recursively split. The "current subset" at each recursion is identified
    # by a set of DataNumbers (matching MATLAB's `FurtherSplit` variable).
    initial_subset = data_numbers.copy()
    group_list = _split_recursively(
        identifiers=identifiers,
        all_split_names=list(all_split_names),
        num_folds=num_folds,
        current_subset=initial_subset,
        split_level=0,
        data_number_column=data_number_column,
    )

    # Build the per-trial group-ID array.
    group_ids = np.full(len(identifiers), fill_value=-1, dtype=np.int64)
    data_number_to_row = {int(dn): row for row, dn in enumerate(data_numbers)}
    for group_id, member_data_numbers in enumerate(group_list, start=1):
        for dn in member_data_numbers:
            group_ids[data_number_to_row[int(dn)]] = group_id

    if (group_ids == -1).any():
        # This should not happen if MATLAB-parity behavior holds — every
        # trial reaches a leaf because the recursion bottoms out either at
        # the deepest level or earlier via the <= num_folds rule.
        unassigned = identifiers[data_number_column].iloc[
            np.where(group_ids == -1)[0]
        ].tolist()
        raise RuntimeError(
            f"Stratification left {len(unassigned)} trials unassigned: "
            f"DataNumbers={unassigned[:10]}{'...' if len(unassigned) > 10 else ''}"
        )

    return group_ids


def assign_folds(
    group_ids: np.ndarray,
    *,
    num_folds: int,
    seed: int = 0,
) -> np.ndarray:
    """Map per-trial group IDs to per-trial fold IDs (1..``num_folds``).

    Each leaf group is assigned to a single fold; folds are filled
    round-robin in shuffled group order so the resulting fold sizes are
    balanced.

    Parameters
    ----------
    group_ids
        Per-trial group IDs as returned by :func:`stratify`.
    num_folds
        Number of folds.
    seed
        Seed for the shuffling step. Different seeds produce different
        fold assignments while keeping the *strata* identical (which is the
        actual MATLAB-parity invariant — see module docstring).

    Returns
    -------
    numpy.ndarray
        Per-trial fold ID in ``1..num_folds``, aligned with ``group_ids``'s
        order.

    Notes
    -----
    MATLAB's ``cgg_getKFoldPartitions`` uses ``cvpartition`` for this step;
    its exact assignment depends on MATLAB's RNG state and cannot be
    bit-matched from Python. The strata themselves are the invariant.
    """
    if num_folds < 1:
        raise ValueError(f"num_folds must be >= 1; got {num_folds}.")

    unique_groups = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    shuffled_order = rng.permutation(unique_groups)

    # Round-robin assign groups to folds.
    group_to_fold: dict[int, int] = {}
    for position, group in enumerate(shuffled_order):
        group_to_fold[int(group)] = (position % num_folds) + 1

    return np.array([group_to_fold[int(g)] for g in group_ids], dtype=np.int64)


# ───────────────────────── Internals ─────────────────────────


def _validate(
    identifiers: pd.DataFrame,
    all_split_names: Sequence[Sequence[str]],
    data_number_column: str,
) -> None:
    """Raise informatively if the identifier table is missing required columns."""
    if data_number_column not in identifiers.columns:
        raise ValueError(
            f"identifiers must contain a '{data_number_column}' column."
        )
    if identifiers[data_number_column].duplicated().any():
        raise ValueError(
            f"'{data_number_column}' must be unique per trial; duplicates found."
        )

    required = {col for level in all_split_names for col in level}
    missing = required - set(identifiers.columns)
    if missing:
        raise KeyError(
            f"identifiers is missing stratification columns: {sorted(missing)}"
        )


def _split_recursively(
    *,
    identifiers: pd.DataFrame,
    all_split_names: list[Sequence[str]],
    num_folds: int,
    current_subset: np.ndarray,
    split_level: int,
    data_number_column: str,
) -> list[np.ndarray]:
    """Recursively walk the stratification tree, returning leaf groups.

    Mirrors ``cgg_procAssignGroupsBySplit.m``. At each level:

    1. If we have exhausted the split-name hierarchy, the entire subset
       becomes a single leaf.
    2. Otherwise, partition the current subset by the cross-product of the
       current level's columns.
    3. Sub-partitions with ``<= num_folds`` trials are emitted as leaves
       immediately; sub-partitions with ``> num_folds`` recurse.

    Parameters
    ----------
    identifiers
        Full identifier table (never sliced — recursion uses DataNumber lookup).
    all_split_names
        Hierarchy of stratification levels.
    num_folds
        Threshold for "small enough to be a leaf".
    current_subset
        DataNumbers of trials still being recursed on.
    split_level
        Index into ``all_split_names``; bumped on each recursion.
    data_number_column
        Trial-ID column name.

    Returns
    -------
    list of numpy.ndarray
        One entry per leaf group; each entry is the DataNumbers in that group.
    """
    # Base case: hierarchy exhausted → emit the whole remaining subset.
    if split_level >= len(all_split_names):
        return [current_subset.copy()]

    # Pick out only the rows in the current subset, in DataNumber order.
    # `.tolist()` because `isin` accepts iterables but pandas' type stubs are
    # strict about Series/Sequence; the ndarray → list copy is cheap relative
    # to the subsequent DataFrame indexing.
    mask = identifiers[data_number_column].isin(current_subset.tolist())
    subset_df = identifiers.loc[mask].copy()

    maintain, further = _split_into_groups(
        subset_df=subset_df,
        split_names=list(all_split_names[split_level]),
        num_folds=num_folds,
        data_number_column=data_number_column,
    )

    leaves: list[np.ndarray] = []
    if maintain.size > 0:
        leaves.append(maintain)

    for sub in further:
        leaves.extend(
            _split_recursively(
                identifiers=identifiers,
                all_split_names=all_split_names,
                num_folds=num_folds,
                current_subset=sub,
                split_level=split_level + 1,
                data_number_column=data_number_column,
            )
        )
    return leaves


def _split_into_groups(
    *,
    subset_df: pd.DataFrame,
    split_names: list[str],
    num_folds: int,
    data_number_column: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Partition a trial subset by the Cartesian product of split-column values.

    Mirrors ``cgg_procSplitIntoGroups.m``. Each (column-value combination)
    produces a sub-partition; partitions with ``<= num_folds`` trials are
    "maintain" (leaves), partitions with more are "further" (recurse).

    Parameters
    ----------
    subset_df
        Rows of the identifier table that belong to the current subset.
    split_names
        Column names whose Cartesian product defines the partition keys at
        this level.
    num_folds
        Leaf threshold.
    data_number_column
        Trial-ID column.

    Returns
    -------
    maintain : numpy.ndarray
        Concatenated DataNumbers from all small-enough partitions (a single
        leaf containing all of them — matches MATLAB's
        ``MaintainSplit=[Possible_Category_DataNumber{...}]`` flattening).
    further : list of numpy.ndarray
        One DataNumber array per partition that needs further splitting.
    """
    if not split_names:
        # No columns to split on at this level → entire subset becomes one
        # category. Behavior matches MATLAB: empty SplitNames means a single
        # category with all trials.
        all_dn = subset_df[data_number_column].to_numpy()
        if len(all_dn) <= num_folds:
            return all_dn, []
        return np.array([], dtype=all_dn.dtype), [all_dn]

    # The MATLAB code generates the full Cartesian product of unique values
    # across all split columns, then assigns each trial to one of those
    # categories. Empty categories are skipped (the >NumFolds / <=NumFolds
    # decision only looks at non-empty buckets).
    grouped = subset_df.groupby(list(split_names), sort=True, dropna=False)

    maintain_parts: list[np.ndarray] = []
    further: list[np.ndarray] = []

    for _category_key, category_rows in grouped:
        category_data_numbers = category_rows[data_number_column].to_numpy()
        if len(category_data_numbers) > num_folds:
            further.append(category_data_numbers)
        else:
            # MATLAB concatenates all small-category DataNumbers into a single
            # MaintainSplit group, not one leaf per small category.
            maintain_parts.append(category_data_numbers)

    if maintain_parts:
        maintain = np.concatenate(maintain_parts)
    else:
        # Use the subset's DataNumber dtype so concatenation downstream is consistent.
        sample_dtype = subset_df[data_number_column].to_numpy().dtype
        maintain = np.array([], dtype=sample_dtype)

    return maintain, further


__all__ = [
    "DEFAULT_DATA_NUMBER_COLUMN",
    "assign_folds",
    "stratify",
]
