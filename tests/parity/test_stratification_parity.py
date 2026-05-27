"""T3 — Python stratification matches the MATLAB reference partition.

This test compares per-trial group IDs produced by
:func:`neural_data_decoding.data.stratification.stratify` against the
``PartitionGroups`` array stored in a MATLAB-generated reference fixture
under ``tests/fixtures/reference_partitions/``.

The reference is produced by ``scripts/generate_stratification_fixture.m``
(driven from Python via ``scripts/prepare_golden_fixtures.py --milestone 0``).
If the fixture is not present, every test in this module is skipped with
an informative message — fresh checkouts that haven't yet generated
fixtures still see a green suite.

The fixture itself is small and self-contained: it doesn't require any
preprocessed neural data, only the MATLAB pipeline source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from neural_data_decoding.data.mat_files import load_mat
from neural_data_decoding.data.stratification import stratify

FIXTURE_FILENAME = "synthetic_easy_partition.mat"


# ───────────────────────── Fixture loading ─────────────────────────


def _unwrap_cell_string_array(raw: np.ndarray) -> list[str]:
    """Decode a ``cellstr``-shaped object array from ``scipy.io.loadmat``.

    MATLAB cell arrays of strings come through scipy as a
    ``(1, N)`` object array whose entries are themselves length-1
    ``<Uk>`` arrays. We flatten that to a plain ``list[str]``.

    Parameters
    ----------
    raw
        The ``.mat`` value as returned by :func:`load_mat`.

    Returns
    -------
    list of str
        The decoded list of strings.
    """
    flat = np.asarray(raw).flatten()
    out: list[str] = []
    for entry in flat:
        # Each entry is itself a length-1 ndarray (or already a scalar str).
        if isinstance(entry, np.ndarray):
            out.append(str(entry.item()))
        else:
            out.append(str(entry))
    return out


def _load_fixture(path: Path) -> dict[str, Any]:
    """Load the MATLAB fixture and decode all entries into clean Python types.

    Parameters
    ----------
    path
        Path to the ``synthetic_easy_partition.mat`` file.

    Returns
    -------
    dict
        Keys:

        * ``identifier_table`` (:class:`pandas.DataFrame`)
        * ``all_split_names`` (list of lists of str)
        * ``num_folds`` (int)
        * ``partition_groups`` (1D :class:`numpy.ndarray` of int)
    """
    raw = load_mat(path)

    columns = _unwrap_cell_string_array(raw["IdentifierColumns"])
    data = np.asarray(raw["IdentifierData"])
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(
            f"Fixture IdentifierData shape {data.shape} is incompatible with "
            f"{len(columns)} columns."
        )
    identifier_table = pd.DataFrame(data, columns=columns)

    # MATLAB uses "Data Number" with a space; the Python module uses
    # "DataNumber" as its default column name. Rename so the existing
    # stratify() API works without a custom column name on the caller's part.
    if "Data Number" in identifier_table.columns:
        identifier_table = identifier_table.rename(
            columns={"Data Number": "DataNumber"}
        )

    num_split_levels = int(np.asarray(raw["NumSplitLevels"]).flatten()[0])
    all_split_names: list[list[str]] = []
    for level in range(1, num_split_levels + 1):
        key = f"AllSplitNamesLevel{level}"
        if key not in raw:
            raise KeyError(
                f"Fixture is missing expected key '{key}' for split level {level}."
            )
        all_split_names.append(_unwrap_cell_string_array(raw[key]))

    num_folds = int(np.asarray(raw["NumFolds"]).flatten()[0])
    partition_groups = np.asarray(raw["PartitionGroups"]).flatten().astype(np.int64)

    return {
        "identifier_table": identifier_table,
        "all_split_names": all_split_names,
        "num_folds": num_folds,
        "partition_groups": partition_groups,
    }


@pytest.fixture(scope="module")
def fixture(reference_partitions_dir: Path) -> dict[str, Any]:
    """Load the MATLAB reference fixture, skipping if it's not on disk."""
    path = reference_partitions_dir / FIXTURE_FILENAME
    if not path.is_file():
        pytest.skip(
            f"Reference fixture not present: {path}\n"
            f"Regenerate locally with:\n"
            f"    python scripts/prepare_golden_fixtures.py --milestone 0"
        )
    return _load_fixture(path)


# ───────────────────────── Tests ─────────────────────────


@pytest.mark.parity
def test_python_strata_match_matlab_exactly(fixture: dict[str, Any]) -> None:
    """Per-trial group IDs from Python match MATLAB element-for-element.

    This is the strictest form of strata parity. It only holds when both
    implementations iterate categories in the same lexicographic order;
    if it fails the weaker test below should still pass — that's our
    diagnostic boundary between "wrong groupings" and "right groupings,
    different IDs".
    """
    python_groups = stratify(
        fixture["identifier_table"],
        all_split_names=fixture["all_split_names"],
        num_folds=fixture["num_folds"],
    )
    np.testing.assert_array_equal(python_groups, fixture["partition_groups"])


@pytest.mark.parity
def test_python_groupings_equivalent_to_matlab(fixture: dict[str, Any]) -> None:
    """Trials co-grouped in Python are also co-grouped in MATLAB, and vice versa.

    A weaker but more robust parity check than exact-ID equality: instead
    of asserting that Python and MATLAB assign the same numeric ID to a
    group, we assert that the *partitioning relation* they induce is the
    same. Equivalent partitions can have different ID labelings.
    """
    python_groups = stratify(
        fixture["identifier_table"],
        all_split_names=fixture["all_split_names"],
        num_folds=fixture["num_folds"],
    )
    matlab_groups = fixture["partition_groups"]
    n = len(matlab_groups)

    # Pairwise co-grouping: for every pair of trials, the two assignments
    # must agree on whether they share a group.
    py_same = python_groups[:, None] == python_groups[None, :]
    ml_same = matlab_groups[:, None] == matlab_groups[None, :]
    mismatches = np.sum(py_same != ml_same)
    assert mismatches == 0, (
        f"Python and MATLAB disagree on co-grouping of "
        f"{mismatches} trial pairs (out of {n * n})."
    )


@pytest.mark.parity
def test_fixture_self_consistency(fixture: dict[str, Any]) -> None:
    """The fixture itself is well-formed (sanity check, not a parity test).

    Verifies that the MATLAB-generated partition assigns every trial a
    group ID and that the IDs form a contiguous 1..K range. Failure here
    means the fixture was generated incorrectly, not that Python diverges
    from MATLAB.
    """
    groups = fixture["partition_groups"]
    assert groups.ndim == 1
    assert len(groups) == len(fixture["identifier_table"])
    assert (groups >= 1).all(), "MATLAB group IDs are 1-indexed; saw 0 or negative."

    unique = np.unique(groups)
    expected = np.arange(1, len(unique) + 1)
    np.testing.assert_array_equal(unique, expected, err_msg=(
        "MATLAB partition IDs should form a contiguous 1..K range."
    ))
