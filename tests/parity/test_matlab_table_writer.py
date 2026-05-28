"""T4 — ``promote_struct_to_table`` produces a real MATLAB table on disk.

Verifies that
:func:`neural_data_decoding.interop.matlab_table_writer.promote_struct_to_table`
turns a Python-written struct ``.mat`` into a ``.mat`` containing a native
MATLAB ``table`` object — what MATLAB's ``istable()`` returns ``true`` for,
and what ``cgg_saveValidationCMTable`` would have written.

Unlike the earlier ``matlab.engine`` approach, these tests drive MATLAB
via ``matlab -batch`` as a subprocess (see
:mod:`neural_data_decoding.interop.matlab_runner`), which works regardless
of the Python interpreter's architecture. They are gated by the
``needs_matlab`` marker and auto-skip when no MATLAB executable is found.

Each test starts a fresh MATLAB process (cold start ~10–20 s), so the
suite intentionally keeps the number of MATLAB round-trips small: one to
promote, one to describe. Module-scoped fixtures share the promoted file
across assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_data_decoding.interop.cm_table_format import (
    VALIDATION_CM_TABLE_FILENAME,
    write_cm_table_mat,
)
from neural_data_decoding.interop.matlab_runner import matlab_available
from neural_data_decoding.interop.matlab_table_writer import (
    describe_table_mat,
    promote_struct_to_table,
)


pytestmark = pytest.mark.needs_matlab


# A module-level guard so the fixtures themselves skip cleanly even if a
# runner invokes them without the collection-time marker filter.
if not matlab_available():  # pragma: no cover - environment dependent
    pytest.skip(
        "MATLAB executable not found; skipping table-writer parity tests.",
        allow_module_level=True,
    )


_EXPECTED_COLUMNS = (
    "DataNumber",
    "TrueValue",
    "Window_1",
    "Window_2",
    "Aggregation_Prediction",
    "TrialConfidence",
    "TaskConfidence",
)


@pytest.fixture(scope="module")
def struct_mat(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a small Python-side struct ``.mat`` to use as conversion input."""
    tmp = tmp_path_factory.mktemp("struct_in")
    n, d = 4, 3
    out = tmp / VALIDATION_CM_TABLE_FILENAME
    write_cm_table_mat(
        out,
        data_numbers=np.arange(101, 101 + n, dtype=np.int64),
        true_values=np.array(
            [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 0, 1]], dtype=np.float64
        ),
        window_predictions=[
            np.zeros((n, d), dtype=np.float64),
            np.ones((n, d), dtype=np.float64),
        ],
        aggregation_prediction=np.full((n, d), 0.5, dtype=np.float64),
    )
    return out


@pytest.fixture(scope="module")
def promoted_table(struct_mat: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Promote the struct to a native MATLAB table once for the module."""
    out_dir = tmp_path_factory.mktemp("table_out")
    out = out_dir / "CM_Table_promoted.mat"
    promote_struct_to_table(struct_mat, out)
    return out


@pytest.fixture(scope="module")
def table_metadata(promoted_table: Path) -> dict[str, Any]:
    """Query MATLAB for the promoted table's metadata once for the module."""
    return describe_table_mat(promoted_table)


def test_promoted_file_exists(promoted_table: Path) -> None:
    """The forward helper wrote the output file."""
    assert promoted_table.exists(), "Forward helper did not write the output file."


def test_promoted_variable_is_a_matlab_table(table_metadata: dict[str, Any]) -> None:
    """MATLAB's ``istable`` returns true for the promoted variable."""
    assert table_metadata["istable"] is True, (
        "Promoted file's CM_Table is not a MATLAB table — istable() returned false."
    )


def test_promoted_table_preserves_columns(table_metadata: dict[str, Any]) -> None:
    """Every expected column survives the struct → table promotion."""
    present = set(table_metadata["variables"])
    for col in _EXPECTED_COLUMNS:
        assert col in present, (
            f"Promoted table missing column {col!r}; got {sorted(present)}."
        )


def test_promoted_table_row_count(table_metadata: dict[str, Any]) -> None:
    """The table has one row per input trial (4 in the fixture)."""
    assert table_metadata["num_rows"] == 4
