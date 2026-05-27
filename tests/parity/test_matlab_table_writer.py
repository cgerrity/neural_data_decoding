"""T4 — ``promote_struct_to_table`` produces a real MATLAB table on disk.

Verifies that
:func:`neural_data_decoding.interop.matlab_table_writer.promote_struct_to_table`
turns a Python-written struct ``.mat`` into a v7.3 ``.mat`` containing a
native MATLAB ``table`` object — what MATLAB's ``istable()`` returns ``true``
for, and what ``cgg_saveValidationCMTable`` would have written.

The whole test is auto-skipped when ``matlab.engine`` (MATLAB Engine for
Python) is not installed in the active environment. Starting an engine is
slow (≈3–5 s on cold start), so we open one per test session via a
session-scoped fixture and reuse it across cases.

To run locally, install the MATLAB-version-matched engine package::

    pip install matlabengine

Then ``pytest tests/parity/test_matlab_table_writer.py`` executes the
parity checks against a freshly started MATLAB.
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
from neural_data_decoding.interop.matlab_table_writer import promote_struct_to_table


@pytest.fixture(scope="session")
def matlab_engine() -> Any:
    """Start a MATLAB Engine for Python instance, or skip if unavailable.

    Reused across every test in this module — engine startup dominates
    runtime. Stopped automatically at session end.
    """
    eng_mod = pytest.importorskip(
        "matlab.engine",
        reason=(
            "matlab.engine not installed. Install via `pip install matlabengine` "
            "to enable the forward-helper parity tests."
        ),
    )
    engine = eng_mod.start_matlab()
    yield engine
    engine.quit()


@pytest.fixture
def python_struct_mat(tmp_path: Path) -> Path:
    """Write a tiny Python-side struct ``.mat`` to use as input."""
    n, d = 4, 3
    out = tmp_path / VALIDATION_CM_TABLE_FILENAME
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


def test_promote_struct_produces_matlab_table(
    matlab_engine: Any, python_struct_mat: Path, tmp_path: Path
) -> None:
    """The promoted file's CM_Table is a MATLAB ``table`` object, not a struct."""
    out = tmp_path / "CM_Table_promoted.mat"
    promote_struct_to_table(python_struct_mat, out, engine=matlab_engine)

    assert out.exists(), "Forward helper did not write the output file."

    # Ask MATLAB to confirm the on-disk variable is a table.
    matlab_engine.eval(f"loaded = load('{out}');", nargout=0)
    is_table = matlab_engine.eval("istable(loaded.CM_Table)", nargout=1)
    assert bool(is_table), (
        "Promoted file's CM_Table is not a MATLAB table — istable() returned false."
    )


def test_promote_struct_preserves_columns_and_values(
    matlab_engine: Any, python_struct_mat: Path, tmp_path: Path
) -> None:
    """Field set + per-cell values survive the struct → table promotion."""
    out = tmp_path / "CM_Table_promoted.mat"
    promote_struct_to_table(python_struct_mat, out, engine=matlab_engine)

    # Field set: pull VariableNames from the MATLAB side as a cell array of chars.
    matlab_engine.eval(
        f"loaded = load('{out}'); vn = loaded.CM_Table.Properties.VariableNames;",
        nargout=0,
    )
    var_count = int(matlab_engine.eval("numel(vn)", nargout=1))
    var_names = [
        str(matlab_engine.eval(f"vn{{{k}}}", nargout=1))
        for k in range(1, var_count + 1)
    ]
    for expected in (
        "DataNumber",
        "TrueValue",
        "Window_1",
        "Window_2",
        "Aggregation_Prediction",
        "TrialConfidence",
        "TaskConfidence",
    ):
        assert expected in var_names, (
            f"Promoted table missing expected column {expected}; "
            f"got {var_names}"
        )

    # Spot-check a value: TrueValue[0, 0] should be 0 (from the fixture above).
    val = float(matlab_engine.eval("loaded.CM_Table.TrueValue(1, 1)", nargout=1))
    assert val == 0.0
    # Window_2 was the all-ones tensor.
    val_w2 = float(matlab_engine.eval("loaded.CM_Table.Window_2(2, 3)", nargout=1))
    assert val_w2 == 1.0
