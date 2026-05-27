"""T4 — Python ``write_cm_table_mat`` produces a structurally-equivalent file.

This test loads a real MATLAB-generated CM_Table that has been pre-converted
into a Python-readable struct form (see
``scripts/convert_reference_cm_tables.m``), then re-writes its content via
:func:`neural_data_decoding.interop.cm_table_format.write_cm_table_mat`. The
Python output must round-trip-load to the same field set, shapes, and
values — confirming the schema (Critical Note #16) matches what MATLAB's
analysis pipeline expects.

We don't compare dtype byte-for-byte because MATLAB's ``save -v7``
optimizes integer-valued doubles into the smallest fitting integer type
(e.g. ``uint8`` for small class labels). The Python writer stores
everything as ``float64`` (the type MATLAB tables natively use). Both
representations land back at the same numeric values when read.

Fixture generation
------------------
The Python-readable fixtures
(``CM_Table_python_struct.mat`` / ``CM_Table_Validation_python_struct.mat``)
are produced from the MATLAB-native ``CM_Table.mat`` /
``CM_Table_Validation.mat`` via a one-shot MATLAB conversion that calls
``table2struct(T, 'ToScalar', true)`` then ``save -v7``. Both files live
under ``tests/fixtures/reference_cm_tables/``. The gitignore keeps them
out of commits — they're regenerated locally when needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from neural_data_decoding.interop.cm_table_format import (
    VALIDATION_CM_TABLE_FILENAME,
    write_cm_table_mat,
)


FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent / "fixtures" / "reference_cm_tables"
)


# ───────────────────────── Helpers ─────────────────────────


def _load_reference_struct(filename: str) -> dict[str, np.ndarray]:
    """Load a MATLAB-converted reference CM_Table as a dict of column arrays."""
    path = FIXTURE_DIR / filename
    if not path.exists():
        pytest.skip(
            f"Reference fixture missing: {path}. "
            "Regenerate via the MATLAB MCP / script that converts the table to a "
            "v7 struct (see module docstring)."
        )
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=False)
    record = mat["CM_Table"][0, 0]
    return {name: getattr(record, name) for name in record._fieldnames}


@pytest.fixture(
    params=[
        "CM_Table_python_struct.mat",
        "CM_Table_Validation_python_struct.mat",
    ],
    ids=["training", "validation"],
)
def reference_struct(request: pytest.FixtureRequest) -> dict[str, np.ndarray]:
    """Parametrized fixture covering both reference CM_Tables."""
    return _load_reference_struct(request.param)


# ───────────────────────── Field-set / shape parity ─────────────────────────


REQUIRED_FIELDS = {
    "DataNumber",
    "TrueValue",
    "Aggregation_Prediction",
    "TrialConfidence",
    "TaskConfidence",
}


def test_reference_contains_required_fields(
    reference_struct: dict[str, np.ndarray],
) -> None:
    """The MATLAB-generated reference must carry every column we depend on."""
    missing = REQUIRED_FIELDS - set(reference_struct.keys())
    assert not missing, f"Missing required CM_Table fields: {sorted(missing)}"


def test_reference_window_columns_present(
    reference_struct: dict[str, np.ndarray],
) -> None:
    """At least one ``Window_k`` column must be present and well-formed."""
    window_keys = sorted(
        (k for k in reference_struct if k.startswith("Window_")),
        key=lambda s: int(s.split("_")[1]),
    )
    assert window_keys, "Reference has no Window_k columns."
    n_trials = reference_struct["TrueValue"].shape[0]
    n_dims = reference_struct["TrueValue"].shape[1]
    for k in window_keys:
        v = reference_struct[k]
        assert v.shape == (n_trials, n_dims), (
            f"{k} shape {v.shape} mismatched TrueValue shape ({n_trials}, {n_dims})."
        )


def test_reference_confidence_shapes(
    reference_struct: dict[str, np.ndarray],
) -> None:
    """TrialConfidence is (N, 1); TaskConfidence is (N, D)."""
    n_trials, n_dims = reference_struct["TrueValue"].shape
    assert reference_struct["TrialConfidence"].shape == (n_trials, 1)
    assert reference_struct["TaskConfidence"].shape == (n_trials, n_dims)


def test_reference_datanumber_shape(
    reference_struct: dict[str, np.ndarray],
) -> None:
    """DataNumber is an (N, 1) column vector — not a (1, N) row."""
    n_trials = reference_struct["TrueValue"].shape[0]
    assert reference_struct["DataNumber"].shape == (n_trials, 1)


# ───────────────────────── Round-trip parity ─────────────────────────


def test_python_writer_round_trips_reference(
    reference_struct: dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Python writer's output, loaded back, equals the reference structurally.

    Re-emits the reference data through ``write_cm_table_mat`` and asserts
    the loaded result has the same field names, the same shapes, and the
    same values (up to the integer-vs-float storage difference noted in
    the module docstring).
    """
    n_trials = reference_struct["TrueValue"].shape[0]
    window_keys = sorted(
        (k for k in reference_struct if k.startswith("Window_")),
        key=lambda s: int(s.split("_")[1]),
    )
    window_predictions = [
        reference_struct[k].astype(np.float64) for k in window_keys
    ]

    out = tmp_path / VALIDATION_CM_TABLE_FILENAME
    write_cm_table_mat(
        out,
        data_numbers=reference_struct["DataNumber"].astype(np.int64).ravel(),
        true_values=reference_struct["TrueValue"].astype(np.float64),
        window_predictions=window_predictions,
        aggregation_prediction=reference_struct["Aggregation_Prediction"].astype(
            np.float64
        ),
        trial_confidence=reference_struct["TrialConfidence"].astype(np.float64).ravel(),
        task_confidence=reference_struct["TaskConfidence"].astype(np.float64),
    )

    roundtrip = scipy.io.loadmat(out, struct_as_record=False, squeeze_me=False)
    rt = roundtrip["CM_Table"][0, 0]
    rt_fields = set(rt._fieldnames)

    # Field set must be a superset of the reference's required+window keys.
    expected_fields = REQUIRED_FIELDS | set(window_keys)
    missing = expected_fields - rt_fields
    assert not missing, f"Round-trip lost fields: {sorted(missing)}"

    # Shapes match.
    for f in REQUIRED_FIELDS | set(window_keys):
        assert getattr(rt, f).shape == reference_struct[f].shape, (
            f"Shape mismatch for {f}: "
            f"round-trip={getattr(rt, f).shape} vs reference={reference_struct[f].shape}"
        )

    # Values match (allow integer-vs-float storage difference — compare numerically).
    for f in REQUIRED_FIELDS | set(window_keys):
        np.testing.assert_array_equal(
            getattr(rt, f).astype(np.float64),
            reference_struct[f].astype(np.float64),
            err_msg=f"Value mismatch for {f}",
        )

    assert n_trials == getattr(rt, "DataNumber").shape[0]


def test_python_writer_data_numbers_use_correct_column_orientation(
    reference_struct: dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Round-trip preserves the column-vector orientation MATLAB uses for DataNumber."""
    n_trials = reference_struct["TrueValue"].shape[0]
    window = reference_struct["Window_1"].astype(np.float64)
    out = tmp_path / "rt.mat"
    write_cm_table_mat(
        out,
        data_numbers=np.arange(1, n_trials + 1, dtype=np.int64),
        true_values=reference_struct["TrueValue"].astype(np.float64),
        window_predictions=[window],
    )
    rt = scipy.io.loadmat(out, struct_as_record=False, squeeze_me=False)["CM_Table"][
        0, 0
    ]
    assert rt.DataNumber.shape == (n_trials, 1)  # column vector
