"""``CM_Table`` schema + ``.mat`` writer (primary MATLAB-interop output).

The MATLAB analysis pipeline (``DATA_cggAllNetworkEncoderResults.m`` →
``FIGURE_cggAllNetworkEncoderResults.m``) consumes per-trial confusion-matrix
telemetry from ``.mat`` files containing a single MATLAB table named
``CM_Table``. Critical Note #16 in the migration plan identifies this as the
**primary** MATLAB-interop surface — get this wrong and the entire downstream
analysis pipeline can't load Python output.

Schema (derived from ``cgg_generateBlankCMTable.m`` and the writer functions
``cgg_getClassifierOutputsFromProbabilities.m`` and
``cgg_procPredictionsFromDatastoreNetwork.m``):

================================  ======================  ===========================
Column                            dtype                   Meaning
================================  ======================  ===========================
``DataNumber``                    ``single`` (N, 1)       Trial identifier (global,
                                                          sparse — NOT 1..N)
``TrueValue``                     ``double`` (N, D)       Ground-truth labels per dim
``Window_1`` … ``Window_K``       ``double`` (N, D)       Per-window predicted class
``Aggregation_Prediction``        ``double`` (N, D)       Cross-window aggregate
``TrialConfidence``               ``double`` (N, 1)       Scalar per-trial confidence
                                                          (=1 when head disabled)
``TaskConfidence``                ``double`` (N, D)       Per-dimension confidence
                                                          (=1 when head disabled)
================================  ======================  ===========================

Shape confirmation from a real MATLAB validation run
(``tests/fixtures/reference_cm_tables/CM_Table.mat`` — 106 trials,
4 dimensions, 59 windows): ``TaskConfidence`` is ``106×4`` (per-dim), not
``106×1``. ``TrialConfidence`` is ``106×1`` (per-trial). Both shape
conventions are pinned by unit tests against that fixture.

The number of ``Window_k`` columns always matches the model's window (``W``)
axis: the caller emits one column per window and the cross-window
``Aggregation_Prediction`` alongside. When a model exposes only a single
window, ``Window_1`` equals ``Aggregation_Prediction`` by definition. A
classifier with no confidence heads (Milestone A logistic) leaves both
confidence columns filled with ones. The writer accepts the full schema so
Milestones B and C can populate the confidence columns without touching this
module.

The serialization format is **a SciPy struct of arrays** (``scipy.io.savemat``
with a single key ``CM_Table`` mapping to a dict of column arrays). MATLAB's
``load`` returns this as a struct; the downstream ``cgg_getAllEncoderCMTable``
accesses fields via ``m_CM_Table.CM_Table.<field>``, which works identically
for structs and tables. A ``struct2table`` shim on the MATLAB side handles
table-typed callers — that shim lives outside this Python package.

Examples
--------
>>> import numpy as np
>>> from pathlib import Path
>>> import tempfile
>>> data_numbers = np.array([1, 2, 3], dtype=np.int32)
>>> true_values = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.float64)
>>> window_predictions = [np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float64)]
>>> with tempfile.TemporaryDirectory() as tmp:
...     out = Path(tmp) / "CM_Table_Validation.mat"
...     write_cm_table_mat(
...         out,
...         data_numbers=data_numbers,
...         true_values=true_values,
...         window_predictions=window_predictions,
...     )
...     out.exists()
True
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import savemat


_VALIDATION_FILENAME = "CM_Table_Validation.mat"
_TEST_FILENAME = "CM_Table.mat"


def write_cm_table_mat(
    output_path: Path,
    *,
    data_numbers: np.ndarray,
    true_values: np.ndarray,
    window_predictions: Sequence[np.ndarray],
    aggregation_prediction: Optional[np.ndarray] = None,
    trial_confidence: Optional[np.ndarray] = None,
    task_confidence: Optional[np.ndarray] = None,
) -> None:
    """Write a ``CM_Table`` ``.mat`` file in the MATLAB-readable schema.

    Parameters
    ----------
    output_path
        Destination path. The file is overwritten if it exists. The MATLAB
        consumer expects the filename ``CM_Table.mat`` (training) or
        ``CM_Table_Validation.mat`` (validation); see the
        :data:`VALIDATION_CM_TABLE_FILENAME` and
        :data:`TEST_CM_TABLE_FILENAME` constants.
    data_numbers
        Per-trial integer identifiers, shape ``(N,)``. Stored as ``single``
        per MATLAB's schema; integer inputs are cast to ``float32`` on write.
    true_values
        Ground-truth labels, shape ``(N, num_dimensions)``. Stored as
        ``double``.
    window_predictions
        One ``(N, num_dimensions)`` array per analysis window. Each becomes
        the ``Window_k`` column (1-indexed). At least one window is required
        — Milestone A passes a single window and is happy with that.
    aggregation_prediction
        Cross-window aggregated prediction, shape ``(N, num_dimensions)``.
        If ``None`` and exactly one window is provided, it defaults to that
        window's predictions (Milestone A behavior).
    trial_confidence
        Per-trial confidence in ``[0, 1]``, shape ``(N,)`` or ``(N, 1)``.
        Stored as a column vector ``(N, 1)`` per MATLAB convention. If
        ``None``, the column is filled with ones — matching MATLAB's
        behavior when the confidence head is disabled.
    task_confidence
        Per-dimension confidence in ``[0, 1]``, shape ``(N, num_dimensions)``.
        **Not** a single column — MATLAB stores one confidence value per
        classification dimension. If ``None``, the column is filled with
        ones of shape ``(N, num_dimensions)``, matching MATLAB's behavior
        when the task-confidence head is disabled.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``window_predictions`` is empty.
    """
    output_path = Path(output_path)

    if not window_predictions:
        raise ValueError("window_predictions must contain at least one window.")

    n_trials = int(data_numbers.shape[0])

    if true_values.ndim != 2 or true_values.shape[0] != n_trials:
        raise ValueError(
            f"true_values must have shape (N, D) with N={n_trials}; "
            f"got shape {tuple(true_values.shape)}."
        )

    for k, win in enumerate(window_predictions, start=1):
        if win.shape != true_values.shape:
            raise ValueError(
                f"window_predictions[{k - 1}] has shape {tuple(win.shape)}; "
                f"expected {tuple(true_values.shape)} to match true_values."
            )

    if aggregation_prediction is None:
        if len(window_predictions) == 1:
            aggregation_prediction = window_predictions[0]
        else:
            raise ValueError(
                "aggregation_prediction is required when more than one "
                "window is supplied."
            )
    elif aggregation_prediction.shape != true_values.shape:
        raise ValueError(
            f"aggregation_prediction shape {tuple(aggregation_prediction.shape)} "
            f"does not match true_values shape {tuple(true_values.shape)}."
        )

    num_dimensions = int(true_values.shape[1])

    if trial_confidence is None:
        trial_confidence = np.ones(n_trials, dtype=np.float64)
    else:
        trial_confidence = np.asarray(trial_confidence)
        if trial_confidence.shape not in {(n_trials,), (n_trials, 1)}:
            raise ValueError(
                f"trial_confidence must have shape ({n_trials},) or ({n_trials}, 1); "
                f"got shape {tuple(trial_confidence.shape)}."
            )

    if task_confidence is None:
        task_confidence = np.ones((n_trials, num_dimensions), dtype=np.float64)
    else:
        task_confidence = np.asarray(task_confidence)
        if task_confidence.shape != (n_trials, num_dimensions):
            raise ValueError(
                f"task_confidence must have shape ({n_trials}, {num_dimensions}) "
                f"to match TrueValue's dimension count; "
                f"got shape {tuple(task_confidence.shape)}."
            )

    table: dict[str, np.ndarray] = {
        "DataNumber": data_numbers.astype(np.float32, copy=False).reshape(-1, 1),
        "TrueValue": np.asarray(true_values, dtype=np.float64),
    }
    for k, win in enumerate(window_predictions, start=1):
        table[f"Window_{k}"] = np.asarray(win, dtype=np.float64)
    table["Aggregation_Prediction"] = np.asarray(aggregation_prediction, dtype=np.float64)
    table["TrialConfidence"] = trial_confidence.astype(np.float64, copy=False).reshape(-1, 1)
    table["TaskConfidence"] = task_confidence.astype(np.float64, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        str(output_path),
        {"CM_Table": table},
        do_compression=True,
        oned_as="column",
    )


VALIDATION_CM_TABLE_FILENAME = _VALIDATION_FILENAME
"""Filename for the **validation-set** CM_Table, written each epoch during
training. Consumed by ``cgg_saveValidationCMTable.m`` and used for model
selection (the Optimal snapshot tracks the best validation metric)."""

TEST_CM_TABLE_FILENAME = _TEST_FILENAME
"""Filename for the **test-set** CM_Table, written once at the end of
training (after restoring the Optimal weights). Consumed by downstream
analysis scripts (``DATA_cggAllNetworkEncoderResults.m`` etc.) for the
final reported results — this is what the MATLAB pipeline writes as
``CM_Table.mat`` (no ``_Validation`` suffix)."""


__all__ = [
    "TEST_CM_TABLE_FILENAME",
    "VALIDATION_CM_TABLE_FILENAME",
    "write_cm_table_mat",
]
