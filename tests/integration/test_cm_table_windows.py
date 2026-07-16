"""The CM_Table writer emits one ``Window_k`` column per model window.

Regression guard for the single-window placeholder that ``_write_cm_table_for_split``
used to write (``window_predictions=[last_window]``). MATLAB's CM_Table carries
one ``Window_k`` column per data window (the reference fixture has 59); Python
was collapsing every window into ``Window_1``.

These assert *structure* — the column count equals the model's actual ``W`` axis,
and each column is a valid ``(N, D)`` block of class indices. Exact per-window
*numeric* parity against MATLAB is a separate, MATLAB-gated check (no Milestone-A
reference ensemble exists yet); see ``tests/parity/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch
from torch.utils.data import DataLoader

from neural_data_decoding.cli import (
    _apply_cfg_flags,
    _build_model,
    _build_synthetic_split,
    _load_config,
    _write_cm_table_for_split,
)
from neural_data_decoding.data.dataset import collate_trials
from neural_data_decoding.models.composite import VariationalOutput


def _build_model_and_loader(config_name: str):
    """Compose a config, build its synthetic val split, model, and a loader."""
    cfg = _load_config(config_name)
    ns = argparse.Namespace(
        fold=1, sweep_index=None, session_run_idx=None, session=None, override=[]
    )
    _apply_cfg_flags(cfg, ns)
    train_ds, val_ds, _ = _build_synthetic_split(cfg, train_load_schedule=None)
    loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_trials)
    sample_x = next(iter(loader))["x"]
    model = _build_model(
        cfg,
        in_features=int(sample_x.shape[-1]),
        num_classes_per_dim=train_ds.num_classes_per_dim,
    )
    return cfg, model, loader, len(val_ds)


def _model_window_count(model: torch.nn.Module, loader: DataLoader) -> int:
    """Number of windows on the model's output ``W`` axis (1 if non-sequence)."""
    model.eval()
    with torch.no_grad():
        out = model(next(iter(loader))["x"])
    logits = out.logits if isinstance(out, VariationalOutput) else out
    first = logits[0]
    return int(first.shape[1]) if first.ndim == 3 else 1


def _load_cm_windows(path: Path) -> dict[str, np.ndarray]:
    """Load a written CM_Table and return its ``Window_k`` arrays by name."""
    struct = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=False)
    cm = struct["CM_Table"][0, 0]
    return {f: getattr(cm, f) for f in cm._fieldnames if f.startswith("Window_")}


@pytest.mark.parametrize("config_name", ["C_optimal_synthetic", "A_logistic_synthetic"])
def test_cm_table_has_one_column_per_model_window(
    config_name: str, tmp_path: Path
) -> None:
    """The written CM_Table has exactly one ``Window_k`` per model window."""
    _, model, loader, n_trials = _build_model_and_loader(config_name)
    expected_windows = _model_window_count(model, loader)

    out_path = tmp_path / "CM_Table.mat"
    _write_cm_table_for_split(model, loader, out_path)

    windows = _load_cm_windows(out_path)
    assert len(windows) == expected_windows, (
        f"{config_name}: wrote {len(windows)} Window_k columns, "
        f"model has {expected_windows} windows"
    )
    # Contiguous 1-indexed naming Window_1 … Window_K.
    assert set(windows) == {f"Window_{k}" for k in range(1, expected_windows + 1)}

    struct = scipy.io.loadmat(out_path, struct_as_record=False, squeeze_me=False)
    true_shape = struct["CM_Table"][0, 0].TrueValue.shape
    for name, col in windows.items():
        assert col.shape == true_shape, f"{name} shape {col.shape} != {true_shape}"
        # Predicted classes are non-negative integers.
        assert (col >= 0).all()
        assert np.array_equal(col, col.astype(np.int64))
    assert true_shape[0] == n_trials


def test_sequence_model_emits_multiple_windows(tmp_path: Path) -> None:
    """A sequence model (C_optimal) writes >1 window — the placeholder is gone."""
    _, model, loader, _ = _build_model_and_loader("C_optimal_synthetic")
    out_path = tmp_path / "CM_Table.mat"
    _write_cm_table_for_split(model, loader, out_path)
    assert len(_load_cm_windows(out_path)) > 1
