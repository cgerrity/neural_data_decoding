"""Checkpoint state machine for resume-from-interruption.

Mirrors ``cgg_getIterationInformation.m`` / ``cgg_saveIterationInformation.m``
with two intentional behaviors carried over from the MATLAB pipeline:

1. **Optimizer state is NOT saved** (Critical Note #3). The original MATLAB
   pipeline deliberately commented out the line that would persist
   ``OptimizerVariables.mat`` to keep checkpoint sizes manageable. On resume,
   the optimizer is reinitialized — meaning interrupt-then-resume produces a
   slightly different trajectory than uninterrupted training. Parity tests
   account for this.

2. **Resume reads "Current", never "Optimal"** (Critical Note #2). Two
   parallel snapshots are tracked:

   * ``current_state.pt`` — written on every save; used for resuming an
     interrupted run. Contains weights + epoch + iteration + best metric.
   * ``optimal_state.pt`` — written only when a new best validation metric
     beats the previous best. Used by downstream evaluation, never for
     resume.

The "save model weights but not optimizer state" decision means resume
restarts the optimizer from scratch. For ``AdamW`` this resets the first /
second moment estimates; the first few iterations after resume will feel
like the very beginning of training, then stabilize. Document in the ADR.

Examples
--------
>>> import torch
>>> from pathlib import Path
>>> import tempfile
>>> model = torch.nn.Linear(4, 2)
>>> with tempfile.TemporaryDirectory() as tmp:
...     ckpt_dir = Path(tmp)
...     save_current_checkpoint(ckpt_dir, model=model, epoch=3, iteration=42,
...                              best_metric=0.81)
...     state = load_current_checkpoint(ckpt_dir, model=model)
...     state.epoch, state.iteration, state.best_metric
(3, 42, 0.81)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


CURRENT_CHECKPOINT_FILENAME = "current_state.pt"
OPTIMAL_CHECKPOINT_FILENAME = "optimal_state.pt"


@dataclass(slots=True)
class CheckpointState:
    """The resume-relevant state restored from a checkpoint.

    Attributes
    ----------
    epoch
        Last completed epoch (0-indexed Python convention; add 1 to compare
        against MATLAB epoch counters).
    iteration
        Last completed minibatch iteration within the run (monotonic across
        epochs).
    best_metric
        Best validation metric observed so far. Used to gate Optimal saves.
    """

    epoch: int
    iteration: int
    best_metric: float


def save_current_checkpoint(
    checkpoint_dir: Path,
    *,
    model: nn.Module,
    epoch: int,
    iteration: int,
    best_metric: float,
) -> Path:
    """Write the ``current_state.pt`` resume snapshot.

    Called at the end of every epoch (or every N iterations) so that an
    interrupted run can pick up from the most recent save.

    Parameters
    ----------
    checkpoint_dir
        Directory to write into. Created if missing.
    model
        Module whose ``state_dict()`` is persisted.
    epoch
        Last completed epoch number.
    iteration
        Last completed minibatch iteration (monotonic across epochs).
    best_metric
        Best validation metric so far. Used to compare on Optimal updates.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / CURRENT_CHECKPOINT_FILENAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "iteration": int(iteration),
            "best_metric": float(best_metric),
        },
        path,
    )
    return path


def save_optimal_checkpoint(
    checkpoint_dir: Path,
    *,
    model: nn.Module,
    epoch: int,
    metric: float,
) -> Path:
    """Write the ``optimal_state.pt`` snapshot — best validation metric so far.

    Called only when a new best validation metric has been observed. The
    Optimal snapshot is **not** used for resume — only for downstream
    evaluation (matches MATLAB's behavior, where Optimal is the "best
    model" snapshot and Current is the "last seen" snapshot).

    Parameters
    ----------
    checkpoint_dir
        Directory to write into. Created if missing.
    model
        Module whose ``state_dict()`` is persisted.
    epoch
        Epoch at which the best metric was achieved.
    metric
        The validation metric that triggered the Optimal write.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / OPTIMAL_CHECKPOINT_FILENAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "metric": float(metric),
        },
        path,
    )
    return path


def load_current_checkpoint(
    checkpoint_dir: Path, *, model: nn.Module
) -> Optional[CheckpointState]:
    """Restore ``current_state.pt`` into ``model`` (in-place).

    Parameters
    ----------
    checkpoint_dir
        Directory to look in.
    model
        Module to load weights into. Modified in-place.

    Returns
    -------
    CheckpointState or None
        ``None`` if no checkpoint exists (fresh run). Otherwise the resume
        state — caller starts the next epoch at ``state.epoch + 1``.

    Notes
    -----
    On resume the optimizer is **not** restored — the caller must
    instantiate a fresh optimizer. See module docstring for rationale.
    """
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / CURRENT_CHECKPOINT_FILENAME
    if not path.exists():
        return None

    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    return CheckpointState(
        epoch=int(payload["epoch"]),
        iteration=int(payload["iteration"]),
        best_metric=float(payload["best_metric"]),
    )


def has_existing_checkpoint(checkpoint_dir: Path) -> bool:
    """Return ``True`` if any checkpoint file exists in ``checkpoint_dir``.

    Used by the pre-flight check (Critical Note #22) to abort training
    rather than silently overwrite a previous run.

    Parameters
    ----------
    checkpoint_dir
        Directory to check. Need not exist.

    Returns
    -------
    bool
        Whether any of the known checkpoint filenames exist in the directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return False
    return any(
        (checkpoint_dir / name).exists()
        for name in (CURRENT_CHECKPOINT_FILENAME, OPTIMAL_CHECKPOINT_FILENAME)
    )


__all__ = [
    "CURRENT_CHECKPOINT_FILENAME",
    "CheckpointState",
    "OPTIMAL_CHECKPOINT_FILENAME",
    "has_existing_checkpoint",
    "load_current_checkpoint",
    "load_optimal_checkpoint",
    "save_current_checkpoint",
    "save_optimal_checkpoint",
]


def load_optimal_checkpoint(
    checkpoint_dir: Path, *, model: nn.Module
) -> Optional[dict]:
    """Restore ``optimal_state.pt`` into ``model`` (in-place).

    Used by downstream evaluation scripts that want the best-validation
    snapshot, not the most-recent one. NEVER call this from the resume
    path — see :func:`load_current_checkpoint` for that.

    Parameters
    ----------
    checkpoint_dir
        Directory to look in.
    model
        Module to load weights into. Modified in-place.

    Returns
    -------
    dict or None
        ``None`` if no Optimal checkpoint exists. Otherwise the payload's
        ``{"epoch", "metric"}`` keys for telemetry.
    """
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / OPTIMAL_CHECKPOINT_FILENAME
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    return {"epoch": int(payload["epoch"]), "metric": float(payload["metric"])}
