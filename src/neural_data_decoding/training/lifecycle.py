"""Two-stage training lifecycle orchestrator.

Mirrors the decision tree in ``cgg_trainAllAutoEncoder_v2.m:171-221``:

* **Stage 1 — Unsupervised pre-training** of encoder+decoder only. The
  classifier is *deliberately not built* (``cgg_trainNetwork`` is called
  with no classifier argument, setting ``HasClassifier=false`` and skipping
  all classification forward/backward code). Runs for
  ``NumEpochsAutoEncoder`` epochs.
* **Stage 2 — Supervised fine-tuning** with the classifier built on top of
  the encoder's optimal pre-training weights. Runs for ``NumEpochsFull``
  epochs.

Current production "Optimal" has ``NumEpochsAutoEncoder=0`` so Stage 1 is
degenerate — but the orchestrator MUST handle the general case so future
configs can re-enable it. Critical Note #1 spells this out.

For **Milestone A** (Logistic Regression) only Stage 2 runs — there is no
encoder/decoder pair to pre-train. ``NumEpochsAutoEncoder`` is forced to 0
in the Milestone A config.

The fit loop here owns:

* Iterating epochs with the resume offset from
  :func:`~neural_data_decoding.training.checkpoint.load_current_checkpoint`.
* Calling :func:`~neural_data_decoding.training.loop.train_one_epoch` then
  :func:`~neural_data_decoding.training.loop.validate`.
* Persisting the Current snapshot every epoch (resume safety) and the
  Optimal snapshot whenever validation accuracy improves.

It does **not** own:

* Curriculum schedules (Milestone C — will be passed in as a callback).
* W&B / monitor logging (Milestone A: print to stdout; later milestones
  will add a logger argument).
* Confidence / VAE / multi-stage loss components (Milestone C+).

Examples
--------
For a logistic-regression Milestone A run::

    history = fit_supervised(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=10,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        checkpoint_dir=results_dir,
    )
    print(history[-1].val.accuracy)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_data_decoding.training.checkpoint import (
    load_current_checkpoint,
    save_current_checkpoint,
    save_optimal_checkpoint,
)
from neural_data_decoding.training.loop import EpochMetrics, train_one_epoch, validate


@dataclass(slots=True)
class EpochHistory:
    """One epoch's train + validation metrics, plus the resume bookkeeping.

    Attributes
    ----------
    epoch
        0-indexed epoch number for this entry.
    train
        Training metrics from :func:`train_one_epoch`.
    val
        Validation metrics from :func:`validate`. ``None`` when validation
        is skipped (e.g., during a pre-training stage with no val loader).
    is_best
        Whether this epoch's validation accuracy beats the previous best.
        Always ``False`` when ``val`` is ``None``.
    """

    epoch: int
    train: EpochMetrics
    val: Optional[EpochMetrics]
    is_best: bool


EpochCallback = Callable[[EpochHistory], None]


def fit_supervised(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    loss_weights: Mapping[str, float],
    checkpoint_dir: Path,
    class_weights_per_dim: Optional[list[torch.Tensor]] = None,
    grad_clip_norm: Optional[float] = None,
    epoch_callback: Optional[EpochCallback] = None,
) -> list[EpochHistory]:
    """Run the supervised Stage 2 fit loop end-to-end.

    Resume semantics: if ``checkpoint_dir`` already contains a Current
    snapshot, weights are loaded from it and training resumes at
    ``state.epoch + 1`` (Critical Note #2 — Resume reads Current, never
    Optimal). The optimizer is **not** restored (Critical Note #3 — the
    MATLAB pipeline intentionally drops optimizer state to keep checkpoint
    sizes manageable, so the first iteration after resume effectively
    restarts ``AdamW``'s moments from zero).

    Parameters
    ----------
    model
        Module to train. Modified in-place. For Milestone A this is the
        classifier; for Milestone B+ it's the composite encoder + classifier.
    train_loader
        Iterates the training-split minibatches.
    val_loader
        Iterates the validation-split minibatches. May be ``None`` (then
        validation is skipped — Stage 1 unsupervised use case).
    optimizer
        Already constructed. Use ``AdamW`` for the Milestone A defaults.
    num_epochs
        Target total epochs (NOT additional epochs — if resuming, only the
        remaining gap is run).
    device
        Where to put tensors.
    loss_weights
        Per-component weight dict (see :mod:`losses.multi_objective`).
    checkpoint_dir
        Directory for the Current and Optimal snapshots. Created if missing.
        Pre-flight checks should have already validated that running here
        won't silently clobber a previous run (Critical Note #22 — that
        check is the caller's responsibility, not this function's).
    class_weights_per_dim
        Inverse-frequency class weights. Compute once before calling fit
        and pass through.
    grad_clip_norm
        Global L2 gradient-norm threshold. ``None`` disables clipping.
    epoch_callback
        Optional hook called after each epoch with the populated
        :class:`EpochHistory`. Useful for W&B logging, schedule updates,
        early stopping in later milestones.

    Returns
    -------
    list of EpochHistory
        One entry per epoch executed in this call (excludes epochs already
        completed in a prior interrupted run).

    Notes
    -----
    Optimal (best-validation) snapshots are written only when
    ``val_loader`` is provided and the new validation accuracy strictly
    beats the previous best. The Current snapshot is written every epoch
    regardless, so resume always works.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resumed = load_current_checkpoint(checkpoint_dir, model=model)
    start_epoch = resumed.epoch + 1 if resumed is not None else 0
    best_metric = resumed.best_metric if resumed is not None else float("-inf")
    iteration = resumed.iteration if resumed is not None else 0

    history: list[EpochHistory] = []
    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            class_weights_per_dim=class_weights_per_dim,
            grad_clip_norm=grad_clip_norm,
        )
        iteration += train_metrics.num_iterations

        val_metrics: Optional[EpochMetrics] = None
        is_best = False
        if val_loader is not None:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                loss_weights=loss_weights,
                class_weights_per_dim=class_weights_per_dim,
            )
            if val_metrics.accuracy > best_metric:
                best_metric = val_metrics.accuracy
                is_best = True
                save_optimal_checkpoint(
                    checkpoint_dir,
                    model=model,
                    epoch=epoch,
                    metric=best_metric,
                )

        save_current_checkpoint(
            checkpoint_dir,
            model=model,
            epoch=epoch,
            iteration=iteration,
            best_metric=best_metric,
        )

        entry = EpochHistory(
            epoch=epoch, train=train_metrics, val=val_metrics, is_best=is_best
        )
        history.append(entry)
        if epoch_callback is not None:
            epoch_callback(entry)

    return history


__all__ = ["EpochCallback", "EpochHistory", "fit_supervised"]
