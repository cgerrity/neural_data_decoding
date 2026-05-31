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
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_data_decoding.models.composite import (
    VariationalAutoencoder,
    VariationalComposite,
    copy_autoencoder_weights,
)
from neural_data_decoding.training.checkpoint import (
    load_current_checkpoint,
    load_optimal_checkpoint,
    save_current_checkpoint,
    save_optimal_checkpoint,
)
from neural_data_decoding.training.freezing import apply_freeze_to_optimizer
from neural_data_decoding.training.loop import (
    EpochMetrics,
    UnsupervisedEpochMetrics,
    train_one_epoch,
    train_unsupervised_epoch,
    validate,
    validate_unsupervised,
)
from neural_data_decoding.training.losses.confidence import ConfidenceHistory
from neural_data_decoding.training.losses.multi_objective import LossPriors
from neural_data_decoding.training.schedules import CurriculumBundle


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
OnOptimalCallback = Callable[[nn.Module, "EpochHistory"], None]
"""Hook called whenever the current epoch becomes the new best validation
metric. Receives the current model (with the just-saved Optimal weights)
and the populated :class:`EpochHistory` for the epoch. Used by the CLI to
write the validation + test CM_Tables — both reflect the optimal model's
predictions and are inspectable mid-training (matches MATLAB's
``cgg_saveValidationCMTable`` + ``cgg_saveCMTableFromSeparateNetwork``
pattern, both gated on ``IsOptimal``)."""


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
    on_optimal_callback: Optional[OnOptimalCallback] = None,
    loss_priors: Optional[LossPriors] = None,
    prior_proportion: float = 0.9,
    curriculum: Optional[CurriculumBundle] = None,
    freeze_base_lr: Optional[float] = None,
    rescale_loss_epoch: int = 0,
    confidence_history: Optional["ConfidenceHistory"] = None,
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
    curriculum
        Optional :class:`CurriculumBundle` (Milestone C #5). When present,
        ``curriculum.update(epoch + 1)`` is called at the start of every
        epoch (the ``+1`` converts Python's 0-indexed loop into MATLAB's
        1-indexed convention). The bundle's weight schedule then drives
        the per-epoch ``loss_weights`` dict (overriding the static one
        passed in for component keys it knows about), and the freeze
        schedule sets per-group learning rates via
        :func:`apply_freeze_to_optimizer` (provided
        ``freeze_base_lr`` is set).
    freeze_base_lr
        Reference learning rate for the freeze applier. Typically the
        config's ``initial_learning_rate``. Required when ``curriculum``
        is present and its freeze schedule should affect the optimizer;
        a ``None`` here means the freeze schedule is computed but not
        applied (useful for testing).
    rescale_loss_epoch
        MATLAB ``RescaleLossEpoch`` cadence for EMA-prior updates
        (Critical Note #6):

        * ``0`` — update every iteration (default; matches the prior
          Python behavior).
        * ``1`` — update only the first iteration of every epoch.
        * ``N > 1`` — update only the first iteration of epochs
          ``1, N+1, 2N+1, ...``.

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
        # MATLAB ordering (cgg_trainNetwork.m:484-502): update dynamic
        # parameters first, then apply freeze, then enter the mini-batch loop.
        if curriculum is not None:
            curriculum.update(epoch + 1)   # +1: Python 0-indexed → MATLAB 1-indexed.
            if freeze_base_lr is not None:
                apply_freeze_to_optimizer(
                    optimizer, curriculum.freeze, base_lr=freeze_base_lr,
                )

        epoch_loss_weights = _resolve_epoch_loss_weights(loss_weights, curriculum)
        strategy = _update_priors_strategy_for(epoch, rescale_loss_epoch)

        train_metrics, loss_priors, confidence_history = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_weights=epoch_loss_weights,
            class_weights_per_dim=class_weights_per_dim,
            grad_clip_norm=grad_clip_norm,
            loss_priors=loss_priors,
            prior_proportion=prior_proportion,
            update_priors_strategy=strategy,
            confidence_history=confidence_history,
        )
        iteration += train_metrics.num_iterations

        val_metrics: Optional[EpochMetrics] = None
        is_best = False
        if val_loader is not None:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                loss_weights=epoch_loss_weights,
                class_weights_per_dim=class_weights_per_dim,
                confidence_history=confidence_history,
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

        # MATLAB pattern (cgg_trainNetwork.m:636-641): when IsOptimal, write
        # both CM_Tables with the just-saved Optimal weights. The on-disk
        # files always reflect the best-so-far model — inspectable
        # mid-training, no end-of-training restore-and-recompute dance.
        if is_best and on_optimal_callback is not None:
            on_optimal_callback(model, entry)

        if epoch_callback is not None:
            epoch_callback(entry)

    return history


@dataclass(slots=True)
class UnsupervisedEpochHistory:
    """One Stage 1 epoch's train + validation metrics, plus best-tracking.

    Mirrors :class:`EpochHistory` but for Stage 1 unsupervised pre-training.
    The ``is_best`` flag here is set on **minimum** validation loss (lower
    is better), in contrast to Stage 2's accuracy-maximizing criterion.
    """

    epoch: int
    train: UnsupervisedEpochMetrics
    val: Optional[UnsupervisedEpochMetrics]
    is_best: bool


def fit_unsupervised(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    loss_weights: Mapping[str, float],
    checkpoint_dir: Path,
    grad_clip_norm: Optional[float] = None,
    epoch_callback: Optional[Callable[[UnsupervisedEpochHistory], None]] = None,
    curriculum: Optional[CurriculumBundle] = None,
    freeze_base_lr: Optional[float] = None,
) -> list[UnsupervisedEpochHistory]:
    """Run Stage 1 unsupervised autoencoder pre-training.

    Same shape as :func:`fit_supervised` but:

    * Model produces :class:`AutoencoderOutput` (no logits).
    * Loss is the weighted sum of reconstruction + KL only.
    * "Best" is **min** validation loss (lower is better) — naturally
      asymmetric with Stage 2's "max accuracy". Optimal weights are saved
      whenever val_loss strictly drops.
    * No EMA prior normalization (those exist to balance recon/KL against
      classification, which is absent here).

    Stage 1 owns its own ``checkpoint_dir`` (typically
    ``<result_dir>/stage1_autoencoder/``); the caller is responsible for
    creating that subdirectory if needed.

    Parameters
    ----------
    See :func:`fit_supervised`. ``loss_priors``, ``prior_proportion``, and
    ``rescale_loss_epoch`` are not accepted here — Stage 1 doesn't need
    EMA priors.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resumed = load_current_checkpoint(checkpoint_dir, model=model)
    start_epoch = resumed.epoch + 1 if resumed is not None else 0
    # best_metric stored as a min-comparison value; +Inf so the first
    # validation always strictly beats it.
    best_metric = (
        resumed.best_metric if resumed is not None else float("inf")
    )
    iteration = resumed.iteration if resumed is not None else 0

    history: list[UnsupervisedEpochHistory] = []
    for epoch in range(start_epoch, num_epochs):
        if curriculum is not None:
            curriculum.update(epoch + 1)
            if freeze_base_lr is not None:
                apply_freeze_to_optimizer(
                    optimizer, curriculum.freeze, base_lr=freeze_base_lr,
                )

        epoch_loss_weights = _resolve_epoch_loss_weights(loss_weights, curriculum)

        train_metrics = train_unsupervised_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_weights=epoch_loss_weights,
            grad_clip_norm=grad_clip_norm,
        )
        iteration += train_metrics.num_iterations

        val_metrics: Optional[UnsupervisedEpochMetrics] = None
        is_best = False
        if val_loader is not None:
            val_metrics = validate_unsupervised(
                model=model,
                dataloader=val_loader,
                device=device,
                loss_weights=epoch_loss_weights,
            )
            if val_metrics.total_loss < best_metric:
                best_metric = val_metrics.total_loss
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

        entry = UnsupervisedEpochHistory(
            epoch=epoch, train=train_metrics, val=val_metrics, is_best=is_best,
        )
        history.append(entry)
        if epoch_callback is not None:
            epoch_callback(entry)

    return history


def fit_two_stage(
    *,
    autoencoder: VariationalAutoencoder,
    composite: VariationalComposite,
    stage1_optimizer: torch.optim.Optimizer,
    stage2_optimizer: torch.optim.Optimizer,
    stage1_num_epochs: int,
    stage2_num_epochs: int,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    loss_weights: Mapping[str, float],
    checkpoint_dir: Path,
    stage1_subdir: str = "stage1_autoencoder",
    class_weights_per_dim: Optional[list[torch.Tensor]] = None,
    grad_clip_norm: Optional[float] = None,
    stage1_epoch_callback: Optional[Callable[[UnsupervisedEpochHistory], None]] = None,
    stage2_epoch_callback: Optional[EpochCallback] = None,
    on_optimal_callback: Optional[OnOptimalCallback] = None,
    curriculum: Optional[CurriculumBundle] = None,
    freeze_base_lr: Optional[float] = None,
    rescale_loss_epoch: int = 0,
    loss_priors: Optional[LossPriors] = None,
    prior_proportion: float = 0.9,
) -> tuple[list[UnsupervisedEpochHistory], list[EpochHistory]]:
    """Stage 1 unsupervised pre-training → handoff → Stage 2 supervised.

    Sequences the two stages with the Optimal Stage 1 autoencoder weights
    handed off into Stage 2's encoder + bottleneck + decoder submodules,
    then Stage 2 trains the full composite (autoencoder + classifier).

    Stage 1 writes its checkpoints to
    ``<checkpoint_dir>/<stage1_subdir>/``; Stage 2 uses
    ``<checkpoint_dir>/`` directly. Resume is per-stage: re-running this
    function with Stage 1 already complete (resume reads
    ``stage1_subdir/`` Current snapshot) is a no-op on the unsupervised
    side, and Stage 2 picks up where it left off independently.

    Parameters
    ----------
    autoencoder
        :class:`~neural_data_decoding.models.composite.VariationalAutoencoder`.
        Trained in Stage 1; its Optimal weights are loaded back into this
        instance after Stage 1 completes, then copied into ``composite``.
    composite
        :class:`~neural_data_decoding.models.composite.VariationalComposite`.
        Receives the autoencoder weights via
        :func:`~neural_data_decoding.models.composite.copy_autoencoder_weights`,
        then trains in Stage 2.
    stage1_optimizer, stage2_optimizer
        Per-stage optimizers. Stage 2's optimizer **must** be built on the
        composite's parameters AFTER the weight handoff would have run, so
        the caller is responsible for ordering: build composite → call
        ``fit_two_stage`` (the weight handoff is in-place on the
        composite, so the optimizer's parameter references stay valid).
    stage1_num_epochs, stage2_num_epochs
        Target epoch counts per stage.
    loss_weights
        Static default weights. The curriculum's weight schedule overrides
        component keys it knows about, per epoch, per stage.
    checkpoint_dir
        Base directory. Stage 1's subdir is created if missing.
    stage1_subdir
        Subdirectory name for Stage 1 checkpoints. Defaults to
        ``"stage1_autoencoder"``.
    (other params)
        See :func:`fit_supervised` and :func:`fit_unsupervised`.

    Returns
    -------
    tuple[list[UnsupervisedEpochHistory], list[EpochHistory]]
        ``(stage1_history, stage2_history)``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    stage1_dir = checkpoint_dir / stage1_subdir

    stage1_history = fit_unsupervised(
        model=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=stage1_optimizer,
        num_epochs=stage1_num_epochs,
        device=device,
        loss_weights=loss_weights,
        checkpoint_dir=stage1_dir,
        grad_clip_norm=grad_clip_norm,
        epoch_callback=stage1_epoch_callback,
        curriculum=curriculum,
        freeze_base_lr=freeze_base_lr,
    )

    # Handoff: load Stage 1 Optimal weights into the autoencoder instance,
    # then copy encoder/bottleneck/sampling/decoder into the composite.
    # If no Optimal was ever written (e.g. val_loader was None), the
    # autoencoder's in-memory weights are already the latest — skip the load.
    optimal_loaded = load_optimal_checkpoint(stage1_dir, model=autoencoder)
    if optimal_loaded is None and val_loader is not None:
        # Should not happen in normal use; surface a clear warning.
        print(
            f"  ↳ Warning: Stage 1 finished but no Optimal snapshot at "
            f"{stage1_dir}; handing off last-epoch weights instead."
        )
    copy_autoencoder_weights(autoencoder, composite)

    stage2_history = fit_supervised(
        model=composite,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=stage2_optimizer,
        num_epochs=stage2_num_epochs,
        device=device,
        loss_weights=loss_weights,
        checkpoint_dir=checkpoint_dir,
        class_weights_per_dim=class_weights_per_dim,
        grad_clip_norm=grad_clip_norm,
        epoch_callback=stage2_epoch_callback,
        on_optimal_callback=on_optimal_callback,
        loss_priors=loss_priors,
        prior_proportion=prior_proportion,
        curriculum=curriculum,
        freeze_base_lr=freeze_base_lr,
        rescale_loss_epoch=rescale_loss_epoch,
    )

    return stage1_history, stage2_history


def _resolve_epoch_loss_weights(
    static_weights: Mapping[str, float],
    curriculum: Optional[CurriculumBundle],
) -> dict[str, float]:
    """Snapshot per-epoch loss weights, blending the static defaults with the curriculum.

    The static ``loss_weights`` dict provides defaults for any keys the
    curriculum's weight schedule does not manage; the schedule overrides
    keys it knows about with its live current values. Snapshotted once
    per epoch — magnitudes don't change mid-epoch in MATLAB either.
    """
    resolved: dict[str, float] = dict(static_weights)
    if curriculum is not None:
        for name in curriculum.weight:
            resolved[name] = curriculum.weight.current(name)
    return resolved


def _update_priors_strategy_for(
    epoch: int, rescale_loss_epoch: int,
) -> Literal["every_iter", "first_iter_only", "never"]:
    """Translate MATLAB ``RescaleLossEpoch`` into the loop's strategy string.

    See the docstring on :func:`fit_supervised` for the cadence semantics.
    """
    if rescale_loss_epoch <= 0:
        return "every_iter"
    if rescale_loss_epoch == 1:
        return "first_iter_only"
    # MATLAB: mod(Epoch+1, N) == 1, with Epoch 1-indexed. Python epoch is
    # 0-indexed, so use (epoch + 2) — i.e., MATLAB Epoch = epoch + 1, then
    # mod(MATLAB Epoch + 1, N) == 1.
    if (epoch + 2) % rescale_loss_epoch == 1:
        return "first_iter_only"
    return "never"


__all__ = [
    "EpochCallback",
    "EpochHistory",
    "OnOptimalCallback",
    "UnsupervisedEpochHistory",
    "fit_supervised",
    "fit_two_stage",
    "fit_unsupervised",
]
