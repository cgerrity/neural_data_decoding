"""Single-stage training & validation kernels.

Ports the per-iteration body of ``cgg_trainNetwork.m`` — the simplified
classifier-only variant used by Milestone A. The two-stage state machine
that decides *when* to call these kernels lives in :mod:`lifecycle`.

Critical Note #28 is the load-bearing decision here: there is **one**
``total_loss.backward()`` call per iteration, on the single aggregate scalar
returned by :func:`~neural_data_decoding.training.losses.multi_objective.aggregate_total_loss`.
Encoder, decoder, and classifier (when all three exist) gradient-flow from
that one scalar via autograd; they do **not** each call ``.backward()``
separately on intermediate sums.

Milestone A only exercises the classifier path. Hooks for the
reconstruction / KL / confidence / offset_and_scale components are present
in :mod:`losses.multi_objective` but unused here — Milestone C will add the
encoder/decoder forward call and feed those components into the aggregator.

Examples
--------
The kernel is straightforward to invoke once the model and dataloader are
ready::

    metrics, priors = train_one_epoch(
        model=classifier,
        dataloader=train_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        grad_clip_norm=1.0,
    )
    print(metrics.total_loss, metrics.classification_loss, metrics.accuracy)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_data_decoding.models.composite import AutoencoderOutput, VariationalOutput
from neural_data_decoding.training.losses.classification import (
    interpolated_multi_head_cross_entropy,
    multi_head_cross_entropy,
)
from neural_data_decoding.training.losses.confidence import (
    ConfidenceHistory,
    apply_confidence_routing,
)
from neural_data_decoding.training.losses.elbo import (
    kl_divergence_loss,
    masked_mse_reconstruction_loss,
)
from neural_data_decoding.training.losses.multi_objective import (
    LossPriors,
    aggregate_normalized_losses,
    aggregate_total_loss,
)


@dataclass(slots=True)
class EpochMetrics:
    """Per-epoch aggregate metrics returned by the train / validate kernels.

    Attributes
    ----------
    total_loss
        Mean (across iterations) of the aggregated multi-objective loss.
    classification_loss
        Mean cross-entropy contribution. Always populated.
    accuracy
        Mean per-dimension classification accuracy (averaged across both
        dimensions and trials). 0.0 when ``targets`` is missing.
    num_iterations
        Number of minibatches processed.
    num_trials
        Total trial count across all minibatches (sum of batch sizes).
    """

    total_loss: float
    classification_loss: float
    accuracy: float
    num_iterations: int
    num_trials: int


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Mapping[str, float],
    class_weights_per_dim: Optional[list[torch.Tensor]] = None,
    grad_clip_norm: Optional[float] = None,
    loss_priors: Optional[LossPriors] = None,
    prior_proportion: float = 0.9,
    update_priors: bool = True,
    update_priors_strategy: Optional[Literal["every_iter", "first_iter_only", "never"]] = None,
    confidence_history: Optional[ConfidenceHistory] = None,
) -> tuple[EpochMetrics, Optional[LossPriors], Optional[ConfidenceHistory]]:
    """Run one supervised training epoch — Milestone A classifier path.

    Parameters
    ----------
    model
        Classifier module. For Milestone A this is a
        :class:`~neural_data_decoding.models.classifier.MultiHeadClassifier`
        directly; later milestones will wrap encoder+decoder+classifier in
        a composite module.
    dataloader
        Yields the dict produced by
        :func:`~neural_data_decoding.data.dataset.collate_trials`:
        ``{"x": (B, T, F), "targets": (B, D), "metadata": [...]}``.
    optimizer
        Already constructed; weight decay handled via ``AdamW`` (Critical
        Note #5 — not via post-hoc gradient hooks).
    device
        Device to move tensors to before forward pass.
    loss_weights
        Per-component weight dict passed to
        :func:`~neural_data_decoding.training.losses.multi_objective.aggregate_total_loss`.
        Milestone A only needs ``{"classification": <w>}``; extra keys are
        forwarded harmlessly.
    class_weights_per_dim
        Optional per-dimension inverse-frequency class weights. Compute
        once from the training labels (see
        :func:`~neural_data_decoding.training.losses.classification.inverse_frequency_class_weights`)
        and pass on every epoch.
    grad_clip_norm
        Global L2 gradient-norm clip threshold (Critical Note: matches
        MATLAB's ``GradientClipType='Global'``). ``None`` disables clipping.
    loss_priors
        Optional :class:`LossPriors` carrying EMA prior state across
        iterations (Milestone C+). When provided, the orchestrator uses
        :func:`aggregate_normalized_losses` (cross-component normalization
        via classification's prior). When ``None``, falls back to the
        simple weighted sum used by Milestones A/B.
    prior_proportion
        EMA smoothing factor passed through when ``loss_priors`` is provided.
    update_priors
        Whether each iteration EMA-updates the priors. Ignored when
        ``update_priors_strategy`` is set. Retained for backward
        compatibility with earlier call sites that don't pass a strategy.
    update_priors_strategy
        Per-iteration EMA-update cadence within this epoch:

        * ``"every_iter"`` (default) — every iteration updates the priors.
          Matches MATLAB ``RescaleLossEpoch == 0``.
        * ``"first_iter_only"`` — only the first batch updates the priors.
          Matches MATLAB ``RescaleLossEpoch >= 1`` on an epoch where
          ``mod(Epoch+1, RescaleLossEpoch) == 1``.
        * ``"never"`` — no iteration updates the priors. Matches MATLAB
          ``RescaleLossEpoch > 1`` on a non-update epoch.
    confidence_history
        Optional :class:`ConfidenceHistory` (Milestone C #7). When
        present AND the model produces confidence outputs (trial /
        task fields on :class:`VariationalOutput`), each iteration
        calls :func:`apply_confidence_routing` to compute the per-trial
        / per-task / total confidence losses + the Beta P-controller
        update; the confidence loss is then folded into the aggregator
        as the ``confidence`` component scaled by the live ``Beta``.

    Returns
    -------
    EpochMetrics
        Aggregated training metrics over this epoch.
    LossPriors or None
        The priors state after the epoch (or ``None`` if no priors were
        provided). Caller persists this for the next epoch.
    ConfidenceHistory or None
        The confidence history state after the epoch (or ``None`` if no
        confidence history was provided). Caller persists this for the
        next epoch.
    """
    model.train()
    sums = _MetricSums()

    # Translate the legacy bool into the strategy enum unless the caller
    # passed an explicit strategy.
    if update_priors_strategy is None:
        update_priors_strategy = "every_iter" if update_priors else "never"

    for batch_idx, batch in enumerate(dataloader):
        update_priors_this_iter = _should_update_priors(
            update_priors_strategy, batch_idx,
        )
        x = batch["x"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        batch_size = int(x.shape[0])

        optimizer.zero_grad(set_to_none=True)
        out = model(x)

        # Detect variational composite vs classifier-only output.
        if isinstance(out, VariationalOutput):
            logits = out.logits
            cls_loss = multi_head_cross_entropy(
                logits, targets, class_weights_per_dim=class_weights_per_dim
            )
            recon_loss: Optional[torch.Tensor] = None
            kl_loss: Optional[torch.Tensor] = None
            if out.reconstruction is not None:
                recon_loss = masked_mse_reconstruction_loss(
                    out.reconstruction, x, batch_dim=0
                )
            kl_loss = kl_divergence_loss(out.mu, out.logvar, channel_dim=-1)

            # Confidence (Milestone C #7 + #7b): if the model produces
            # trial or task confidence AND the caller threaded a
            # ConfidenceHistory state, advance the Beta P-controller,
            # EMAs, and per-branch losses. The summed branch losses
            # become the confidence component; Beta scales it inside the
            # prior normalizer. The dropped per-dim total confidence
            # additionally drives Eq. 2 interpolated CE on the
            # classification loss (C #7b — replaces the standard CE
            # computed above when confidence is active).
            confidence_loss: Optional[torch.Tensor] = None
            confidence_beta_scalar: float = 1.0
            if confidence_history is not None and (
                out.trial_confidence is not None or out.task_confidence is not None
            ):
                cb = apply_confidence_routing(
                    y=logits[0], target=logits[0].detach(),  # ignored: see below
                    trial_confidence=out.trial_confidence,
                    task_confidence=out.task_confidence,
                    history=confidence_history,
                    batch_fraction=1.0,
                    compute_interpolation=False,
                    loss_type="L1",
                )
                # Sum the three branch losses → single confidence component.
                # Matches MATLAB cgg_getConfidenceLossInformation's
                # Loss_Confidence = TrialConf + TaskConf + TotalConf summation.
                confidence_loss = cb.total_loss + cb.trial_loss + cb.task_loss
                confidence_beta_scalar = float(cb.updated_history.beta)
                confidence_history = cb.updated_history

                # Eq. 2 interpolated CE replaces the standard CE on raw
                # logits. Matches MATLAB cgg_lossClassification's
                # crossentropy(Y_interpolated, T) on the post-dropout
                # interpolated probabilities. Uses the SAME dropped
                # confidence the branch-loss path used (via cb.total_dropped),
                # so dropout consistency is preserved per call.
                if cb.total_dropped is not None:
                    cls_loss = interpolated_multi_head_cross_entropy(
                        logits, targets, cb.total_dropped,
                        class_weights_per_dim=class_weights_per_dim,
                    )

            if loss_priors is not None:
                breakdown = aggregate_normalized_losses(
                    reconstruction_loss=recon_loss,
                    kl_loss=kl_loss,
                    classification_loss=cls_loss,
                    confidence_loss=confidence_loss,
                    weights=loss_weights,
                    priors=loss_priors,
                    prior_proportion=prior_proportion,
                    update_priors=update_priors_this_iter,
                    confidence_beta=confidence_beta_scalar,
                )
                total_loss = breakdown.total
                loss_priors = breakdown.updated_priors
            else:
                # No EMA priors → simple weighted sum (Milestone C without
                # the full orchestrator wired in yet).
                total_loss, _ = aggregate_total_loss(
                    classification_loss=cls_loss,
                    reconstruction_loss=recon_loss,
                    kl_loss=kl_loss,
                    confidence_loss=confidence_loss,
                    weights=loss_weights,
                )
        else:
            # Milestone A/B: model(x) returns list[Tensor] of logits.
            logits = out
            cls_loss = multi_head_cross_entropy(
                logits, targets, class_weights_per_dim=class_weights_per_dim
            )
            total_loss, _ = aggregate_total_loss(
                classification_loss=cls_loss, weights=loss_weights
            )

        total_loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        sums.update(
            total_loss=total_loss.item(),
            classification_loss=cls_loss.item(),
            accuracy=_per_dim_accuracy(logits, targets),
            batch_size=batch_size,
        )

    return sums.finalize(), loss_priors, confidence_history


@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_weights: Mapping[str, float],
    class_weights_per_dim: Optional[list[torch.Tensor]] = None,
) -> EpochMetrics:
    """Run one validation pass — no gradients, no BN updates (Critical Note #34).

    Parameters
    ----------
    model
        Same module as training; will be put in ``.eval()`` mode here.
    dataloader
        Validation dataloader (typically a different split than train).
    device
        Device to move tensors to.
    loss_weights
        Same weight dict as training, so the reported validation loss is
        comparable.
    class_weights_per_dim
        Class weights — typically the **training** distribution's weights
        (so the metric reflects training-aware loss); MATLAB does the same.

    Returns
    -------
    EpochMetrics
        Aggregated validation metrics over the dataloader.
    """
    model.eval()
    sums = _MetricSums()

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        batch_size = int(x.shape[0])

        out = model(x)
        # Detect variational vs classifier-only output (same logic as train).
        if isinstance(out, VariationalOutput):
            logits = out.logits
            cls_loss = multi_head_cross_entropy(
                logits, targets, class_weights_per_dim=class_weights_per_dim
            )
            recon_loss = None
            if out.reconstruction is not None:
                recon_loss = masked_mse_reconstruction_loss(
                    out.reconstruction, x, batch_dim=0
                )
            kl_loss = kl_divergence_loss(out.mu, out.logvar, channel_dim=-1)
            total_loss, _ = aggregate_total_loss(
                classification_loss=cls_loss,
                reconstruction_loss=recon_loss,
                kl_loss=kl_loss,
                weights=loss_weights,
            )
        else:
            logits = out
            cls_loss = multi_head_cross_entropy(
                logits, targets, class_weights_per_dim=class_weights_per_dim
            )
            total_loss, _ = aggregate_total_loss(
                classification_loss=cls_loss, weights=loss_weights
            )

        sums.update(
            total_loss=total_loss.item(),
            classification_loss=cls_loss.item(),
            accuracy=_per_dim_accuracy(logits, targets),
            batch_size=batch_size,
        )

    return sums.finalize()


def _should_update_priors(
    strategy: Literal["every_iter", "first_iter_only", "never"], batch_idx: int,
) -> bool:
    """Whether to update EMA priors on the iteration with index ``batch_idx``.

    Encodes the MATLAB ``RescaleLossEpoch`` cadence (Critical Note #6):
    every iteration, only the first iteration of the epoch, or never.
    """
    if strategy == "every_iter":
        return True
    if strategy == "first_iter_only":
        return batch_idx == 0
    return False


def _per_dim_accuracy(
    logits_per_dim: list[torch.Tensor], targets: torch.Tensor
) -> float:
    """Mean classification accuracy across dimensions and trials.

    For sequence logits ``(B, T, K)`` predictions are taken at the **last
    time step** — matches MATLAB's ``cgg_getLastSequenceValue`` convention
    used by the confidence and prediction kernels (Critical Note #36).
    """
    num_dims = len(logits_per_dim)
    if num_dims == 0:
        return 0.0
    total_correct = 0.0
    total = 0.0
    for d, logits in enumerate(logits_per_dim):
        if logits.ndim == 3:
            logits = logits[:, -1, :]  # (B, K) — last time step
        preds = logits.argmax(dim=-1)  # (B,)
        total_correct += (preds == targets[:, d]).float().sum().item()
        total += float(preds.shape[0])
    return total_correct / total if total > 0 else 0.0


class _MetricSums:
    """Running sums for :class:`EpochMetrics` assembly."""

    __slots__ = (
        "total_loss",
        "classification_loss",
        "accuracy_weighted",
        "iterations",
        "trials",
    )

    def __init__(self) -> None:
        self.total_loss = 0.0
        self.classification_loss = 0.0
        self.accuracy_weighted = 0.0
        self.iterations = 0
        self.trials = 0

    def update(
        self,
        *,
        total_loss: float,
        classification_loss: float,
        accuracy: float,
        batch_size: int,
    ) -> None:
        """Accumulate per-iteration metrics."""
        self.total_loss += total_loss
        self.classification_loss += classification_loss
        # Weight accuracy by batch_size so different-sized batches don't skew it.
        self.accuracy_weighted += accuracy * batch_size
        self.iterations += 1
        self.trials += batch_size

    def finalize(self) -> EpochMetrics:
        """Compute mean metrics across iterations."""
        iters = max(self.iterations, 1)
        trials = max(self.trials, 1)
        return EpochMetrics(
            total_loss=self.total_loss / iters,
            classification_loss=self.classification_loss / iters,
            accuracy=self.accuracy_weighted / trials,
            num_iterations=self.iterations,
            num_trials=self.trials,
        )


@dataclass(slots=True)
class UnsupervisedEpochMetrics:
    """Per-epoch aggregate metrics for Stage 1 (autoencoder-only).

    Mirrors :class:`EpochMetrics` but with reconstruction + KL components
    instead of classification + accuracy. No classification accuracy field
    because Stage 1 has no classifier.

    Attributes
    ----------
    total_loss
        Mean (across iterations) of the weighted ``recon + KL`` sum.
    reconstruction_loss
        Mean reconstruction component.
    kl_loss
        Mean KL divergence component.
    num_iterations, num_trials
        Same as :class:`EpochMetrics`.
    """

    total_loss: float
    reconstruction_loss: float
    kl_loss: float
    num_iterations: int
    num_trials: int


def train_unsupervised_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Mapping[str, float],
    grad_clip_norm: Optional[float] = None,
) -> UnsupervisedEpochMetrics:
    """Run one Stage 1 unsupervised training epoch.

    Model must produce :class:`AutoencoderOutput` (encoder + bottleneck +
    sampling + decoder, no classifier). Computes the NaN-masked MSE
    reconstruction loss and the KL divergence on the latent, then their
    weighted sum.

    Parameters
    ----------
    model
        :class:`~neural_data_decoding.models.composite.VariationalAutoencoder`
        (or any module whose ``forward`` returns an :class:`AutoencoderOutput`).
    dataloader
        Yields ``collate_trials`` dicts; ``targets`` is ignored.
    optimizer
        Already constructed.
    device
        Where to put tensors.
    loss_weights
        Per-component weight dict; expected keys ``"reconstruction"`` and
        ``"kl"``. Missing keys default to 1.0; extra keys are ignored.
    grad_clip_norm
        Global L2 gradient-norm clip threshold. ``None`` disables.

    Returns
    -------
    UnsupervisedEpochMetrics
    """
    model.train()
    sum_total = 0.0
    sum_recon = 0.0
    sum_kl = 0.0
    iterations = 0
    trials = 0
    w_recon = float(loss_weights.get("reconstruction", 1.0))
    w_kl = float(loss_weights.get("kl", 1.0))

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        batch_size = int(x.shape[0])

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        if not isinstance(out, AutoencoderOutput):
            raise TypeError(
                f"train_unsupervised_epoch expected AutoencoderOutput; "
                f"got {type(out).__name__}."
            )
        recon_loss = masked_mse_reconstruction_loss(out.reconstruction, x, batch_dim=0)
        kl_loss = kl_divergence_loss(out.mu, out.logvar, channel_dim=-1)
        total_loss = w_recon * recon_loss + w_kl * kl_loss
        total_loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        sum_total += float(total_loss.item())
        sum_recon += float(recon_loss.item())
        sum_kl += float(kl_loss.item())
        iterations += 1
        trials += batch_size

    iters = max(iterations, 1)
    return UnsupervisedEpochMetrics(
        total_loss=sum_total / iters,
        reconstruction_loss=sum_recon / iters,
        kl_loss=sum_kl / iters,
        num_iterations=iterations,
        num_trials=trials,
    )


@torch.no_grad()
def validate_unsupervised(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_weights: Mapping[str, float],
) -> UnsupervisedEpochMetrics:
    """Stage 1 validation pass — no gradients, no priors, no classification."""
    model.eval()
    sum_total = 0.0
    sum_recon = 0.0
    sum_kl = 0.0
    iterations = 0
    trials = 0
    w_recon = float(loss_weights.get("reconstruction", 1.0))
    w_kl = float(loss_weights.get("kl", 1.0))

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        batch_size = int(x.shape[0])

        out = model(x)
        if not isinstance(out, AutoencoderOutput):
            raise TypeError(
                f"validate_unsupervised expected AutoencoderOutput; "
                f"got {type(out).__name__}."
            )
        recon_loss = masked_mse_reconstruction_loss(out.reconstruction, x, batch_dim=0)
        kl_loss = kl_divergence_loss(out.mu, out.logvar, channel_dim=-1)
        total_loss = w_recon * recon_loss + w_kl * kl_loss

        sum_total += float(total_loss.item())
        sum_recon += float(recon_loss.item())
        sum_kl += float(kl_loss.item())
        iterations += 1
        trials += batch_size

    iters = max(iterations, 1)
    return UnsupervisedEpochMetrics(
        total_loss=sum_total / iters,
        reconstruction_loss=sum_recon / iters,
        kl_loss=sum_kl / iters,
        num_iterations=iterations,
        num_trials=trials,
    )


__all__ = [
    "EpochMetrics",
    "UnsupervisedEpochMetrics",
    "train_one_epoch",
    "train_unsupervised_epoch",
    "validate",
    "validate_unsupervised",
]
