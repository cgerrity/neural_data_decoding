"""Confidence routing + PD-controller loss — port of ``cgg_lossConfidence.m``.

Critical Note #29 in the migration plan flags this as the **highest-risk
port in the entire migration**: five subtleties packed into one kernel,
each silently parity-loss-prone if the Python port drifts.

The five subtleties (each independently parity-tested)
------------------------------------------------------

1. **Multiplicative conjunction** (Eq. 1) — when both Task and Trial
   confidence are available, ``TotalConfidence = TaskConfidence
   .* TrialConfidence`` (element-wise). The TOTAL stream is what the
   prediction interpolation (Eq. 2) consumes.

2. **ConfidenceDropout** (default 0.5, separate from network dropout) — a
   parallel "dropped" path is built where, with probability
   ``1 - confidence_dropout``, each entry is reset to ``1``. The dropped
   version is used for the prediction-to-truth interpolation; the
   un-dropped version is used for the budget regularizer. So the dropout
   biases the *interpolation* toward "keep predicting" (confidence=1 means
   no interpolation) while still letting the regularizer see the real
   batch mean.

3. **Prediction-to-truth interpolation** (Eq. 2) — the classification
   prediction is interpolated with the target by confidence:
   ``Y' = TotalConfidence_Dropped * Y + (1 - TotalConfidence_Dropped) * T``.
   Low-confidence trials get their classifier loss pulled toward zero
   (predicting near-truth); the budget regularizer (push confidence toward
   1) prevents the trivial "always low confidence → no loss" solution.
   This is **the mechanism**, not postprocessing.

4. **Stop-gradient on historical EMA** — the dataset-level confidence
   history is detached from autograd (``cgg_extractData`` in MATLAB →
   ``.detach()`` here). Only the current batch's contribution flows
   gradient through the EMA. Without this, gradient would leak backward
   through the whole training history.

5. **BatchFraction-governed EMA cadence** (Eq. 7) —
   ``Updated = Historical * (1 - γ) + BatchMean * γ`` where ``γ =
   BatchFraction``. So a small batch contributes a small update; a
   full-dataset batch (γ = 1) completely replaces history with batch
   mean. **NOT** a fixed coefficient.

Optional **batch correction** (Eq. 10) scales the loss by ``1/γ`` — for
gradient-magnitude consistency across batch sizes when EMA is active.

Sequence input convention
-------------------------
Confidence heads emit sequence outputs (per-timestep). Only the **last
timestep** is consumed (Critical Note #36 — ports
``cgg_getLastSequenceValue``). The dropout / conjunction / interpolation
all operate on the last-timestep slice.

Examples
--------
>>> import torch
>>> history = ConfidenceHistory.initial()
>>> y = torch.randn(2, 3, 4)
>>> target = torch.randn(2, 3, 4)
>>> trial = torch.rand(2, 3, 1) * 0.5 + 0.3        # (B, T, 1)
>>> task  = torch.rand(2, 3, 4) * 0.5 + 0.3        # (B, T, K)
>>> out = apply_confidence_routing(
...     y, target, trial, task,
...     history=history,
...     confidence_dropout=1.0,                    # disable random reset
...     want_dataset_confidence=False,
... )
>>> out.y_interpolated.shape
torch.Size([2, 3, 4])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


LossType = Literal["L1", "L2", "L1 & L2", "CrossEntropy"]


@dataclass(slots=True)
class ConfidenceHistory:
    """Detached EMA state + P-controller Beta for the confidence streams.

    Each field holds a scalar tensor (the running mean confidence over the
    dataset for that stream, or the controller's Beta scalar). Values are
    detached from any autograd graph — see subtlety #4 in the module
    docstring.

    Attributes
    ----------
    total
        EMA of ``TotalConfidence`` (the conjuncted Task × Trial stream).
    trial
        EMA of ``TrialConfidence``.
    task
        EMA of ``TaskConfidence``.
    beta
        "Autonomous Equilibrium Controller" Beta (despite the MATLAB
        function's "PD-controller" name, the implementation is a pure
        P-controller). Multiplicatively scales the final confidence
        contribution to the total loss. Updated per batch via
        ``Beta_new = Beta_prev * (1 + (target - batch_mean) * rate)``,
        clamped to ``[beta_min, beta_max]``. See module-level constants
        and :func:`apply_confidence_routing`.
    """

    total: torch.Tensor
    trial: torch.Tensor
    task: torch.Tensor
    beta: torch.Tensor

    @classmethod
    def initial(
        cls,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        value: float = 1.0,
    ) -> "ConfidenceHistory":
        """Build the MATLAB-equivalent initial history (all ``1.0``).

        Mirrors ``cgg_lossConfidence.m`` lines 98-109 + ``cgg_getLossInformation.m``
        line 102: when history is empty / NaN, MATLAB initializes each EMA
        stream to 1.0 and Beta to 1.0. The same initialization happens here.
        """
        t = torch.full((), value, dtype=dtype, device=device)
        beta = torch.full((), 1.0, dtype=dtype, device=device)
        return cls(total=t.clone(), trial=t.clone(), task=t.clone(), beta=beta)


# Confidence_Beta P-controller hyperparameters (MATLAB
# cgg_getConfidenceLossInformation.m lines 10-15).
CONFIDENCE_BETA_MAX: float = 10.0
CONFIDENCE_BETA_MIN: float = 0.1
CONFIDENCE_BETA_TARGET: float = 0.5
CONFIDENCE_BETA_DIFFERENCE_RATE: float = 1.0


def _update_confidence_beta(
    beta_prev: torch.Tensor,
    total_batch_mean: torch.Tensor,
) -> torch.Tensor:
    """Advance the Confidence_Beta P-controller by one step.

    Mirrors ``cgg_getConfidenceLossInformation.m`` lines 60-75. The
    function is named "Autonomous Equilibrium Controller" in MATLAB but
    the implementation is a pure P-controller on the gap between the
    target (0.5) and the batch's mean TotalConfidence.

    Parameters
    ----------
    beta_prev
        The previous step's Beta (scalar tensor).
    total_batch_mean
        ``mean(TotalConfidence)`` for the current batch (scalar tensor).
        Use the **undropped** confidence (matching MATLAB).

    Returns
    -------
    torch.Tensor
        The clamped new Beta (detached).
    """
    diff = CONFIDENCE_BETA_TARGET - total_batch_mean
    new_beta = beta_prev * (1.0 + diff * CONFIDENCE_BETA_DIFFERENCE_RATE)
    new_beta = new_beta.clamp(CONFIDENCE_BETA_MIN, CONFIDENCE_BETA_MAX)
    return new_beta.detach()


@dataclass(slots=True)
class ConfidenceLossBreakdown:
    """Output of :func:`apply_confidence_routing`.

    Attributes
    ----------
    y_interpolated
        ``Y'`` from Eq. 2 — the confidence-weighted blend of the original
        prediction and the target. The caller feeds this into the
        classification cross-entropy.
    total_loss, trial_loss, task_loss
        The three branch losses (scalar). Sum these (or weight them) when
        assembling the multi-objective total.
    updated_history
        Fresh :class:`ConfidenceHistory` with the EMA-updated values, all
        detached. The caller persists this for the next minibatch.
    """

    y_interpolated: torch.Tensor
    total_loss: torch.Tensor
    trial_loss: torch.Tensor
    task_loss: torch.Tensor
    updated_history: ConfidenceHistory


def _last_timestep(x: torch.Tensor, *, time_dim: int) -> torch.Tensor:
    """Extract the last index along ``time_dim``, collapsing that axis.

    Ports MATLAB's ``cgg_getLastSequenceValue`` (which strips the 'T' axis
    after taking the final time index). Critical Note #36 mandates the
    last-timestep convention for confidence heads.
    """
    last_idx = x.shape[time_dim] - 1
    return x.select(time_dim, last_idx)


def _apply_confidence_dropout(
    confidence: torch.Tensor,
    confidence_dropout: float,
    *,
    mask: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Stochastically replace entries with ``1`` (subtlety #2).

    Mirrors MATLAB::

        DropoutMask = rand(size, "like", x) > ConfidenceDropout
        confidence_dropped[DropoutMask] = 1

    With ``confidence_dropout=0.5``, ~50% of entries get reset to 1.
    With ``confidence_dropout=1.0``, no resets (mask always False).
    With ``confidence_dropout=0.0``, all entries reset to 1.

    Parameters
    ----------
    confidence
        Input confidence tensor.
    confidence_dropout
        Probability threshold. Higher → fewer resets.
    mask
        Optional explicit boolean mask (for deterministic testing). When
        provided, used in place of ``torch.rand > threshold``.
    generator
        Optional :class:`torch.Generator` controlling the random mask
        (only consulted when ``mask is None``).

    Returns
    -------
    torch.Tensor
        ``confidence`` with masked positions replaced by ``1``.
    """
    if mask is None:
        rand = torch.rand(
            confidence.shape, generator=generator, device=confidence.device,
            dtype=confidence.dtype,
        )
        mask = rand > confidence_dropout
    return torch.where(mask, torch.ones_like(confidence), confidence)


def _branch_loss(
    batch_instances: torch.Tensor,
    historical: torch.Tensor,
    *,
    batch_fraction: float,
    want_dataset_confidence: bool,
    want_batch_correction: bool,
    loss_type: LossType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one confidence branch's loss and the new EMA value.

    Mirrors MATLAB's local ``compute_confidence_branch`` (lines 86-162).
    The branch is one of {Total, Trial, Task}; this function is called
    once per branch with the corresponding batch tensor and history.

    Returns
    -------
    branch_loss : torch.Tensor
        Scalar loss. With batch correction enabled, scaled by ``1/γ``.
    updated_history : torch.Tensor
        Detached scalar — the new EMA value to persist for the next
        minibatch.
    """
    batch_mean = batch_instances.mean()

    if want_dataset_confidence:
        # Subtlety #4: STOP-GRADIENT on history.
        hist_detached = historical.detach()
        # Subtlety #5: BatchFraction-governed cadence (Eq. 7).
        # MATLAB also detaches gamma; here gamma is a Python float so
        # detachment is implicit.
        updated = hist_detached * (1 - batch_fraction) + batch_mean * batch_fraction
    else:
        updated = batch_mean

    target = torch.ones_like(updated)

    if loss_type == "L1":
        raw_loss = F.l1_loss(updated, target)
    elif loss_type == "L2":
        raw_loss = F.mse_loss(updated, target)
    elif loss_type == "L1 & L2":
        raw_loss = F.l1_loss(updated, target) + F.mse_loss(updated, target)
    elif loss_type == "CrossEntropy":
        # MATLAB's crossentropy(Y, T=1) on a scalar reduces to -log(Y).
        # Guard against log(0) at zero confidence.
        raw_loss = -torch.log(updated.clamp(min=1e-12))
    else:  # pragma: no cover — guarded by Literal type
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    if want_dataset_confidence and want_batch_correction:
        # Eq. 10: explicit gradient correction scales by 1/γ.
        branch_loss = raw_loss / batch_fraction
    else:
        branch_loss = raw_loss

    return branch_loss, updated.detach()


def apply_confidence_routing(
    y: torch.Tensor,
    target: torch.Tensor,
    trial_confidence: Optional[torch.Tensor],
    task_confidence: Optional[torch.Tensor],
    *,
    history: ConfidenceHistory,
    batch_fraction: float = 1.0,
    confidence_dropout: float = 0.5,
    want_dataset_confidence: bool = True,
    want_batch_correction: bool = False,
    loss_type: LossType = "L1",
    time_dim: int = 1,
    compute_interpolation: bool = True,
    explicit_trial_dropout_mask: Optional[torch.Tensor] = None,
    explicit_task_dropout_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> ConfidenceLossBreakdown:
    """Apply confidence routing + PD-controller losses to a minibatch.

    Implements the full ``cgg_lossConfidence`` kernel, including all five
    Critical Note #29 subtleties. See the module docstring for the
    mathematical statements; the function comments below identify which
    block implements which subtlety.

    Parameters
    ----------
    y
        Classifier prediction sequence, shape ``(B, T, K)`` by default
        (configurable via ``time_dim``). What goes through the
        interpolation (Eq. 2).
    target
        Classification target tensor, same shape as ``y``.
    trial_confidence
        Per-trial confidence sequence, shape ``(B, T, 1)`` — typically the
        sigmoid output of the Trial confidence head. ``None`` disables
        the Trial branch entirely.
    task_confidence
        Per-class confidence sequence, shape ``(B, T, K)`` — typically the
        Task confidence head. ``None`` disables the Task branch.
    history
        :class:`ConfidenceHistory` carrying the current EMA state. NOT
        modified in place; the new state is returned in the breakdown.
    batch_fraction
        ``γ`` in Eq. 7 — the fraction of the dataset the current batch
        represents. Drives EMA update magnitude.
    confidence_dropout
        Probability threshold for the dropout mask (subtlety #2). Default
        ``0.5`` matches MATLAB.
    want_dataset_confidence
        When ``True`` (default), each branch loss uses the EMA-updated
        confidence. When ``False``, the batch mean is used directly (no
        EMA).
    want_batch_correction
        When ``True``, scales each branch loss by ``1/γ`` (Eq. 10).
    loss_type
        Per-branch loss kernel — ``"L1"`` (default — MATLAB default-ish),
        ``"L2"``, ``"L1 & L2"``, or ``"CrossEntropy"``.
    time_dim
        Axis of ``y`` / ``target`` / confidences treated as time. Default
        ``1`` for ``(B, T, K)``.
    compute_interpolation
        When ``True`` (default), compute and return ``y_interpolated``
        (Eq. 2). When ``False``, return ``y`` unchanged in that slot —
        the caller wants only the loss + Beta + EMA outputs and will
        compute its own prediction interpolation per output dim (or skip
        it). Skipping is mandatory when ``y`` and the confidence tensors
        don't share a last-axis size (e.g. per-dim classification logits
        vs ``(B, T, num_dims)`` task confidence) — otherwise the
        interpolation's broadcasting would silently misalign.
    explicit_trial_dropout_mask, explicit_task_dropout_mask
        Optional boolean masks (same shape as the last-timestep
        confidence). For deterministic testing of subtlety #2 without
        depending on RNG cross-language behavior.
    generator
        Optional :class:`torch.Generator` for reproducible random dropout
        masks (only consulted when the explicit mask is ``None``).

    Returns
    -------
    ConfidenceLossBreakdown
        ``y_interpolated`` (Eq. 2), three branch losses, and the updated
        history (detached). Caller persists the history for the next
        minibatch.
    """
    zero = torch.zeros((), dtype=y.dtype, device=y.device)

    # ── Task branch (subtleties #1, #2) ───────────────────────────────────
    task_undropped: Optional[torch.Tensor] = None
    task_dropped: Optional[torch.Tensor] = None
    if task_confidence is not None:
        task_undropped = _last_timestep(task_confidence, time_dim=time_dim)
        task_dropped = _apply_confidence_dropout(
            task_undropped, confidence_dropout,
            mask=explicit_task_dropout_mask, generator=generator,
        )

    # ── Trial branch + conjunction (Eq. 1) ────────────────────────────────
    trial_undropped: Optional[torch.Tensor] = None
    total_undropped: Optional[torch.Tensor] = task_undropped  # init: Task-only
    total_dropped: Optional[torch.Tensor] = task_dropped
    if trial_confidence is not None:
        trial_undropped = _last_timestep(trial_confidence, time_dim=time_dim)
        trial_dropped = _apply_confidence_dropout(
            trial_undropped, confidence_dropout,
            mask=explicit_trial_dropout_mask, generator=generator,
        )
        if total_undropped is not None:
            total_undropped = total_undropped * trial_undropped  # Eq. 1
            total_dropped = total_dropped * trial_dropped         # Eq. 1
        else:
            total_undropped = trial_undropped
            total_dropped = trial_dropped

    if total_dropped is None:
        # No confidence at all → no interpolation, no losses, no EMA / Beta update.
        return ConfidenceLossBreakdown(
            y_interpolated=y,
            total_loss=zero, trial_loss=zero, task_loss=zero,
            updated_history=history,
        )
    # total_dropped and total_undropped are built in parallel — non-None
    # together. The assertion narrows the type for the checker.
    assert total_undropped is not None

    # ── Beta P-controller (mirrors cgg_getConfidenceLossInformation) ──────
    # MATLAB subtlety: the Beta controller uses the **full-tensor mean** of
    # TotalConfidence (`mean(TotalConfidence, "all")` in
    # cgg_getConfidenceLossInformation.m line 51) — NOT the last-timestep
    # mean that the EMA / branch-loss path uses (which mirrors
    # cgg_lossConfidence.m's cgg_getLastSequenceValue convention). Compute
    # both means: the full one for Beta, the last-timestep ones for EMAs
    # and losses.
    if trial_confidence is not None and task_confidence is not None:
        full_total = trial_confidence * task_confidence
    elif trial_confidence is not None:
        full_total = trial_confidence
    else:
        assert task_confidence is not None
        full_total = task_confidence
    beta_batch_mean = full_total.detach().mean()
    new_beta = _update_confidence_beta(history.beta, beta_batch_mean)

    # ── Interpolation (Eq. 2, subtlety #3) ───────────────────────────────
    # total_dropped is shape (B, K); broadcast a time-axis-1 to apply the
    # per-trial weighting to all timesteps of y. Skipped when the caller
    # explicitly opts out (typical for multi-dim classification where
    # per-dim interpolation happens elsewhere).
    if compute_interpolation:
        td_broadcast = total_dropped.unsqueeze(time_dim)
        y_interpolated = td_broadcast * y + (1 - td_broadcast) * target
    else:
        y_interpolated = y

    # ── Branch losses (subtleties #4, #5; plus Eq. 10 correction) ────────
    total_loss, new_total = _branch_loss(
        total_undropped, history.total,
        batch_fraction=batch_fraction,
        want_dataset_confidence=want_dataset_confidence,
        want_batch_correction=want_batch_correction,
        loss_type=loss_type,
    )
    if trial_undropped is not None:
        trial_loss, new_trial = _branch_loss(
            trial_undropped, history.trial,
            batch_fraction=batch_fraction,
            want_dataset_confidence=want_dataset_confidence,
            want_batch_correction=want_batch_correction,
            loss_type=loss_type,
        )
    else:
        trial_loss, new_trial = zero, history.trial
    if task_undropped is not None:
        task_loss, new_task = _branch_loss(
            task_undropped, history.task,
            batch_fraction=batch_fraction,
            want_dataset_confidence=want_dataset_confidence,
            want_batch_correction=want_batch_correction,
            loss_type=loss_type,
        )
    else:
        task_loss, new_task = zero, history.task

    return ConfidenceLossBreakdown(
        y_interpolated=y_interpolated,
        total_loss=total_loss,
        trial_loss=trial_loss,
        task_loss=task_loss,
        updated_history=ConfidenceHistory(
            total=new_total, trial=new_trial, task=new_task, beta=new_beta,
        ),
    )


__all__ = [
    "ConfidenceHistory",
    "ConfidenceLossBreakdown",
    "apply_confidence_routing",
]
