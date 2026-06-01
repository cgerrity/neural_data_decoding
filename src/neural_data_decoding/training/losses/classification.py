"""Multi-head weighted classification loss.

Ports the supervised branch of ``cgg_calcClassificationLoss.m``: per-
output-dimension cross-entropy with optional inverse-frequency class
weights. For Milestone A this is the *only* loss component active —
later milestones (B/C) layer ELBO, confidence, and EMA prior
normalization on top via the :mod:`multi_objective` orchestrator.

The classifier head emits **logits** (see
:class:`neural_data_decoding.models.classifier.MultiHeadClassifier`), so
this kernel uses ``F.cross_entropy`` which fuses log-softmax + NLL for
numerical stability.

When the classifier head's output has a time dimension, the loss is
averaged across time AND batch by default — matching MATLAB's
``crossentropy(Y, T)`` reduction when ``T`` is broadcast across time
samples.

Examples
--------
>>> import torch
>>> logits = [torch.randn(4, 5, 3), torch.randn(4, 5, 2)]  # (B, T, K) per dim
>>> targets = torch.tensor([[0, 1], [1, 0], [2, 1], [0, 0]])  # (B, num_dim)
>>> loss = multi_head_cross_entropy(logits, targets)
>>> loss.ndim
0
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def multi_head_cross_entropy(
    logits_per_dim: Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    class_weights_per_dim: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute summed per-dimension weighted cross-entropy.

    Each output dimension contributes its own cross-entropy term; the
    returned scalar is the **sum** across dimensions. Per-dimension class
    weights, if supplied, scale the per-class contribution within that
    dimension's term (matches MATLAB's ``crossentropy(Y, T, Weights)``).

    The loss kernel handles two logit shapes:

    * ``(batch, num_classes)`` — typical for non-sequential classification.
    * ``(batch, time, num_classes)`` — sequence outputs; the same target
      applies across all time steps. Reduction is over time as well as
      batch.

    Parameters
    ----------
    logits_per_dim
        Per-output-dimension logits. Each tensor has shape
        ``(batch, *, num_classes_d)`` where ``num_classes_d`` may differ
        across dimensions.
    targets
        Integer class labels, shape ``(batch, num_dimensions)``. Each
        column ``d`` indexes into ``logits_per_dim[d]``'s last axis.
    class_weights_per_dim
        Optional per-dimension class weight vectors. If provided, must
        be the same length as ``logits_per_dim`` and each element must
        be a 1-D tensor of length ``num_classes_d``. Pass ``None`` to
        disable class weighting (the MATLAB default when
        ``WeightedLoss=''``).

    Returns
    -------
    torch.Tensor
        Scalar (0-D) loss = sum across dimensions of per-dim mean
        cross-entropy.

    Raises
    ------
    ValueError
        If the input shapes are inconsistent (wrong number of dimensions,
        mismatched batch sizes, etc.).
    """
    if not logits_per_dim:
        raise ValueError("logits_per_dim must be a non-empty sequence.")

    num_dimensions = len(logits_per_dim)
    if targets.ndim != 2 or targets.shape[1] != num_dimensions:
        raise ValueError(
            f"targets must have shape (batch, {num_dimensions}); "
            f"got shape {tuple(targets.shape)}."
        )

    if class_weights_per_dim is not None and len(class_weights_per_dim) != num_dimensions:
        raise ValueError(
            f"class_weights_per_dim must have {num_dimensions} entries; "
            f"got {len(class_weights_per_dim)}."
        )

    total = torch.zeros((), dtype=torch.float32, device=logits_per_dim[0].device)

    for dim_idx, logits in enumerate(logits_per_dim):
        dim_target = targets[:, dim_idx]
        weight = (
            class_weights_per_dim[dim_idx]
            if class_weights_per_dim is not None
            else None
        )

        if logits.ndim == 2:
            # (B, K) — simple case.
            loss = F.cross_entropy(logits, dim_target, weight=weight)
        elif logits.ndim == 3:
            # (B, T, K) — broadcast the target across time then flatten.
            batch, time, num_classes = logits.shape
            flat_logits = logits.reshape(batch * time, num_classes)
            broadcast_target = dim_target.unsqueeze(1).expand(batch, time).reshape(-1)
            loss = F.cross_entropy(flat_logits, broadcast_target, weight=weight)
        else:
            raise ValueError(
                f"Each logits tensor must have 2 or 3 dimensions; got "
                f"shape {tuple(logits.shape)} for dimension {dim_idx}."
            )

        total = total + loss

    return total


def inverse_frequency_class_weights(
    targets: torch.Tensor, num_classes_per_dim: Sequence[int]
) -> list[torch.Tensor]:
    """Build per-dimension inverse-frequency class weights from training labels.

    Ports the ``WeightedLoss='Inverse'`` branch of
    ``cgg_getWeightsForLoss.m``: classes that appear rarely in the
    training set get larger weights so the classifier doesn't collapse
    onto majority predictions.

    Implementation detail: the weight for class ``c`` in dimension ``d``
    is ``N / (K_d * count_c)`` where ``N`` is the total trial count,
    ``K_d`` is the class count for that dimension, and ``count_c`` is the
    number of training trials labelled ``c`` along dimension ``d``. This
    normalizes so that the average weight is 1.0 (matches MATLAB's
    convention).

    Parameters
    ----------
    targets
        Per-trial labels, shape ``(num_trials, num_dimensions)``. Integer
        type.
    num_classes_per_dim
        Per-dimension class count. Determines the length of each weight
        vector and lets us handle classes with zero training examples
        (their weight is set to 1.0 — the contribution is zero anyway,
        but this avoids divide-by-zero NaNs).

    Returns
    -------
    list of torch.Tensor
        One 1-D tensor per dimension, length ``num_classes_per_dim[d]``.

    Raises
    ------
    ValueError
        If ``targets`` shape doesn't match ``num_classes_per_dim`` length.
    """
    if targets.ndim != 2:
        raise ValueError(
            f"targets must be 2-D (num_trials, num_dimensions); "
            f"got shape {tuple(targets.shape)}."
        )
    if targets.shape[1] != len(num_classes_per_dim):
        raise ValueError(
            f"targets.shape[1] ({targets.shape[1]}) does not match "
            f"len(num_classes_per_dim) ({len(num_classes_per_dim)})."
        )

    num_trials = float(targets.shape[0])
    if num_trials == 0:
        raise ValueError("Cannot compute weights from empty targets.")

    weights: list[torch.Tensor] = []
    for dim_idx, num_classes in enumerate(num_classes_per_dim):
        counts = torch.bincount(targets[:, dim_idx], minlength=num_classes).float()
        # Avoid /0 for classes with no training examples.
        safe = counts.clamp(min=1.0)
        w = num_trials / (num_classes * safe)
        # Classes with zero examples get weight 1 (no influence anyway,
        # since they never appear in the loss term).
        w = torch.where(counts > 0, w, torch.ones_like(w))
        weights.append(w)

    return weights


def mil_multi_head_cross_entropy(
    logits_per_dim: Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    class_weights_per_dim: Sequence[torch.Tensor] | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Multiple-Instance-Learning cross-entropy on per-dim marginal probabilities.

    Mirrors the MATLAB MIL path: ``cgg_softmaxLayer('SCT')`` applies a
    joint softmax across S/C/T axes (here just T/K — no spatial axis in
    the synthetic data), then ``cgg_getPredictionFromClassifierProbabilities.m``
    line 163 aggregates with ``sum(Y, [S, T])`` to get a per-trial
    marginal over the class axis. CE is then computed on the marginal.

    For each output dim ``d`` with logits shape ``(B, T, K_d)``:

    1. Joint softmax over ``(T, K_d)``:
       ``probs = softmax(logits.reshape(B, T*K_d))``, then reshape back.
       Each ``(B,)`` slice sums to 1 over its ``T*K_d`` cells.
    2. Marginal over K via ``probs.sum(dim=1)`` → ``(B, K_d)``. Still sums
       to 1 over ``K_d``.
    3. ``-log(marginal[target_d])`` per trial, mean over B, sum across dims.

    Parameters
    ----------
    logits_per_dim
        Per-dim logits, each ``(B, T, K_d)``. T must be present (the
        whole point of MIL is bag aggregation over time).
    targets
        Integer class labels ``(B, num_dimensions)``.
    class_weights_per_dim
        Optional per-dim class weight vectors (same contract as
        :func:`multi_head_cross_entropy`).
    eps
        Lower clamp on marginal probability before ``log``.

    Returns
    -------
    torch.Tensor
        Scalar (0-D) summed across dimensions of per-dim mean MIL CE.
    """
    if not logits_per_dim:
        raise ValueError("logits_per_dim must be a non-empty sequence.")

    num_dimensions = len(logits_per_dim)
    if targets.ndim != 2 or targets.shape[1] != num_dimensions:
        raise ValueError(
            f"targets must have shape (batch, {num_dimensions}); "
            f"got shape {tuple(targets.shape)}."
        )
    if class_weights_per_dim is not None and len(class_weights_per_dim) != num_dimensions:
        raise ValueError(
            f"class_weights_per_dim must have {num_dimensions} entries; "
            f"got {len(class_weights_per_dim)}."
        )

    total = torch.zeros((), dtype=logits_per_dim[0].dtype, device=logits_per_dim[0].device)

    for dim_idx, logits in enumerate(logits_per_dim):
        if logits.ndim != 3:
            raise ValueError(
                f"MIL CE requires (batch, time, num_classes) logits; "
                f"got shape {tuple(logits.shape)} for dim {dim_idx}."
            )
        dim_target = targets[:, dim_idx]                          # (B,)
        marginal = _mil_marginal_probs(logits)                    # (B, K_d)

        # NLL on marginal: -log(marginal[target_class]).
        p_target = marginal.gather(-1, dim_target.unsqueeze(-1)).squeeze(-1)
        per_trial_loss = -torch.log(p_target.clamp(min=eps))      # (B,)

        if class_weights_per_dim is not None:
            weights = class_weights_per_dim[dim_idx]              # (K_d,)
            per_trial_loss = per_trial_loss * weights.gather(0, dim_target)

        total = total + per_trial_loss.mean()

    return total


def _mil_marginal_probs(logits: torch.Tensor) -> torch.Tensor:
    """Joint softmax over ``(T, K)`` then sum over ``T`` → ``(B, K)`` marginal.

    Mirrors MATLAB's ``cgg_softmaxLayer('SCT').predict`` followed by
    ``sum(Y, [S, T])`` from ``cgg_getPredictionFromClassifierProbabilities.m``.
    No spatial ``S`` axis in this codebase yet; the function handles
    only the ``(B, T, K)`` case. When real data introduces ``S``, extend
    by reshaping appropriately.
    """
    batch, time, k = logits.shape
    flat = logits.reshape(batch, time * k)
    joint_probs = torch.softmax(flat, dim=-1).reshape(batch, time, k)
    return joint_probs.sum(dim=1)


def interpolated_multi_head_cross_entropy(
    logits_per_dim: Sequence[torch.Tensor],
    targets: torch.Tensor,
    total_dropped: torch.Tensor,
    *,
    class_weights_per_dim: Sequence[torch.Tensor] | None = None,
    eps: float = 1e-12,
    mil: bool = False,
) -> torch.Tensor:
    """Per-dimension cross-entropy on confidence-interpolated predictions (Eq. 2).

    Mirrors the MATLAB chain ``cgg_lossClassification`` → ``cgg_lossConfidence``
    (line 75 ``Y = c .* Y + (1 - c) .* T``) → ``crossentropy(Y_interpolated, T)``.

    For one-hot targets, the interpolation collapses to a single scalar per
    timestep per trial:

    .. math::

        Y'[k^*] = c \\cdot p_{target} + (1 - c)
        \\quad\\text{(where }k^*\\text{ is the true class)}

    and the cross-entropy reduces to :math:`-\\log(Y'[k^*])`. This avoids
    materializing the full interpolated probability tensor; we gather the
    predicted target probability and apply the closed-form.

    Parameters
    ----------
    logits_per_dim
        Per-output-dimension logits, each ``(batch, time, num_classes_d)``.
    targets
        Integer class labels, shape ``(batch, num_dimensions)``.
    total_dropped
        Per-trial TotalConfidence after dropout, shape ``(B, K_conf)``.
        ``K_conf`` is ``1`` (Trial-only — same confidence shared across
        all output dims, broadcast) or ``num_dimensions`` (per-dim Task or
        Trial × Task conjunction). Other shapes raise.
    class_weights_per_dim
        Optional per-dim inverse-frequency class weight vectors (same
        contract as :func:`multi_head_cross_entropy`).
    eps
        Lower clamp on the interpolated probability before ``log`` to
        avoid ``-inf`` at zero confidence on a wrong prediction.
    mil
        When ``True``, apply the MIL joint softmax + temporal aggregation
        before the interpolation (mirrors :func:`mil_multi_head_cross_entropy`'s
        first two steps). The interpolation closed-form is then applied
        to the per-trial marginal probability instead of the per-
        timestep probability. ``False`` (default): per-timestep
        prediction-to-truth interpolation as in C #7b.

    Returns
    -------
    torch.Tensor
        Scalar (0-D) summed across dimensions of per-dim mean interpolated
        cross-entropy. Same reduction convention as
        :func:`multi_head_cross_entropy`.

    Raises
    ------
    ValueError
        On shape inconsistencies between ``logits_per_dim``, ``targets``,
        and ``total_dropped``.
    """
    if not logits_per_dim:
        raise ValueError("logits_per_dim must be a non-empty sequence.")

    num_dimensions = len(logits_per_dim)
    if targets.ndim != 2 or targets.shape[1] != num_dimensions:
        raise ValueError(
            f"targets must have shape (batch, {num_dimensions}); "
            f"got shape {tuple(targets.shape)}."
        )
    if total_dropped.ndim != 2:
        raise ValueError(
            f"total_dropped must be 2-D (batch, K_conf); "
            f"got shape {tuple(total_dropped.shape)}."
        )
    k_conf = total_dropped.shape[-1]
    if k_conf != 1 and k_conf != num_dimensions:
        raise ValueError(
            f"total_dropped.shape[-1] ({k_conf}) must be 1 (shared "
            f"across dims) or num_dimensions ({num_dimensions}) — got "
            f"neither."
        )
    if class_weights_per_dim is not None and len(class_weights_per_dim) != num_dimensions:
        raise ValueError(
            f"class_weights_per_dim must have {num_dimensions} entries; "
            f"got {len(class_weights_per_dim)}."
        )

    total = torch.zeros((), dtype=logits_per_dim[0].dtype, device=logits_per_dim[0].device)

    for dim_idx, logits in enumerate(logits_per_dim):
        if logits.ndim != 3:
            raise ValueError(
                f"interpolated CE requires (batch, time, num_classes) "
                f"logits; got shape {tuple(logits.shape)} for dim {dim_idx}."
            )
        batch, time, _ = logits.shape
        dim_target = targets[:, dim_idx]                          # (B,)

        # Slice per-dim confidence — broadcast across dims when K_conf == 1.
        c_slice = total_dropped[:, 0] if k_conf == 1 else total_dropped[:, dim_idx]

        if mil:
            # MIL: joint softmax over (T, K) → sum over T → marginal (B, K).
            # Then interpolate the SCALAR per-trial marginal of the target
            # class. No time axis to preserve here.
            marginal = _mil_marginal_probs(logits)                # (B, K_d)
            p_target = marginal.gather(-1, dim_target.unsqueeze(-1)).squeeze(-1)  # (B,)
            interpolated = c_slice * p_target + (1.0 - c_slice)   # (B,)
            per_trial_loss = -torch.log(interpolated.clamp(min=eps))  # (B,)
            if class_weights_per_dim is not None:
                weights = class_weights_per_dim[dim_idx]
                per_trial_loss = per_trial_loss * weights.gather(0, dim_target)
        else:
            # Standard per-timestep interpolation (C #7b path).
            c = c_slice.unsqueeze(-1)                              # (B, 1)
            probs = torch.softmax(logits, dim=-1)                  # (B, T, K_d)
            target_broadcast = dim_target.view(batch, 1, 1).expand(batch, time, 1)
            p_target = probs.gather(-1, target_broadcast).squeeze(-1)  # (B, T)
            interpolated = c * p_target + (1.0 - c)
            per_trial_loss = -torch.log(interpolated.clamp(min=eps))   # (B, T)
            if class_weights_per_dim is not None:
                weights = class_weights_per_dim[dim_idx]
                per_trial_weight = weights.gather(0, dim_target)
                per_trial_loss = per_trial_loss * per_trial_weight.unsqueeze(-1)

        total = total + per_trial_loss.mean()

    return total


__all__ = [
    "interpolated_multi_head_cross_entropy",
    "inverse_frequency_class_weights",
    "mil_multi_head_cross_entropy",
    "multi_head_cross_entropy",
]
