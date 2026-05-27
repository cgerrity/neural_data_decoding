"""Multi-objective loss orchestrator.

Ports the **gradient root** computation from ``cgg_lossComponents.m`` plus
``cgg_getLossInformation.m`` — i.e., the path that produces a single
scalar ``Loss_Encoder`` that all three subnetworks backprop from
(Critical Note #28). The full MATLAB orchestrator combines five
components (Reconstruction + KL + Classification + Confidence +
OffsetAndScale) with EMA prior normalization (Critical Notes #6 + #30);
this Milestone A version only handles **Classification**, because that's
the only component active for ``ModelName='Logistic Regression'``. The
hooks for the other components are in place but stubbed out.

The orchestrator is intentionally side-effect-free: it takes pre-computed
component values and returns the combined scalar + a dict of per-component
values for telemetry. EMA prior normalization is *deferred to a future
milestone* — for Milestone A every component's contribution is just its
raw value scaled by its configured weight.

Examples
--------
>>> import torch
>>> loss, info = aggregate_total_loss(
...     classification_loss=torch.tensor(1.5),
...     weights={"classification": 10.0},
... )
>>> float(loss)
15.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(slots=True)
class LossBreakdown:
    """Per-component loss values for telemetry / logging.

    Each field defaults to ``None`` when the corresponding component
    isn't active. The ``total`` field always holds the scalar that the
    training loop backprops from.

    Attributes
    ----------
    total
        The single scalar that the training loop backprops from.
    classification
        Raw (un-weighted) classification cross-entropy. Always present.
    reconstruction
        Reconstruction loss (ELBO). Milestone C+.
    kl
        KL divergence. Milestone C+.
    confidence
        Confidence-budget regularizer. Milestone C+.
    offset_and_scale
        Augmentation loss. Milestone CC+.
    weights
        Snapshot of the per-component weights used to assemble ``total``.
    """

    total: torch.Tensor
    classification: torch.Tensor
    reconstruction: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    offset_and_scale: Optional[torch.Tensor] = None
    weights: dict[str, float] = field(default_factory=dict)


def aggregate_total_loss(
    *,
    classification_loss: torch.Tensor,
    reconstruction_loss: Optional[torch.Tensor] = None,
    kl_loss: Optional[torch.Tensor] = None,
    confidence_loss: Optional[torch.Tensor] = None,
    offset_and_scale_loss: Optional[torch.Tensor] = None,
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, LossBreakdown]:
    """Sum the active loss components into the single gradient-root scalar.

    For Milestone A only ``classification_loss`` is non-``None`` in
    practice. The other arguments are accepted (and tested) so the
    interface is stable across milestones — Milestone C+ will start
    passing values for the additional components.

    Parameters
    ----------
    classification_loss
        Scalar classification loss (e.g., from
        :func:`~neural_data_decoding.training.losses.classification.multi_head_cross_entropy`).
    reconstruction_loss, kl_loss, confidence_loss, offset_and_scale_loss
        Optional scalar components. Pass ``None`` (the default) when the
        component isn't active.
    weights
        Per-component weight, keyed by component name (``"classification"``,
        ``"reconstruction"``, ``"kl"``, ``"confidence"``, ``"offset_and_scale"``).
        Missing keys default to ``1.0``. The actual MATLAB pipeline pulls
        these from ``cfg_Encoder.Weight*`` at run time; the Python loop
        builds the dict from the resolved config.

    Returns
    -------
    total : torch.Tensor
        The 0-D scalar to call ``.backward()`` on.
    breakdown : LossBreakdown
        Per-component values + weight snapshot, suitable for telemetry.

    Notes
    -----
    EMA prior normalization (Critical Note #30) is *not* applied here in
    Milestone A. Milestone C will add an EMAPriorNormalizer stateful
    object that re-scales each component before summation; that change
    will be backward-compatible — callers won't need to pass it for the
    Milestone A target.
    """

    def w(key: str) -> float:
        return float(weights.get(key, 1.0))

    breakdown = LossBreakdown(
        total=torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device),
        classification=classification_loss,
        reconstruction=reconstruction_loss,
        kl=kl_loss,
        confidence=confidence_loss,
        offset_and_scale=offset_and_scale_loss,
        weights={
            "classification": w("classification"),
            "reconstruction": w("reconstruction"),
            "kl": w("kl"),
            "confidence": w("confidence"),
            "offset_and_scale": w("offset_and_scale"),
        },
    )

    total = breakdown.total + w("classification") * classification_loss

    if reconstruction_loss is not None:
        total = total + w("reconstruction") * reconstruction_loss
    if kl_loss is not None:
        total = total + w("kl") * kl_loss
    if confidence_loss is not None:
        total = total + w("confidence") * confidence_loss
    if offset_and_scale_loss is not None:
        total = total + w("offset_and_scale") * offset_and_scale_loss

    breakdown.total = total
    return total, breakdown


__all__ = ["LossBreakdown", "aggregate_total_loss"]
