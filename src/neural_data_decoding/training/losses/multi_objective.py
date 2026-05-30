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


# ───────────────────────── EMA prior normalization (Milestone C+) ─────────────────────────


@dataclass(slots=True)
class LossPriors:
    """Per-component EMA priors for cross-component loss normalization.

    Ports MATLAB's ``LossInformation.Prior_Loss_*`` fields (see
    ``cgg_getLossInformation.m`` lines 90-103). Each field holds a
    **detached** scalar tensor — the running mean magnitude of that loss
    component over training, used to normalize new losses so all
    components share the same effective scale (Critical Notes #6, #30).

    A field of ``None`` indicates the corresponding component hasn't been
    seen yet; the orchestrator falls back to the current batch's loss as
    the initial prior (matching MATLAB's first-iteration behavior).

    Attributes
    ----------
    reconstruction, kl, classification, offset_and_scale, confidence
        Per-component EMA priors. All ``None`` initially.

    Examples
    --------
    >>> priors = LossPriors.initial()
    >>> priors.classification is None
    True
    """

    reconstruction: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None
    classification: Optional[torch.Tensor] = None
    offset_and_scale: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None

    @classmethod
    def initial(cls) -> "LossPriors":
        """Build an uninitialized prior state. First iteration self-bootstraps."""
        return cls()


@dataclass(slots=True)
class NormalizedLossBreakdown:
    """Output of :func:`aggregate_normalized_losses`.

    Like :class:`LossBreakdown` but tracks the **normalized + rescaled +
    weighted** values used in the gradient sum, plus the updated priors
    for the caller to persist across iterations.

    Attributes
    ----------
    total
        The single gradient root (``Loss_Encoder`` in MATLAB). Call
        ``total.backward()`` on this.
    decoder, classifier
        Intermediate sums (``Loss_Decoder`` = recon+KL+offset/scale;
        ``Loss_Classifier`` = classification+confidence). For telemetry —
        **do not** call ``.backward()`` on these (would double-count).
    reconstruction, kl, classification, confidence, offset_and_scale
        Per-component normalized+weighted contributions to the total. All
        ``None`` when the corresponding input wasn't provided.
    updated_priors
        Fresh :class:`LossPriors` with this iteration's EMA-updated values.
        Caller persists this for the next iteration.
    rescale_value
        The reference magnitude (Classification's prior, or fallback) all
        components were rescaled to. Detached scalar; saved for logging.
    """

    total: torch.Tensor
    decoder: Optional[torch.Tensor]
    classifier: Optional[torch.Tensor]
    reconstruction: Optional[torch.Tensor]
    kl: Optional[torch.Tensor]
    classification: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor]
    offset_and_scale: Optional[torch.Tensor]
    updated_priors: LossPriors
    rescale_value: torch.Tensor


def _determine_rescale_value(
    priors: LossPriors,
    classification_loss: Optional[torch.Tensor],
    reconstruction_loss: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute the cross-component normalization reference (``Rescale_Value``).

    Mirrors ``cgg_getLossInformation.m`` lines 117-134:

    * If classification is active and has a prior (or this is the first
      iteration), use Classification's prior (or current batch loss).
    * Otherwise fall back to Reconstruction's prior / current loss.
    * Otherwise ``1.0`` (no rescaling).

    The returned tensor is **always detached** — it acts as a constant
    scaling factor in the gradient.
    """
    # Effective priors: stored EMA, or current batch's loss as fallback.
    eff_class = priors.classification
    if eff_class is None and classification_loss is not None:
        eff_class = classification_loss.detach()
    eff_recon = priors.reconstruction
    if eff_recon is None and reconstruction_loss is not None:
        eff_recon = reconstruction_loss.detach()

    if classification_loss is not None and eff_class is not None:
        return eff_class.detach()
    if reconstruction_loss is not None and eff_recon is not None:
        return eff_recon.detach()
    return torch.ones((), dtype=dtype, device=device)


def _process_component(
    loss: Optional[torch.Tensor],
    current_prior: Optional[torch.Tensor],
    *,
    rescale_value: torch.Tensor,
    weight: float,
    update_prior: bool,
    prior_proportion: float,
    beta: Optional[float] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Apply EMA-update → normalize → rescale → weight to one component.

    Ports ``cgg_processLossComponent`` (lines 216-274 of the MATLAB source).
    Returns ``(processed_loss, updated_prior)``. The processed loss
    preserves gradient flow w.r.t. its input; the updated prior is
    detached (a constant for future iterations).

    Returns ``(None, current_prior)`` when ``loss is None`` (component
    inactive) or ``weight == 0``.
    """
    if loss is None or weight == 0:
        return None, current_prior

    raw_detached = loss.detach()

    # 1. Update prior via EMA (matches MATLAB line 241).
    if update_prior and current_prior is not None:
        new_prior = (
            raw_detached * (1.0 - prior_proportion) + current_prior * prior_proportion
        )
    else:
        # First iteration OR caller asked to skip update: prior = current raw.
        new_prior = raw_detached

    loss_working = loss

    # 2. Normalize by the (updated) prior — avoid /0.
    if (new_prior != 0).all():
        loss_working = loss_working / new_prior

    # 3. Rescale to classification's prior magnitude.
    loss_working = loss_working * rescale_value

    # 4. Apply per-component weight.
    loss_working = loss_working * weight

    # 5. Optional Beta (used for the confidence component in MATLAB).
    if beta is not None:
        loss_working = loss_working * beta

    return loss_working, new_prior.detach()


def aggregate_normalized_losses(
    *,
    reconstruction_loss: Optional[torch.Tensor] = None,
    kl_loss: Optional[torch.Tensor] = None,
    classification_loss: Optional[torch.Tensor] = None,
    confidence_loss: Optional[torch.Tensor] = None,
    offset_and_scale_loss: Optional[torch.Tensor] = None,
    weights: Mapping[str, float],
    priors: LossPriors,
    prior_proportion: float = 0.9,
    update_priors: bool = True,
    confidence_beta: float = 1.0,
) -> NormalizedLossBreakdown:
    """EMA-normalized multi-objective loss aggregator (port of cgg_getLossInformation).

    Each active loss component is processed through the
    normalize-rescale-weight pipeline (Critical Notes #6, #30): divide by
    its own EMA prior (so all components have unit magnitude), rescale by
    the reference component's prior (Classification's by default, hence
    all components end up at classification's magnitude), then apply the
    configured weight. The five processed components are then summed into
    ``Loss_Decoder`` (recon + KL + offset/scale), ``Loss_Classifier``
    (classification + confidence), and finally ``Loss_Encoder`` (decoder +
    classifier) — the single gradient root per Critical Note #28.

    Parameters
    ----------
    reconstruction_loss, kl_loss, classification_loss, confidence_loss, offset_and_scale_loss
        Per-component scalar losses. Pass ``None`` for components that
        aren't active.
    weights
        Per-component weight dict (keys: ``"reconstruction"``, ``"kl"``,
        ``"classification"``, ``"confidence"``, ``"offset_and_scale"``).
        Missing keys default to ``1.0``; weight of ``0`` deactivates the
        component (no contribution, no prior update).
    priors
        :class:`LossPriors` carrying the running EMA state. NOT modified
        in place; the new state is returned in the breakdown.
    prior_proportion
        EMA smoothing factor (``π`` in MATLAB). New prior is
        ``raw*(1-π) + old*π``. Default ``0.9`` matches the Optimal config.
    update_priors
        When ``True`` (default), priors get EMA-updated by this batch.
        When ``False``, priors are read but not modified — matches MATLAB's
        ``WantUpdateLossPrior`` and is driven by ``RescaleLossEpoch`` /
        cadence in the training loop.
    confidence_beta
        Additional scalar multiplier for the confidence component
        (matches MATLAB's ``LossInformation.Confidence_Beta``). The
        dynamic Beta-tracker is not yet ported; for now pass a constant
        (typically ``1.0``).

    Returns
    -------
    NormalizedLossBreakdown
        ``total`` is the gradient root; per-component fields, intermediate
        sums, and ``updated_priors`` are also returned.
    """

    def w(key: str) -> float:
        return float(weights.get(key, 1.0))

    # Reference dtype/device from any active loss.
    ref_tensor: Optional[torch.Tensor] = None
    for c in (
        classification_loss,
        reconstruction_loss,
        kl_loss,
        confidence_loss,
        offset_and_scale_loss,
    ):
        if c is not None:
            ref_tensor = c
            break

    if ref_tensor is None:
        # No components active — return a zero scalar.
        zero = torch.zeros(())
        return NormalizedLossBreakdown(
            total=zero, decoder=None, classifier=None,
            reconstruction=None, kl=None, classification=None,
            confidence=None, offset_and_scale=None,
            updated_priors=priors,
            rescale_value=torch.ones((), dtype=zero.dtype, device=zero.device),
        )

    # Determine the cross-component normalization reference BEFORE any
    # prior gets updated (Critical Note #30: Rescale_Value uses the
    # pre-update priors).
    rescale_value = _determine_rescale_value(
        priors, classification_loss, reconstruction_loss,
        dtype=ref_tensor.dtype, device=ref_tensor.device,
    )

    # Per-component processing.
    recon_out, new_recon_prior = _process_component(
        reconstruction_loss, priors.reconstruction,
        rescale_value=rescale_value,
        weight=w("reconstruction"),
        update_prior=update_priors,
        prior_proportion=prior_proportion,
    )
    kl_out, new_kl_prior = _process_component(
        kl_loss, priors.kl,
        rescale_value=rescale_value,
        weight=w("kl"),
        update_prior=update_priors,
        prior_proportion=prior_proportion,
    )
    class_out, new_class_prior = _process_component(
        classification_loss, priors.classification,
        rescale_value=rescale_value,
        weight=w("classification"),
        update_prior=update_priors,
        prior_proportion=prior_proportion,
    )
    offset_out, new_offset_prior = _process_component(
        offset_and_scale_loss, priors.offset_and_scale,
        rescale_value=rescale_value,
        weight=w("offset_and_scale"),
        update_prior=update_priors,
        prior_proportion=prior_proportion,
    )
    conf_out, new_conf_prior = _process_component(
        confidence_loss, priors.confidence,
        rescale_value=rescale_value,
        weight=w("confidence"),
        update_prior=update_priors,
        prior_proportion=prior_proportion,
        beta=confidence_beta,
    )

    # Assembly: Loss_Decoder = recon + KL + offset/scale.
    decoder_parts = [p for p in (recon_out, kl_out, offset_out) if p is not None]
    loss_decoder: Optional[torch.Tensor] = (
        torch.stack(decoder_parts).sum() if decoder_parts else None
    )

    # Loss_Classifier = classification + confidence.
    classifier_parts = [p for p in (class_out, conf_out) if p is not None]
    loss_classifier: Optional[torch.Tensor] = (
        torch.stack(classifier_parts).sum() if classifier_parts else None
    )

    # Loss_Encoder = decoder + classifier (the gradient root).
    encoder_parts = [p for p in (loss_decoder, loss_classifier) if p is not None]
    if encoder_parts:
        loss_encoder = torch.stack(encoder_parts).sum()
    else:
        loss_encoder = torch.zeros((), dtype=ref_tensor.dtype, device=ref_tensor.device)

    return NormalizedLossBreakdown(
        total=loss_encoder,
        decoder=loss_decoder,
        classifier=loss_classifier,
        reconstruction=recon_out,
        kl=kl_out,
        classification=class_out,
        confidence=conf_out,
        offset_and_scale=offset_out,
        updated_priors=LossPriors(
            reconstruction=new_recon_prior,
            kl=new_kl_prior,
            classification=new_class_prior,
            offset_and_scale=new_offset_prior,
            confidence=new_conf_prior,
        ),
        rescale_value=rescale_value,
    )


__all__ = [
    "LossBreakdown",
    "LossPriors",
    "NormalizedLossBreakdown",
    "aggregate_normalized_losses",
    "aggregate_total_loss",
]
