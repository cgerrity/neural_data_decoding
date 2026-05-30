"""Unit tests for the EMA-normalized multi-objective orchestrator.

Covers the math from ``cgg_getLossInformation.m`` (Critical Notes #6, #30):

* EMA update: ``new_prior = raw*(1-π) + old*π``
* First-iteration behavior: prior = raw (no EMA yet)
* Rescale_Value selection (Classification preferred, Reconstruction
  fallback, then 1.0)
* Per-component pipeline: normalize → rescale → weight
* Stop-gradient on prior (it's a constant scaling factor)
* Assembly: ``Loss_Encoder = Loss_Decoder + Loss_Classifier``
* Backward compat with old ``aggregate_total_loss``
"""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.training.losses.multi_objective import (
    LossPriors,
    NormalizedLossBreakdown,
    aggregate_normalized_losses,
    aggregate_total_loss,
)


# ───────────────────────── First-iteration semantics ─────────────────────────


def test_first_iteration_classification_only_reduces_to_weight() -> None:
    """With only classification active and uninitialized priors,
    the normalized loss equals ``raw * weight``.

    First iteration: Prior_Class = current raw; Rescale_Value = raw;
    normalize → 1.0; rescale → raw; weight → raw * weight.
    """
    raw = torch.tensor(2.0)
    out = aggregate_normalized_losses(
        classification_loss=raw,
        weights={"classification": 3.0},
        priors=LossPriors.initial(),
    )
    assert float(out.total) == pytest.approx(6.0)            # raw * weight
    assert float(out.classification) == pytest.approx(6.0)
    # Updated prior = the current raw (first iteration).
    assert float(out.updated_priors.classification) == pytest.approx(2.0)


def test_first_iteration_classification_plus_reconstruction_uses_class_prior_for_rescale() -> None:
    """Recon's contribution is rescaled by classification's prior.

    First iteration: Rescale = raw_class. Recon: raw_recon / raw_recon
    (normalize) * raw_class (rescale) * weight_recon = raw_class * weight_recon.
    """
    raw_class = torch.tensor(2.0)
    raw_recon = torch.tensor(10.0)
    out = aggregate_normalized_losses(
        classification_loss=raw_class,
        reconstruction_loss=raw_recon,
        weights={"classification": 1.0, "reconstruction": 4.0},
        priors=LossPriors.initial(),
    )
    # Class contribution: 2.0 * 1.0 = 2.0
    # Recon contribution: 2.0 * 4.0 = 8.0 (cross-normalized so recon == class*weight)
    # Total: 10.0
    assert float(out.classification) == pytest.approx(2.0)
    assert float(out.reconstruction) == pytest.approx(8.0)
    assert float(out.total) == pytest.approx(10.0)


def test_rescale_value_falls_back_to_reconstruction_when_no_classification() -> None:
    """With only reconstruction + KL active, rescale uses recon's prior."""
    raw_recon = torch.tensor(5.0)
    raw_kl = torch.tensor(2.0)
    out = aggregate_normalized_losses(
        reconstruction_loss=raw_recon,
        kl_loss=raw_kl,
        weights={"reconstruction": 1.0, "kl": 3.0},
        priors=LossPriors.initial(),
    )
    # First iteration: Rescale = raw_recon = 5.0.
    # Recon: 5/5 * 5 * 1 = 5
    # KL:    2/2 * 5 * 3 = 15
    # Total: 20
    assert float(out.reconstruction) == pytest.approx(5.0)
    assert float(out.kl) == pytest.approx(15.0)
    assert float(out.total) == pytest.approx(20.0)


# ───────────────────────── EMA update math ─────────────────────────


def test_subsequent_iteration_uses_ema_update() -> None:
    """Second iteration: prior = raw*0.1 + old*0.9 (PriorProportion=0.9)."""
    priors = LossPriors(classification=torch.tensor(2.0))
    raw = torch.tensor(4.0)
    out = aggregate_normalized_losses(
        classification_loss=raw,
        weights={"classification": 1.0},
        priors=priors,
        prior_proportion=0.9,
    )
    # new_prior = 4.0 * 0.1 + 2.0 * 0.9 = 0.4 + 1.8 = 2.2
    assert float(out.updated_priors.classification) == pytest.approx(2.2)


def test_update_priors_false_keeps_prior_constant() -> None:
    """When ``update_priors=False``, the stored prior is unchanged."""
    priors = LossPriors(classification=torch.tensor(2.0))
    out = aggregate_normalized_losses(
        classification_loss=torch.tensor(10.0),
        weights={"classification": 1.0},
        priors=priors,
        update_priors=False,
    )
    # The MATLAB semantics: when WantUpdate=False, processLossComponent
    # still computes "new_prior = raw" (line 243) and uses that for
    # normalization. Verify by what comes out for the loss.
    # raw=10, Rescale=stored=2.0, normalize by new_prior=raw=10 → 10/10=1.
    # rescale * 2.0 * weight 1.0 = 2.0.
    assert float(out.total) == pytest.approx(2.0)


# ───────────────────────── Stop-gradient on prior ─────────────────────────


def test_prior_is_detached_after_update() -> None:
    """Updated priors carry no autograd graph."""
    raw = torch.tensor(3.0, requires_grad=True)
    out = aggregate_normalized_losses(
        classification_loss=raw,
        weights={"classification": 1.0},
        priors=LossPriors.initial(),
    )
    assert not out.updated_priors.classification.requires_grad


def test_gradient_flows_through_loss_but_prior_acts_as_constant() -> None:
    """``d(total) / d(raw)`` should be ``weight`` (since prior cancels)."""
    raw = torch.tensor(2.0, requires_grad=True)
    out = aggregate_normalized_losses(
        classification_loss=raw,
        weights={"classification": 5.0},
        priors=LossPriors.initial(),
    )
    out.total.backward()
    # First iter: total = raw / raw_detached * raw_detached * 5 = raw * 5.
    # d/d(raw) = 5.
    assert raw.grad is not None
    assert float(raw.grad) == pytest.approx(5.0)


# ───────────────────────── Weight = 0 deactivates ─────────────────────────


def test_weight_zero_deactivates_component() -> None:
    """A weight of 0 should skip the component entirely."""
    out = aggregate_normalized_losses(
        classification_loss=torch.tensor(2.0),
        reconstruction_loss=torch.tensor(5.0),
        weights={"classification": 1.0, "reconstruction": 0.0},
        priors=LossPriors.initial(),
    )
    assert out.reconstruction is None
    assert out.updated_priors.reconstruction is None  # not touched


# ───────────────────────── Confidence Beta ─────────────────────────


def test_confidence_beta_multiplies_confidence_only() -> None:
    """Beta scales the confidence component but not others."""
    out_b1 = aggregate_normalized_losses(
        classification_loss=torch.tensor(2.0),
        confidence_loss=torch.tensor(2.0),
        weights={"classification": 1.0, "confidence": 1.0},
        priors=LossPriors.initial(),
        confidence_beta=1.0,
    )
    out_b3 = aggregate_normalized_losses(
        classification_loss=torch.tensor(2.0),
        confidence_loss=torch.tensor(2.0),
        weights={"classification": 1.0, "confidence": 1.0},
        priors=LossPriors.initial(),
        confidence_beta=3.0,
    )
    # Confidence with β=1 is 2.0, with β=3 is 6.0; classification stays 2.0.
    assert float(out_b3.confidence) == pytest.approx(3 * float(out_b1.confidence))
    assert float(out_b3.classification) == pytest.approx(float(out_b1.classification))


# ───────────────────────── Assembly (Loss_Decoder / Loss_Classifier) ─────────────────────────


def test_decoder_sums_recon_kl_offset_scale() -> None:
    """Loss_Decoder = Recon + KL + OffsetScale."""
    out = aggregate_normalized_losses(
        reconstruction_loss=torch.tensor(1.0),
        kl_loss=torch.tensor(2.0),
        offset_and_scale_loss=torch.tensor(3.0),
        weights={"reconstruction": 1.0, "kl": 1.0, "offset_and_scale": 1.0},
        priors=LossPriors.initial(),
    )
    # With no classification, rescale falls back to recon's prior = 1.
    # Recon: 1/1 * 1 * 1 = 1; KL: 2/2 * 1 * 1 = 1; OffsetScale: 3/3 * 1 * 1 = 1.
    # Decoder = 1 + 1 + 1 = 3.
    assert float(out.decoder) == pytest.approx(3.0)
    assert out.classifier is None  # no classification or confidence
    assert float(out.total) == pytest.approx(3.0)


def test_classifier_sums_classification_confidence() -> None:
    """Loss_Classifier = Classification + Confidence."""
    out = aggregate_normalized_losses(
        classification_loss=torch.tensor(2.0),
        confidence_loss=torch.tensor(4.0),
        weights={"classification": 1.0, "confidence": 1.0},
        priors=LossPriors.initial(),
    )
    # Class: 2/2 * 2 * 1 = 2; Conf: 4/4 * 2 * 1 = 2. Classifier = 4.
    assert float(out.classifier) == pytest.approx(4.0)
    assert out.decoder is None
    assert float(out.total) == pytest.approx(4.0)


def test_total_is_decoder_plus_classifier() -> None:
    """Loss_Encoder = Loss_Decoder + Loss_Classifier; the gradient root."""
    out = aggregate_normalized_losses(
        reconstruction_loss=torch.tensor(1.0),
        classification_loss=torch.tensor(2.0),
        weights={"reconstruction": 1.0, "classification": 1.0},
        priors=LossPriors.initial(),
    )
    # First iter: rescale = class prior = 2.0.
    # Recon: 1/1 * 2 * 1 = 2; Class: 2/2 * 2 * 1 = 2.
    # Decoder = 2; Classifier = 2; total = 4.
    assert float(out.decoder) == pytest.approx(2.0)
    assert float(out.classifier) == pytest.approx(2.0)
    assert float(out.total) == pytest.approx(4.0)


# ───────────────────────── Backward compat with simple orchestrator ─────────────────────────


def test_aggregate_total_loss_still_works_for_classification_only() -> None:
    """Milestone A's simple orchestrator path is unchanged."""
    loss, _ = aggregate_total_loss(
        classification_loss=torch.tensor(1.5),
        weights={"classification": 10.0},
    )
    assert float(loss) == pytest.approx(15.0)
