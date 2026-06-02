"""Tests for :mod:`neural_data_decoding.training.losses`."""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.training.losses.classification import (
    inverse_frequency_class_weights,
    multi_head_cross_entropy,
)
from neural_data_decoding.training.losses.multi_objective import (
    aggregate_total_loss,
)


# ───────────────────────── multi_head_cross_entropy ─────────────────────────


def test_multi_head_cross_entropy_returns_scalar() -> None:
    """With (B, K) logits the output is a 0-D tensor (sum across dims)."""
    logits = [torch.randn(4, 3), torch.randn(4, 2)]
    targets = torch.tensor([[0, 1], [1, 0], [2, 1], [0, 0]])
    loss = multi_head_cross_entropy(logits, targets)
    assert loss.shape == ()
    assert loss.dtype == torch.float32


def test_multi_head_cross_entropy_supports_sequence_logits() -> None:
    """(B, T, K) logits are flattened over time and averaged."""
    logits = [torch.randn(4, 5, 3)]
    targets = torch.tensor([[0], [1], [2], [0]])
    loss = multi_head_cross_entropy(logits, targets)
    assert loss.ndim == 0


def test_multi_head_cross_entropy_is_zero_when_predictions_are_perfect() -> None:
    """Logits massively biased toward the correct class give near-zero loss."""
    # Predict class 0 with near-100% confidence for both trials.
    logits = [torch.tensor([[100.0, -100.0], [100.0, -100.0]])]
    targets = torch.tensor([[0], [0]])
    loss = multi_head_cross_entropy(logits, targets)
    assert loss.item() < 1e-5


def test_multi_head_cross_entropy_target_shape_mismatch_raises() -> None:
    """A target tensor with the wrong number of dimensions is rejected."""
    logits = [torch.randn(4, 3), torch.randn(4, 2)]
    bad_targets = torch.tensor([[0], [1], [2], [0]])  # only 1 dim — expected 2
    with pytest.raises(ValueError, match="targets must have shape"):
        multi_head_cross_entropy(logits, bad_targets)


def test_multi_head_cross_entropy_applies_class_weights() -> None:
    """Per-class weights change the loss value."""
    logits = [torch.tensor([[0.0, 1.0], [0.0, 1.0]])]
    targets = torch.tensor([[0], [1]])
    unweighted = multi_head_cross_entropy(logits, targets)
    weighted = multi_head_cross_entropy(
        logits, targets, class_weights_per_dim=[torch.tensor([5.0, 0.5])]
    )
    assert weighted.item() != unweighted.item()


def test_multi_head_cross_entropy_rejects_bad_weight_count() -> None:
    """``class_weights_per_dim`` length must equal ``logits_per_dim`` length."""
    logits = [torch.randn(2, 3), torch.randn(2, 4)]
    targets = torch.tensor([[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="class_weights_per_dim"):
        multi_head_cross_entropy(
            logits, targets, class_weights_per_dim=[torch.ones(3)]  # only 1
        )


# ───────────────────────── CC.7 — WeightedLoss='' unweighted path ─────────────────────────


def test_unweighted_cross_entropy_via_none(
) -> None:
    """``class_weights_per_dim=None`` exercises the unweighted branch.

    MATLAB ``cgg_getWeightsForLoss.m`` lines 8-14 (called from the
    active ``cgg_trainNetwork.m`` line 386):
      switch WeightedLoss
        case 'Inverse'  -> per-class weights
        otherwise       -> Weights = cell(0)  (empty)

    The Python port maps ``None`` to that ``otherwise`` branch —
    ``F.cross_entropy`` receives ``weight=None`` which is plain CE
    without class weighting. Regression-pin for the CC.7
    ``WeightedLoss=''`` config alternative.
    """
    logits = [torch.randn(4, 3), torch.randn(4, 2)]
    targets = torch.tensor([[0, 0], [1, 1], [2, 0], [0, 1]])
    loss_none = multi_head_cross_entropy(logits, targets, class_weights_per_dim=None)
    loss_default = multi_head_cross_entropy(logits, targets)  # default = None
    assert float(loss_none) == pytest.approx(float(loss_default))


def test_unweighted_differs_from_inverse_weighted(
) -> None:
    """Unweighted CE differs from inverse-frequency-weighted CE on imbalanced data."""
    # Heavily imbalanced labels: dim 0 has class 0 4x, class 1 1x.
    targets = torch.tensor([[0], [0], [0], [0], [1]])
    logits = [torch.randn(5, 2)]
    unweighted = multi_head_cross_entropy(logits, targets, class_weights_per_dim=None)
    weights = inverse_frequency_class_weights(targets, num_classes_per_dim=[2])
    weighted = multi_head_cross_entropy(logits, targets, class_weights_per_dim=weights)
    assert float(unweighted) != pytest.approx(float(weighted))


# ───────────────────────── inverse_frequency_class_weights ─────────────────────────


def test_inverse_frequency_weights_returns_one_per_dim() -> None:
    """One weight tensor per output dimension."""
    targets = torch.tensor([[0, 1], [0, 0], [1, 1], [2, 0]])
    weights = inverse_frequency_class_weights(targets, num_classes_per_dim=[3, 2])
    assert len(weights) == 2
    assert weights[0].shape == (3,)
    assert weights[1].shape == (2,)


def test_inverse_frequency_weights_larger_for_rare_classes() -> None:
    """A rarer class gets a larger weight than a common one."""
    targets = torch.tensor([[0]] * 10 + [[1]] * 1)  # class 0 = 10x as common
    weights = inverse_frequency_class_weights(targets, num_classes_per_dim=[2])
    assert weights[0][1] > weights[0][0]


def test_inverse_frequency_weights_handle_missing_classes() -> None:
    """A class with zero training examples gets weight 1.0 (no NaNs)."""
    targets = torch.tensor([[0], [0], [0]])  # class 1 never appears
    weights = inverse_frequency_class_weights(targets, num_classes_per_dim=[2])
    assert torch.isfinite(weights[0]).all()
    assert weights[0][1].item() == 1.0


def test_inverse_frequency_weights_empty_targets_raises() -> None:
    """Computing weights from zero trials is a programming error."""
    with pytest.raises(ValueError):
        inverse_frequency_class_weights(
            torch.zeros((0, 2), dtype=torch.int64),
            num_classes_per_dim=[2, 3],
        )


# ───────────────────────── aggregate_total_loss ─────────────────────────


def test_aggregate_classification_only() -> None:
    """With only classification active, total = w * classification."""
    cls = torch.tensor(2.5)
    total, info = aggregate_total_loss(
        classification_loss=cls, weights={"classification": 10.0}
    )
    assert total.item() == pytest.approx(25.0)
    assert info.classification is cls
    assert info.reconstruction is None


def test_aggregate_default_weights_are_one() -> None:
    """Missing weight keys default to 1.0."""
    cls = torch.tensor(2.5)
    total, _ = aggregate_total_loss(classification_loss=cls, weights={})
    assert total.item() == pytest.approx(2.5)


def test_aggregate_sums_multiple_components() -> None:
    """When ELBO is active too, total = w_cls * cls + w_recon * recon + w_kl * kl."""
    total, _ = aggregate_total_loss(
        classification_loss=torch.tensor(1.0),
        reconstruction_loss=torch.tensor(2.0),
        kl_loss=torch.tensor(3.0),
        weights={"classification": 10.0, "reconstruction": 100.0, "kl": 1.0},
    )
    expected = 10.0 * 1.0 + 100.0 * 2.0 + 1.0 * 3.0
    assert total.item() == pytest.approx(expected)


def test_aggregate_is_differentiable() -> None:
    """The total loss is differentiable; calling backward propagates gradients."""
    x = torch.tensor(2.0, requires_grad=True)
    cls = (x ** 2)  # scalar
    total, _ = aggregate_total_loss(
        classification_loss=cls, weights={"classification": 3.0}
    )
    total.backward()
    # d/dx [3 * x^2] = 6x; at x=2 that's 12.
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(12.0)
