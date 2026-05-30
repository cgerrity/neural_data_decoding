"""Unit tests for the Eq. 2 interpolated multi-head cross-entropy helper.

Analytical correctness: with one-hot targets, the interpolated CE
collapses to ``-log(c * p_target + (1 - c))``. Verified against a hand-
computed expected value for a known small input. Edge cases: ``c = 1``
recovers standard CE; ``c = 0`` gives zero loss (predicting the target
exactly). Shape validation: trial-only (K_conf=1) broadcasts across
dims; per-dim (K_conf=num_dims) is consumed directly.
"""

from __future__ import annotations

import math

import pytest
import torch

from neural_data_decoding.training.losses.classification import (
    interpolated_multi_head_cross_entropy,
    multi_head_cross_entropy,
)


# ───────────────────────── analytical correctness ─────────────────────────


def test_interpolated_ce_matches_closed_form_on_known_input() -> None:
    """Single-dim, single-trial, single-timestep: hand-computed match."""
    # Logits → after softmax produce known probabilities.
    # Use sharp logits so p_target is precise.
    logits = torch.tensor([[[2.0, 0.0, 0.0]]])  # (B=1, T=1, K=3)
    targets = torch.tensor([[0]])                # target class index 0
    total_dropped = torch.tensor([[0.7]])        # (B=1, K_conf=1)

    p_target = torch.softmax(logits[0, 0], dim=-1)[0].item()
    c = 0.7
    expected_per_loss = -math.log(c * p_target + (1 - c))

    loss = interpolated_multi_head_cross_entropy(
        [logits], targets, total_dropped,
    )
    assert float(loss) == pytest.approx(expected_per_loss, rel=1e-7, abs=1e-7)


def test_interpolated_ce_at_c_equals_one_matches_standard_ce() -> None:
    """When confidence = 1 (no dropout-out), interpolated CE == standard CE."""
    torch.manual_seed(0)
    logits = [torch.randn(3, 4, 5), torch.randn(3, 4, 2)]
    targets = torch.tensor([[1, 0], [2, 1], [0, 1]])
    total_dropped = torch.ones(3, 2)  # c=1 → interpolated == raw probs

    interp = interpolated_multi_head_cross_entropy(logits, targets, total_dropped)
    standard = multi_head_cross_entropy(logits, targets)
    # Both should match when c=1 (target-class log-prob is identical).
    assert float(interp) == pytest.approx(float(standard), rel=1e-5, abs=1e-5)


def test_interpolated_ce_at_c_equals_zero_yields_zero_loss() -> None:
    """When confidence = 0, prediction is replaced by target → log(1) = 0."""
    torch.manual_seed(0)
    logits = [torch.randn(3, 4, 5)]
    targets = torch.tensor([[2], [1], [0]])
    total_dropped = torch.zeros(3, 1)

    loss = interpolated_multi_head_cross_entropy(logits, targets, total_dropped)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


# ───────────────────────── shape handling ─────────────────────────


def test_interpolated_ce_with_trial_only_confidence_broadcasts_across_dims() -> None:
    """Trial-only (K_conf=1) confidence shares the same c across dims."""
    torch.manual_seed(0)
    logits = [torch.randn(2, 3, 4), torch.randn(2, 3, 5)]
    targets = torch.tensor([[1, 2], [0, 4]])
    total_dropped = torch.tensor([[0.5], [0.8]])  # (B=2, K_conf=1)

    # Equivalent expanded version: same c value duplicated per dim.
    total_dropped_expanded = total_dropped.expand(-1, 2)

    loss_a = interpolated_multi_head_cross_entropy(logits, targets, total_dropped)
    loss_b = interpolated_multi_head_cross_entropy(logits, targets, total_dropped_expanded)
    assert float(loss_a) == pytest.approx(float(loss_b), rel=1e-7, abs=1e-7)


def test_interpolated_ce_per_dim_confidence() -> None:
    """Per-dim confidence: each dim consumes its own column of total_dropped."""
    torch.manual_seed(0)
    logits = [torch.randn(2, 3, 4), torch.randn(2, 3, 5)]
    targets = torch.tensor([[1, 2], [0, 4]])
    total_dropped = torch.tensor([[0.3, 0.9], [0.7, 0.4]])  # (B=2, K_conf=2)

    # Loss is the sum of per-dim mean losses; should run without error and
    # produce a positive scalar (random logits guarantee non-zero CE).
    loss = interpolated_multi_head_cross_entropy(logits, targets, total_dropped)
    assert loss.ndim == 0
    assert float(loss) > 0


def test_interpolated_ce_with_class_weights() -> None:
    """class_weights_per_dim scales each per-trial loss by the target class's weight."""
    torch.manual_seed(0)
    logits = [torch.randn(3, 4, 5)]
    targets = torch.tensor([[0], [1], [2]])
    total_dropped = torch.tensor([[0.5], [0.5], [0.5]])
    # Class 0 weight = 3, others = 1. Should triple trial 0's contribution.
    weights = [torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0])]

    loss_weighted = interpolated_multi_head_cross_entropy(
        logits, targets, total_dropped, class_weights_per_dim=weights,
    )
    loss_unweighted = interpolated_multi_head_cross_entropy(
        logits, targets, total_dropped,
    )
    # Weighted ≠ unweighted (since the per-trial losses differ in scale).
    assert float(loss_weighted) != pytest.approx(float(loss_unweighted))


# ───────────────────────── input validation ─────────────────────────


def test_interpolated_ce_rejects_empty_logits() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        interpolated_multi_head_cross_entropy(
            [], torch.zeros(2, 0), torch.zeros(2, 1),
        )


def test_interpolated_ce_rejects_wrong_target_shape() -> None:
    with pytest.raises(ValueError, match="targets must have shape"):
        interpolated_multi_head_cross_entropy(
            [torch.zeros(2, 3, 4)], torch.zeros(2),
            torch.zeros(2, 1),
        )


def test_interpolated_ce_rejects_wrong_total_dropped_shape() -> None:
    with pytest.raises(ValueError, match="must be 1.*or num_dimensions"):
        interpolated_multi_head_cross_entropy(
            [torch.zeros(2, 3, 4), torch.zeros(2, 3, 5)],
            torch.zeros(2, 2, dtype=torch.long),
            torch.zeros(2, 3),  # K_conf=3, but num_dims=2 → reject
        )


def test_interpolated_ce_rejects_2d_logits() -> None:
    """The CE assumes (B, T, K) — flat (B, K) is rejected (the orchestrator
    always passes sequence outputs from the variational classifier)."""
    with pytest.raises(ValueError, match="requires.*batch, time, num_classes"):
        interpolated_multi_head_cross_entropy(
            [torch.zeros(2, 5)],
            torch.zeros(2, 1, dtype=torch.long),
            torch.zeros(2, 1),
        )
