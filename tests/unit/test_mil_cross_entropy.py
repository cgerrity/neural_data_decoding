"""Unit tests for MIL multi-head cross-entropy + the MIL flag on the
interpolated variant.

The MIL pipeline (matches MATLAB ``cgg_softmaxLayer('SCT')`` →
aggregation → CE):

1. Per-dim logits ``(B, T, K_d)`` → joint softmax over ``(T, K_d)`` →
   probabilities summing to 1 over ``T*K_d`` per trial.
2. Sum over ``T`` → marginal ``(B, K_d)`` summing to 1 over ``K_d``.
3. NLL on the marginal of the target class.

Tested invariants:
- Joint softmax + marginal sums to 1 per trial.
- Single-timestep input (``T=1``) reduces to standard softmax+NLL.
- Edge: deterministic logits → predictable loss.
- MIL + interpolated CE composition (``mil=True`` flag).
"""

from __future__ import annotations

import math

import pytest
import torch

from neural_data_decoding.training.losses.classification import (
    interpolated_multi_head_cross_entropy,
    mil_multi_head_cross_entropy,
    multi_head_cross_entropy,
)


# ───────────────────────── mil_multi_head_cross_entropy ─────────────────────────


def test_mil_marginal_sums_to_one_per_trial() -> None:
    """Joint softmax → sum over T → marginal sums to 1 over K per trial."""
    from neural_data_decoding.training.losses.classification import _mil_marginal_probs
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 3)        # (B=4, T=7, K=3)
    marginal = _mil_marginal_probs(logits)
    assert marginal.shape == (4, 3)
    # Each trial's marginal sums to 1 over K (because joint summed to 1 over T*K).
    torch.testing.assert_close(
        marginal.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6,
    )


def test_mil_reduces_to_standard_softmax_at_t_equals_one() -> None:
    """With T=1, the joint softmax over (T, K) ≡ softmax over K.

    Marginal then equals the softmax probabilities directly, so the loss
    equals standard cross-entropy on the same logits (up to scaling
    differences from how each kernel reduces).
    """
    torch.manual_seed(0)
    logits = [torch.randn(3, 1, 4)]      # T=1
    targets = torch.tensor([[1], [2], [0]])

    mil_loss = mil_multi_head_cross_entropy(logits, targets)
    standard_loss = multi_head_cross_entropy(logits, targets)
    # At T=1, the two formulations produce the same scalar.
    assert float(mil_loss) == pytest.approx(float(standard_loss), rel=1e-5, abs=1e-5)


def test_mil_closed_form_on_known_input() -> None:
    """Single trial, T=2, K=2: hand-compute the joint softmax + marginal + NLL."""
    # logits: (B=1, T=2, K=2) — design so the joint is easy.
    logits = [torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])]   # list with one (1, 2, 2) tensor
    targets = torch.tensor([[0]])                          # target class index 0

    # Joint softmax over (T, K) on the 4-element flat vector [1, 0, 0, 1].
    # Hand-compute: exp = [e, 1, 1, e]; sum = 2e + 2.
    e = math.e
    z = 2 * e + 2
    # Probabilities for cells (t=0,k=0)=e/z, (0,1)=1/z, (1,0)=1/z, (1,1)=e/z.
    # Marginal for K=0 = (e + 1) / z.
    expected_marginal_target = (e + 1) / z
    expected_loss = -math.log(expected_marginal_target)

    loss = mil_multi_head_cross_entropy(logits, targets)
    assert float(loss) == pytest.approx(expected_loss, rel=1e-7, abs=1e-7)


def test_mil_with_class_weights_scales_per_trial() -> None:
    """class_weights_per_dim scales each trial's loss by the target-class weight."""
    torch.manual_seed(0)
    logits = [torch.randn(3, 4, 5)]
    targets = torch.tensor([[0], [1], [2]])
    weights = [torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0])]   # class 0 → 3× weight

    weighted = mil_multi_head_cross_entropy(
        logits, targets, class_weights_per_dim=weights,
    )
    unweighted = mil_multi_head_cross_entropy(logits, targets)
    assert float(weighted) != pytest.approx(float(unweighted))


# ───────────────────────── input validation ─────────────────────────


def test_mil_rejects_empty_logits() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mil_multi_head_cross_entropy([], torch.zeros(2, 0))


def test_mil_rejects_2d_logits() -> None:
    """MIL requires a time axis (T) — the whole point is bag aggregation."""
    with pytest.raises(ValueError, match="requires.*time"):
        mil_multi_head_cross_entropy(
            [torch.zeros(2, 5)], torch.zeros(2, 1, dtype=torch.long),
        )


def test_mil_rejects_wrong_target_shape() -> None:
    with pytest.raises(ValueError, match="targets must have shape"):
        mil_multi_head_cross_entropy(
            [torch.zeros(2, 3, 4)], torch.zeros(2),
        )


# ───────────────────────── interpolated CE with mil=True flag ─────────────────────────


def test_interpolated_ce_with_mil_uses_marginal_for_interpolation() -> None:
    """mil=True applies MIL transformation, then closed-form interpolation on marginal."""
    torch.manual_seed(0)
    logits = [torch.randn(3, 4, 5)]
    targets = torch.tensor([[2], [1], [0]])
    total_dropped = torch.tensor([[0.7], [0.5], [1.0]])  # (B=3, K_conf=1)

    loss = interpolated_multi_head_cross_entropy(
        logits, targets, total_dropped, mil=True,
    )
    # Verify it ran and produced a positive scalar.
    assert loss.ndim == 0
    assert float(loss) > 0


def test_interpolated_ce_mil_at_c_equals_one_matches_standalone_mil() -> None:
    """With c=1 everywhere, mil=True interpolated CE ≡ standalone mil CE."""
    torch.manual_seed(0)
    logits = [torch.randn(3, 4, 5)]
    targets = torch.tensor([[2], [1], [0]])
    total_dropped = torch.ones(3, 1)

    interp = interpolated_multi_head_cross_entropy(
        logits, targets, total_dropped, mil=True,
    )
    standalone = mil_multi_head_cross_entropy(logits, targets)
    assert float(interp) == pytest.approx(float(standalone), rel=1e-5, abs=1e-5)


def test_interpolated_ce_mil_at_c_equals_zero_yields_zero_loss() -> None:
    """With c=0, prediction is replaced by target → log(1) = 0."""
    torch.manual_seed(0)
    logits = [torch.randn(3, 4, 5)]
    targets = torch.tensor([[2], [1], [0]])
    total_dropped = torch.zeros(3, 1)

    loss = interpolated_multi_head_cross_entropy(
        logits, targets, total_dropped, mil=True,
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


def test_interpolated_ce_mil_differs_from_non_mil() -> None:
    """mil=True and mil=False produce different scalar values (different math)."""
    torch.manual_seed(0)
    logits = [torch.randn(3, 4, 5)]
    targets = torch.tensor([[2], [1], [0]])
    total_dropped = torch.tensor([[0.5], [0.5], [0.5]])

    mil_path = interpolated_multi_head_cross_entropy(
        logits, targets, total_dropped, mil=True,
    )
    standard_path = interpolated_multi_head_cross_entropy(
        logits, targets, total_dropped, mil=False,
    )
    assert float(mil_path) != pytest.approx(float(standard_path), abs=1e-5)


# ───────────────────────── aggregate_classifier_predictions ─────────────────────────


def test_aggregate_non_mil_matches_mean_of_per_timestep_softmax() -> None:
    """Non-MIL aggregate = mean of per-timestep softmax (uniform-prior aggregation)."""
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 4)               # (B=3, T=7, K=4)
    aggregated = aggregate_from_helper([logits], mil_mode=False)
    expected = torch.softmax(logits, dim=-1).mean(dim=1)  # uniform prior over T
    torch.testing.assert_close(aggregated[0], expected, rtol=1e-6, atol=1e-6)


def test_aggregate_mil_matches_marginal() -> None:
    """MIL aggregate = joint-softmax marginal over T (already a valid distribution)."""
    from neural_data_decoding.training.losses.classification import _mil_marginal_probs
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 4)
    aggregated = aggregate_from_helper([logits], mil_mode=True)
    expected = _mil_marginal_probs(logits)
    torch.testing.assert_close(aggregated[0], expected, rtol=1e-6, atol=1e-6)


def test_aggregate_rows_sum_to_one_per_trial() -> None:
    """Both modes produce per-trial distributions that sum to 1 over K."""
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 3)
    for mil_mode in (False, True):
        aggregated = aggregate_from_helper([logits], mil_mode=mil_mode)
        torch.testing.assert_close(
            aggregated[0].sum(dim=-1), torch.ones(4),
            rtol=1e-6, atol=1e-6,
        )


def test_aggregate_supports_multiple_dims() -> None:
    """Multi-head: one aggregated distribution per output dim, correct shapes."""
    torch.manual_seed(0)
    logits = [torch.randn(2, 5, 3), torch.randn(2, 5, 7)]
    aggregated = aggregate_from_helper(logits, mil_mode=False)
    assert len(aggregated) == 2
    assert aggregated[0].shape == (2, 3)
    assert aggregated[1].shape == (2, 7)


def test_aggregate_rejects_2d_logits() -> None:
    """Aggregation requires a time axis."""
    with pytest.raises(ValueError, match="requires.*time"):
        aggregate_from_helper([torch.zeros(2, 5)], mil_mode=False)


def test_aggregate_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_from_helper([], mil_mode=False)


def aggregate_from_helper(logits, *, mil_mode):
    """Tiny wrapper so the test functions read cleanly."""
    from neural_data_decoding.training.losses.classification import (
        aggregate_classifier_predictions,
    )
    return aggregate_classifier_predictions(logits, mil_mode=mil_mode)
