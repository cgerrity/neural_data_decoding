"""Tests for the offset/scale augmentation loss + decoder block (CC.6)."""

from __future__ import annotations

import math

import pytest
import torch

from neural_data_decoding.models.layers.offset_scale import (
    LearnableOffsetScale,
    find_learnable_offset_scale,
)
from neural_data_decoding.training.losses.offset_and_scale import (
    offset_and_scale_loss,
    offset_and_scale_targets,
)


# ───────────────────────── Target computation ─────────────────────────


def test_targets_mX_plus_b_plus_X_default() -> None:
    """For 'mX+b+X': T_Scale = range(x)-1, T_Offset = median(x)."""
    # Construct x where range and median are easy to verify.
    # 5-D (B, W, T, A, C) with B=1, W=1, T=1, A=1, C=4.
    x = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]])
    t_scale, t_offset = offset_and_scale_targets(x)
    # range = 4 - 1 = 3 → T_Scale = 3 - 1 = 2
    # median = (2+3)/2 = 2.5 → T_Offset = 2.5
    assert t_scale.shape == (1, 1, 1, 1)
    assert t_offset.shape == (1, 1, 1, 1)
    assert math.isclose(float(t_scale), 2.0, abs_tol=1e-5)
    assert math.isclose(float(t_offset), 2.5, abs_tol=1e-5)


def test_targets_m_X_plus_b_variant() -> None:
    """For 'm(X+b)': T_Scale = range(x), T_Offset = median(x) / (T_Scale + eps)."""
    x = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]])
    t_scale, t_offset = offset_and_scale_targets(x, augment_equation="m(X+b)")
    assert math.isclose(float(t_scale), 3.0, abs_tol=1e-5)
    # median / (range + epsilon) ≈ 2.5 / 3.00001 ≈ 0.8333
    assert math.isclose(float(t_offset), 2.5 / (3.0 + 1e-5), abs_tol=1e-4)


def test_targets_reduce_axis_collapses_correctly() -> None:
    """Reduced shape: ``x`` axis removed by reduction matches the spatial_dim."""
    x = torch.randn(2, 5, 4, 3, 8)  # (B, W, T, A, C)
    t_scale, t_offset = offset_and_scale_targets(x)
    assert t_scale.shape == (2, 5, 4, 3)
    assert t_offset.shape == (2, 5, 4, 3)


# ───────────────────────── Loss kernel ─────────────────────────


def test_loss_is_zero_when_predictions_match_targets() -> None:
    """If Y == T, the loss is 0."""
    x = torch.randn(2, 5, 4, 3, 8)
    t_scale, t_offset = offset_and_scale_targets(x)
    loss = offset_and_scale_loss(x, t_scale, t_offset)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-5)


def test_loss_is_positive_for_random_predictions() -> None:
    x = torch.randn(2, 5, 4, 3, 8)
    t_scale, t_offset = offset_and_scale_targets(x)
    y_scale = t_scale + torch.randn_like(t_scale) * 0.5
    y_offset = t_offset + torch.randn_like(t_offset) * 0.5
    loss = offset_and_scale_loss(x, y_scale, y_offset)
    assert float(loss) > 0


def test_loss_normalization_by_batch_size() -> None:
    """Doubling the batch with the same per-sample error halves the per-element mean.

    Matches MATLAB's ``l2loss`` batch-size normalization (Critical
    Note #38) — total loss scales linearly with batch size, but the
    per-element loss stays the same after dividing by batch.
    """
    x = torch.randn(1, 5, 4, 3, 8)
    t_scale, t_offset = offset_and_scale_targets(x)
    y_scale = t_scale + 1.0  # constant offset
    y_offset = t_offset + 1.0
    loss_b1 = offset_and_scale_loss(x, y_scale, y_offset)
    # Double the batch by stacking — same per-sample error.
    x2 = torch.cat([x, x], dim=0)
    ts2, to2 = offset_and_scale_targets(x2)
    ys2, yo2 = ts2 + 1.0, to2 + 1.0
    loss_b2 = offset_and_scale_loss(x2, ys2, yo2)
    # 2x batch → sum doubles, batch_size doubles → loss equal.
    assert math.isclose(float(loss_b1), float(loss_b2), rel_tol=1e-4)


def test_loss_nan_mask_excludes_nan_slices() -> None:
    """Spatial slices containing any NaN are excluded from the loss."""
    x = torch.randn(2, 5, 4, 3, 8)
    x[0, 0, 0, 0, 0] = float("nan")
    t_scale, t_offset = offset_and_scale_targets(x)
    # Targets have NaN where x had NaN (because range/median NaN-propagate).
    # The loss must still be finite — NaN slices are masked out.
    y_scale = torch.zeros_like(t_scale)
    y_offset = torch.zeros_like(t_offset)
    loss = offset_and_scale_loss(x, y_scale, y_offset)
    assert torch.isfinite(loss)


def test_loss_shape_mismatch_raises() -> None:
    x = torch.randn(2, 5, 4, 3, 8)
    t_scale, t_offset = offset_and_scale_targets(x)
    with pytest.raises(ValueError, match="y_scale shape"):
        offset_and_scale_loss(x, torch.zeros(2, 5, 4, 7), t_offset)
    with pytest.raises(ValueError, match="y_offset shape"):
        offset_and_scale_loss(x, t_scale, torch.zeros(2, 5, 4, 7))


def test_loss_gradients_flow_to_predictions() -> None:
    """Backward pass populates gradients on Y_Scale and Y_Offset."""
    x = torch.randn(2, 5, 4, 3, 8)
    t_scale, t_offset = offset_and_scale_targets(x)
    y_scale = (t_scale + torch.randn_like(t_scale)).requires_grad_(True)
    y_offset = (t_offset + torch.randn_like(t_offset)).requires_grad_(True)
    loss = offset_and_scale_loss(x, y_scale, y_offset)
    loss.backward()
    assert y_scale.grad is not None and y_scale.grad.abs().sum() > 0
    assert y_offset.grad is not None and y_offset.grad.abs().sum() > 0


# ───────────────────────── LearnableOffsetScale module ─────────────────────────


def test_module_output_shapes() -> None:
    """Module: (B, W, latent) -> ((B, W, T, A), (B, W, T, A))."""
    head = LearnableOffsetScale(latent_dim=8, samples_per_window=4, num_areas=3)
    z = torch.randn(2, 5, 8)
    scale, offset = head(z)
    assert scale.shape == (2, 5, 4, 3)
    assert offset.shape == (2, 5, 4, 3)


def test_module_rejects_wrong_ndim() -> None:
    head = LearnableOffsetScale(latent_dim=8, samples_per_window=4, num_areas=3)
    with pytest.raises(ValueError, match="3-D z"):
        head(torch.randn(2, 8))  # 2-D


def test_module_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        LearnableOffsetScale(latent_dim=0, samples_per_window=4, num_areas=3)
    with pytest.raises(ValueError, match=">= 1"):
        LearnableOffsetScale(latent_dim=8, samples_per_window=0, num_areas=3)


def test_module_two_independent_heads() -> None:
    """scale_head and offset_head are separate Linear stacks."""
    head = LearnableOffsetScale(latent_dim=8, samples_per_window=4, num_areas=3)
    # Distinct nn.Sequential instances.
    assert head.scale_head is not head.offset_head


def test_module_gradient_flow() -> None:
    """Backward on the loss flows gradients into both heads."""
    head = LearnableOffsetScale(latent_dim=8, samples_per_window=4, num_areas=3)
    x = torch.randn(2, 5, 4, 3, 6)
    z = torch.randn(2, 5, 8)
    y_scale, y_offset = head(z)
    loss = offset_and_scale_loss(x, y_scale, y_offset)
    loss.backward()
    n_with_grad = sum(
        1 for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad >= 4  # both heads × 2 layers' weights


# ───────────────────────── Auto-activation helper ─────────────────────────


def test_find_learnable_offset_scale_returns_head_when_present() -> None:
    """``find_learnable_offset_scale`` locates the module via ``isinstance``."""
    import torch.nn as nn
    head = LearnableOffsetScale(latent_dim=4, samples_per_window=2, num_areas=2)
    composite = nn.Sequential(nn.Linear(4, 4), head, nn.Linear(4, 4))
    found = find_learnable_offset_scale(composite)
    assert found is head


def test_find_learnable_offset_scale_returns_none_when_absent() -> None:
    """No augmentation head in the tree → ``None`` (loss term is a no-op)."""
    import torch.nn as nn
    composite = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    assert find_learnable_offset_scale(composite) is None
