"""Tests for the per-window flatten/unflatten layers + 5-D data path.

Exercises the data-restructure to ``(W, T, A, C)`` end-to-end: the
:class:`SyntheticTrialDataset` emits the canonical 5-D batched shape
when configured for ``T > 1`` or ``A > 1``, the composite flattens
before the encoder, and the decoder's reconstruction is unflattened
back to 5-D for the loss.
"""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.data.dataset import SyntheticTrialDataset, collate_trials
from neural_data_decoding.models.composite import (
    build_variational_autoencoder,
    build_variational_composite,
)
from neural_data_decoding.models.layers.data_prep import (
    FlattenPerWindow,
    UnflattenPerWindow,
)


# ───────────────────────── FlattenPerWindow ─────────────────────────


def test_flatten_5d_collapses_within_window_dims() -> None:
    """``(B, W, T, A, C) → (B, W, T*A*C)``."""
    f = FlattenPerWindow()
    x = torch.randn(2, 5, 3, 4, 7)
    y = f(x)
    assert y.shape == (2, 5, 3 * 4 * 7)


def test_flatten_3d_is_passthrough_backwards_compat() -> None:
    """3-D input is returned unchanged (backwards-compat)."""
    f = FlattenPerWindow()
    x = torch.randn(2, 5, 8)
    y = f(x)
    assert y is x or torch.equal(y, x)
    assert y.shape == x.shape


def test_flatten_wrong_ndim_raises() -> None:
    f = FlattenPerWindow()
    with pytest.raises(ValueError, match="3-D fallback"):
        f(torch.randn(2, 5))  # 2-D


# ───────────────────────── UnflattenPerWindow ─────────────────────────


def test_unflatten_multi_dim_expands_to_5d() -> None:
    """When ``t > 1`` or ``a > 1``, ``(B, W, T*A*C) → (B, W, T, A, C)``."""
    u = UnflattenPerWindow(t=3, a=4, c=7)
    x = torch.randn(2, 5, 3 * 4 * 7)
    y = u(x)
    assert y.shape == (2, 5, 3, 4, 7)


def test_unflatten_singleton_is_passthrough() -> None:
    """When ``t = a = 1`` the layer is identity on the 3-D input."""
    u = UnflattenPerWindow(t=1, a=1, c=8)
    x = torch.randn(2, 5, 8)
    y = u(x)
    assert torch.equal(y, x)


def test_unflatten_trailing_axis_mismatch_raises() -> None:
    u = UnflattenPerWindow(t=2, a=2, c=3)
    with pytest.raises(ValueError, match="trailing axis"):
        u(torch.randn(2, 5, 7))  # 7 != 2*2*3


def test_unflatten_invalid_dims_raise() -> None:
    with pytest.raises(ValueError, match="positive"):
        UnflattenPerWindow(t=0, a=1, c=1)
    with pytest.raises(ValueError, match="positive"):
        UnflattenPerWindow(t=1, a=1, c=-1)


# ───────────────────────── Dataset multi-dim emit ─────────────────────────


def test_dataset_singleton_emits_2d() -> None:
    """Default ``T = A = 1`` emits 2-D ``(W, C)`` for backwards compat."""
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=2,
        num_samples=4, num_features=3,
        num_classes_per_dim=[2], seed=0,
    )
    x, _, _ = ds[0]
    assert x.shape == (4, 3)


def test_dataset_multidim_emits_4d() -> None:
    """``T = 2``, ``A = 3`` emits 4-D ``(W, T, A, C)``."""
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=2,
        num_samples=4, num_features=3,
        num_classes_per_dim=[2],
        samples_per_window=2, num_areas=3, seed=0,
    )
    x, _, _ = ds[0]
    assert x.shape == (4, 2, 3, 3)


def test_dataset_partial_singleton_emits_4d() -> None:
    """``T = 1, A = 2`` still emits 4-D (any non-singleton triggers full shape)."""
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=2,
        num_samples=4, num_features=3,
        num_classes_per_dim=[2],
        samples_per_window=1, num_areas=2, seed=0,
    )
    x, _, _ = ds[0]
    assert x.shape == (4, 1, 2, 3)


def test_collate_stacks_multidim() -> None:
    """``collate_trials`` stacks 4-D per-trial into 5-D ``(B, W, T, A, C)``."""
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=3,
        num_samples=4, num_features=3,
        num_classes_per_dim=[2],
        samples_per_window=2, num_areas=2, seed=0,
    )
    batch = collate_trials([ds[0], ds[1], ds[2]])
    assert batch["x"].shape == (3, 4, 2, 2, 3)


# ───────────────────────── Composite end-to-end with multi-dim ─────────────────────────


def _multidim_cfg(*, t: int, a: int, c: int) -> dict:
    return {
        "in_features": c,
        "samples_per_window": t,
        "num_areas": a,
        "hidden_sizes": [16, 4],
        "num_classes_per_dim": [3],
        "classifier_hidden_size": [8, 4],
        "transform": "GRU",
        "loss_type_decoder": "MSE",
    }


def test_variational_composite_multidim_forward_shapes() -> None:
    """Composite consumes ``(B, W, T, A, C)`` and reconstructs the same shape."""
    cfg = _multidim_cfg(t=2, a=3, c=4)
    model = build_variational_composite(cfg)
    model.eval()
    x = torch.randn(2, 5, 2, 3, 4)
    out = model(x)
    assert out.reconstruction is not None
    assert out.reconstruction.shape == (2, 5, 2, 3, 4)
    assert out.logits[0].shape == (2, 5, 3)


def test_variational_autoencoder_multidim_forward_shapes() -> None:
    """Stage 1 autoencoder respects the multi-dim shape contract."""
    cfg = _multidim_cfg(t=2, a=3, c=4)
    ae = build_variational_autoencoder(cfg)
    ae.eval()
    x = torch.randn(2, 5, 2, 3, 4)
    out = ae(x)
    assert out.reconstruction.shape == (2, 5, 2, 3, 4)


def test_variational_composite_multidim_gradients_flow() -> None:
    """Backward pass populates gradients with 5-D input."""
    cfg = _multidim_cfg(t=2, a=3, c=4)
    model = build_variational_composite(cfg)
    x = torch.randn(2, 5, 2, 3, 4)
    out = model(x)
    assert out.reconstruction is not None
    loss = out.reconstruction.pow(2).sum() + sum(lg.sum() for lg in out.logits)
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0


def test_variational_composite_singleton_still_3d_backwards_compat() -> None:
    """Without samples_per_window/num_areas, the 3-D path still works."""
    cfg = {
        "in_features": 6,
        "hidden_sizes": [16, 4],
        "num_classes_per_dim": [3],
        "classifier_hidden_size": [8, 4],
        "transform": "GRU",
        "loss_type_decoder": "MSE",
    }
    model = build_variational_composite(cfg)
    model.eval()
    x = torch.randn(2, 5, 6)
    out = model(x)
    assert out.reconstruction is not None
    assert out.reconstruction.shape == (2, 5, 6)
