"""End-to-end tests for the 'Default' S&F variant (CC #3 Phase 2).

Exercises the per-window 2-D conv encoder/decoder
(:class:`PerWindowConvolutionalCoder`) wrapped with the leading/trailing
``Linear`` projection — operates on the new 5-D ``(B, W, T, A, C)`` data
layout, with conv kernels ``[1, n]`` over ``T`` (matching MATLAB's
within-window 2-D conv semantics).
"""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.models.composite import (
    build_variational_autoencoder,
    build_variational_composite,
)
from neural_data_decoding.models.stitching_fusion.convolutional import (
    PerWindowConvolutionalCoder,
)


# ───────────────────────── PerWindowConvolutionalCoder ─────────────────────────


def test_per_window_conv_encoder_shape_split_areas() -> None:
    """Encoder split-areas: (B, W, T, A, C) -> (B, W, T', A_out, C)."""
    enc = PerWindowConvolutionalCoder(
        num_areas=2, filter_hidden_sizes=[4, 8], kernel_t=3,
        coder="Encoder", stride_t=2, want_split_areas=True,
    )
    x = torch.randn(2, 5, 8, 2, 4)
    y = enc(x)
    # T reduces by stride twice (8 -> 4 -> 2); A becomes 8; C unchanged.
    assert y.shape == (2, 5, 2, 8, 4)


def test_per_window_conv_decoder_reverses_encoder() -> None:
    """Decoder unwinds the encoder back to original shape."""
    enc = PerWindowConvolutionalCoder(
        num_areas=2, filter_hidden_sizes=[4, 8], kernel_t=3,
        coder="Encoder", stride_t=2,
    )
    dec = PerWindowConvolutionalCoder(
        num_areas=2, filter_hidden_sizes=[4, 8], kernel_t=3,
        coder="Decoder", stride_t=2,
    )
    x = torch.randn(2, 5, 8, 2, 4)
    z = dec(enc(x))
    assert z.shape == x.shape


def test_per_window_conv_cross_area_mode() -> None:
    """want_split_areas=False uses ungrouped convs (cross-area mixing)."""
    enc = PerWindowConvolutionalCoder(
        num_areas=2, filter_hidden_sizes=[6, 12], kernel_t=3,
        coder="Encoder", stride_t=2, want_split_areas=False,
    )
    x = torch.randn(2, 5, 8, 2, 4)
    y = enc(x)
    assert y.shape == (2, 5, 2, 12, 4)


def test_per_window_conv_kernel_does_not_mix_C() -> None:
    """Kernel shape (1, kernel_t) preserves the per-area C axis exactly.

    Verifies the user-flagged semantics: ``[1, n]`` kernels operate over
    ``T`` only and do **not** propagate information across the ``C``
    (channels-per-area) axis. We perturb a single (C, ?) row and verify
    other C rows are unchanged.
    """
    enc = PerWindowConvolutionalCoder(
        num_areas=1, filter_hidden_sizes=[2], kernel_t=3,
        coder="Encoder", stride_t=1, want_split_areas=False,
        want_resnet=False, repetitions_per_block=1,
    )
    enc.eval()
    x = torch.zeros(1, 1, 4, 1, 3)  # (B=1, W=1, T=4, A=1, C=3)
    y_base = enc(x).detach().clone()
    # Perturb only C=0 across all T.
    x_perturbed = x.clone()
    x_perturbed[0, 0, :, 0, 0] = 1.0
    y_perturbed = enc(x_perturbed).detach()
    # C=0 outputs should differ; C=1 and C=2 should match exactly (the
    # kernel never sees C=0 information when computing C=1 or C=2).
    assert not torch.allclose(y_base[..., 0], y_perturbed[..., 0])
    assert torch.allclose(y_base[..., 1], y_perturbed[..., 1])
    assert torch.allclose(y_base[..., 2], y_perturbed[..., 2])


def test_per_window_conv_gradients_flow() -> None:
    """Backward pass populates gradients."""
    enc = PerWindowConvolutionalCoder(
        num_areas=2, filter_hidden_sizes=[4, 8], kernel_t=3,
        coder="Encoder", stride_t=2,
    )
    y = enc(torch.randn(2, 5, 8, 2, 4))
    y.sum().backward()
    for p in enc.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_per_window_conv_input_a_axis_validation() -> None:
    """Wrong A axis raises ValueError with a clear message."""
    enc = PerWindowConvolutionalCoder(
        num_areas=2, filter_hidden_sizes=[4], kernel_t=3,
        coder="Encoder", stride_t=2,
    )
    with pytest.raises(ValueError, match="A axis"):
        enc(torch.randn(2, 5, 8, 3, 4))  # A=3, expected 2


# ───────────────────────── Default S&F via composite ─────────────────────────


def _default_cfg(*, t: int = 8, a: int = 1, c: int = 4) -> dict:
    return {
        "in_features": c,
        "samples_per_window": t,
        "num_areas": a,
        "hidden_sizes": [16, 4],
        "num_classes_per_dim": [3],
        "classifier_hidden_size": [8, 4],
        "transform": "GRU",
        "loss_type_decoder": "MSE",
        "stitching_and_fusion_layer": "Default",
    }


def test_default_sf_composite_forward_5d_shape() -> None:
    """Composite with Default S&F: 5-D input -> 5-D reconstruction matching input shape."""
    cfg = _default_cfg(t=8, a=1, c=4)
    model = build_variational_composite(cfg)
    model.eval()
    x = torch.randn(2, 5, 8, 1, 4)
    out = model(x)
    assert out.reconstruction is not None
    assert out.reconstruction.shape == x.shape


def test_default_sf_autoencoder_forward_5d_shape() -> None:
    """Stage 1 autoencoder with Default S&F."""
    cfg = _default_cfg(t=8, a=1, c=4)
    ae = build_variational_autoencoder(cfg)
    ae.eval()
    x = torch.randn(2, 5, 8, 1, 4)
    out = ae(x)
    assert out.reconstruction.shape == x.shape


def test_default_sf_gradients_flow_through_conv_and_linear() -> None:
    """Backward pass populates gradients in both the conv stack and Linear projections."""
    cfg = _default_cfg(t=8, a=1, c=4)
    model = build_variational_composite(cfg)
    x = torch.randn(2, 5, 8, 1, 4)
    out = model(x)
    assert out.reconstruction is not None
    loss = out.reconstruction.pow(2).sum() + sum(lg.sum() for lg in out.logits)
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad >= 2  # at minimum the conv weights + one Linear
