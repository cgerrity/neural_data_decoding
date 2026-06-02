"""Tests for the frozen PCA encode/decode layers (CC.2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_data_decoding.models.layers.pca import (
    PCADecodingLayer,
    PCAEncodingLayer,
    fit_pca_encoder_decoder,
)


# ───────────────────────── Construction validation ─────────────────────────


def test_encoder_rejects_non_positive_dims() -> None:
    with pytest.raises(ValueError, match="in_features"):
        PCAEncodingLayer(in_features=0, n_components=4)
    with pytest.raises(ValueError, match="n_components"):
        PCAEncodingLayer(in_features=8, n_components=0)


def test_encoder_rejects_n_components_greater_than_in_features() -> None:
    with pytest.raises(ValueError, match="must be <= in_features"):
        PCAEncodingLayer(in_features=4, n_components=8)


def test_decoder_rejects_non_positive_dims() -> None:
    with pytest.raises(ValueError, match="must both be >= 1"):
        PCADecodingLayer(in_features=0, n_components=4)


# ───────────────────────── Fit + frozen buffer semantics ─────────────────────────


def test_components_and_mean_are_buffers_not_parameters() -> None:
    """PCA components and mean must NOT be learnable."""
    enc = PCAEncodingLayer(in_features=8, n_components=4)
    param_names = {n for n, _ in enc.named_parameters()}
    assert param_names == set()  # no learnable params
    buffer_names = {n for n, _ in enc.named_buffers()}
    assert "components" in buffer_names and "mean" in buffer_names


def test_fit_populates_components_from_sklearn() -> None:
    """After fit, components match sklearn's PCA output."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((64, 8)).astype(np.float32)
    enc = PCAEncodingLayer(in_features=8, n_components=4)
    enc.fit(data)
    assert enc.is_fitted
    # Components shape: (n_components, in_features).
    assert enc.components.shape == (4, 8)
    assert enc.mean.shape == (8,)
    # Centered data projected onto components should match sklearn.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    pca.fit(data)
    np.testing.assert_allclose(
        enc.components.numpy(), pca.components_, rtol=1e-5,
    )
    np.testing.assert_allclose(enc.mean.numpy(), pca.mean_, rtol=1e-5)


def test_fit_rejects_wrong_shape() -> None:
    enc = PCAEncodingLayer(in_features=8, n_components=4)
    with pytest.raises(ValueError, match="expects"):
        enc.fit(np.zeros((10, 6)))  # wrong trailing dim
    with pytest.raises(ValueError, match="expects"):
        enc.fit(np.zeros((10, 8, 1)))  # not 2-D


# ───────────────────────── Forward pass ─────────────────────────


def test_encoder_forward_shape() -> None:
    enc = PCAEncodingLayer(in_features=8, n_components=4)
    enc.fit(np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32))
    x = torch.randn(2, 5, 8)
    z = enc(x)
    assert z.shape == (2, 5, 4)


def test_encoder_forward_no_gradient_on_buffers() -> None:
    """Gradient flow through forward must not touch components or mean."""
    enc = PCAEncodingLayer(in_features=8, n_components=4)
    enc.fit(np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32))
    x = torch.randn(2, 5, 8, requires_grad=True)
    z = enc(x)
    z.sum().backward()
    # No params, so optimizer can't change anything; verify the buffers
    # are not in the gradient graph.
    assert not enc.components.requires_grad
    assert not enc.mean.requires_grad
    assert x.grad is not None  # input gradient still flows


def test_encoder_rejects_wrong_trailing_dim() -> None:
    enc = PCAEncodingLayer(in_features=8, n_components=4)
    with pytest.raises(ValueError, match="trailing axis"):
        enc(torch.zeros(2, 5, 6))


# ───────────────────────── Decoder / round-trip ─────────────────────────


def test_decoder_load_from_copies_buffers() -> None:
    enc = PCAEncodingLayer(in_features=8, n_components=4)
    enc.fit(np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32))
    dec = PCADecodingLayer(in_features=8, n_components=4)
    dec.load_from(enc)
    assert torch.equal(dec.components, enc.components)
    assert torch.equal(dec.mean, enc.mean)
    assert dec.is_fitted


def test_decoder_round_trip_recovers_low_rank_data() -> None:
    """For data lying in the PCA subspace, encode→decode is exact.

    Construct rank-4 data (8-D vectors in a 4-D subspace) and verify
    PCA encoder→decoder round-trips with negligible error.
    """
    rng = np.random.default_rng(0)
    # 4-D latent → 8-D observed via random orthonormal basis.
    basis = np.linalg.qr(rng.standard_normal((8, 8)))[0][:, :4]  # (8, 4)
    latent = rng.standard_normal((200, 4)).astype(np.float32)
    data = (latent @ basis.T).astype(np.float32)  # (200, 8), rank 4
    enc, dec = fit_pca_encoder_decoder(data, n_components=4)
    x = torch.from_numpy(data[:32])
    z = enc(x)
    x_hat = dec(z)
    np.testing.assert_allclose(x.numpy(), x_hat.numpy(), atol=1e-4)


def test_decoder_load_from_mismatched_shape_raises() -> None:
    enc = PCAEncodingLayer(in_features=8, n_components=4)
    dec = PCADecodingLayer(in_features=8, n_components=2)
    with pytest.raises(ValueError, match="n_components"):
        dec.load_from(enc)
    dec2 = PCADecodingLayer(in_features=6, n_components=4)
    with pytest.raises(ValueError, match="in_features"):
        dec2.load_from(enc)


# ───────────────────────── fit_pca_encoder_decoder convenience ─────────────────────────


def test_fit_helper_returns_paired_modules() -> None:
    data = np.random.default_rng(0).standard_normal((100, 8)).astype(np.float32)
    enc, dec = fit_pca_encoder_decoder(data, n_components=3)
    assert enc.is_fitted and dec.is_fitted
    assert enc.out_features == 3
    assert dec.out_features == 8


def test_fit_helper_accepts_torch_tensor() -> None:
    data = torch.randn(100, 8)
    enc, _dec = fit_pca_encoder_decoder(data, n_components=3)
    assert enc.is_fitted


def test_fit_helper_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        fit_pca_encoder_decoder(np.zeros((10, 8, 2)), n_components=3)


# ───────────────────────── PCAEncoder (registry-facing wrapper) ─────────────────────────


def test_pca_encoder_registered_under_pca_name() -> None:
    """``build_encoder('PCA', cfg)`` works through the encoder registry."""
    from neural_data_decoding.models.layers.pca import PCAEncoder
    from neural_data_decoding.models.registry import build_encoder, list_encoders
    assert "PCA" in list_encoders()
    enc = build_encoder("PCA", {"in_features": 8, "n_components": 4})
    assert isinstance(enc, PCAEncoder)
    assert enc.out_features == 4


def test_pca_encoder_forward_raises_before_fit() -> None:
    """PCAEncoder.forward must error if called before fit."""
    from neural_data_decoding.models.layers.pca import PCAEncoder
    enc = PCAEncoder(in_features=8, n_components=4)
    with pytest.raises(RuntimeError, match="before .fit"):
        enc(torch.zeros(2, 5, 8))


def test_pca_encoder_fit_then_forward() -> None:
    """After fit, PCAEncoder forwards 3-D (B, W, in) → 3-D (B, W, n_components)."""
    from neural_data_decoding.models.layers.pca import PCAEncoder
    enc = PCAEncoder(in_features=8, n_components=4)
    enc.fit(np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32))
    y = enc(torch.randn(2, 5, 8))
    assert y.shape == (2, 5, 4)


def test_pca_encoder_no_parameters_only_buffers() -> None:
    """PCAEncoder must have no learnable params (PCA is frozen)."""
    from neural_data_decoding.models.layers.pca import PCAEncoder
    enc = PCAEncoder(in_features=8, n_components=4)
    assert sum(1 for _ in enc.parameters()) == 0
    buffer_names = {n for n, _ in enc.named_buffers()}
    assert "pca.components" in buffer_names and "pca.mean" in buffer_names
