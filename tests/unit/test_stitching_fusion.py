"""Tests for the Stitching+Fusion bridges (Milestone CC #3 Phase 1)."""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.models.composite import (
    build_variational_autoencoder,
    build_variational_composite,
)
from neural_data_decoding.models.stitching_fusion import (
    FeedforwardStitchingFusion,
    build_stitching_fusion,
)


# ───────────────────────── FeedforwardStitchingFusion ─────────────────────────


def test_feedforward_sf_output_shape_matches_out_features() -> None:
    """Output channel dim equals out_features regardless of input dim."""
    m = FeedforwardStitchingFusion(in_features=8, out_features=12)
    x = torch.randn(3, 5, 8)
    y = m(x)
    assert y.shape == (3, 5, 12)


def test_feedforward_sf_per_timestep_independence() -> None:
    """Per-timestep linear: changing one timestep only affects that timestep."""
    m = FeedforwardStitchingFusion(in_features=4, out_features=4)
    x = torch.randn(2, 6, 4)
    y_base = m(x).detach().clone()
    x_perturb = x.clone()
    x_perturb[:, 2, :] += 100.0
    y_perturb = m(x_perturb).detach()
    same_axes = [t for t in range(6) if t != 2]
    for t in same_axes:
        assert torch.allclose(y_base[:, t, :], y_perturb[:, t, :])
    assert not torch.allclose(y_base[:, 2, :], y_perturb[:, 2, :])


def test_feedforward_sf_learnable_params_count() -> None:
    """Only a Linear → weights + bias."""
    m = FeedforwardStitchingFusion(in_features=8, out_features=16)
    p = list(m.parameters())
    assert len(p) == 2  # weight, bias
    assert p[0].shape == (16, 8)
    assert p[1].shape == (16,)


# ───────────────────────── build_stitching_fusion dispatcher ─────────────────────────


def test_dispatcher_feedforward_encoder_mode() -> None:
    """Encoder mode: in_features → cross_area_fusion_size."""
    m = build_stitching_fusion(
        "Feedforward",
        in_features=8, cross_area_fusion_size=32, mode="Encoder",
    )
    assert isinstance(m, FeedforwardStitchingFusion)
    assert m.in_features == 8
    assert m.out_features == 32


def test_dispatcher_feedforward_decoder_mode() -> None:
    """Decoder mode: cross_area_fusion_size → in_features (reverse direction)."""
    m = build_stitching_fusion(
        "Feedforward",
        in_features=8, cross_area_fusion_size=32, mode="Decoder",
    )
    assert isinstance(m, FeedforwardStitchingFusion)
    assert m.in_features == 32
    assert m.out_features == 8


@pytest.mark.parametrize(
    "variant",
    [
        "Default",
        "Parallel Single Level",
        "Cascade Single Kernel - Single Reduction",
        "Cascade Single Kernel - Progressive Reduction",
    ],
)
def test_dispatcher_pending_variants_raise_notimplemented(variant: str) -> None:
    """Phase 2/3 variants are explicitly pending."""
    with pytest.raises(NotImplementedError, match="pending"):
        build_stitching_fusion(
            variant,
            in_features=8, cross_area_fusion_size=32, mode="Encoder",
        )


def test_dispatcher_unknown_variant_raises() -> None:
    """Unknown S&F name → ValueError listing expected options."""
    with pytest.raises(ValueError, match="Unknown stitching_and_fusion_layer"):
        build_stitching_fusion(
            "BogusVariant",
            in_features=8, cross_area_fusion_size=32, mode="Encoder",
        )


# ───────────────────────── Composite wiring ─────────────────────────


def _base_cfg() -> dict:
    return {
        "in_features": 6,
        "hidden_sizes": [10, 4],  # encoder [10], latent 4 → cross_area_fusion_size = 20
        "num_classes_per_dim": [3],
        "classifier_hidden_size": [8, 4],
        "transform": "GRU",
        "loss_type_decoder": "MSE",
    }


def test_variational_composite_without_sf_has_no_bridges() -> None:
    """No cfg.stitching_and_fusion_layer → composite has None bridges."""
    composite = build_variational_composite(_base_cfg())
    assert composite.pre_encoder is None
    assert composite.post_decoder is None


def test_variational_composite_with_feedforward_sf_wires_bridges() -> None:
    """cfg.stitching_and_fusion_layer='Feedforward' → bridges with right shapes."""
    cfg = _base_cfg()
    cfg["stitching_and_fusion_layer"] = "Feedforward"
    composite = build_variational_composite(cfg)
    assert isinstance(composite.pre_encoder, FeedforwardStitchingFusion)
    assert isinstance(composite.post_decoder, FeedforwardStitchingFusion)
    # cross_area_fusion_size = hidden_sizes[0] * 2 = 10 * 2 = 20
    assert composite.pre_encoder.in_features == 6
    assert composite.pre_encoder.out_features == 20
    assert composite.post_decoder.in_features == 20
    assert composite.post_decoder.out_features == 6


def test_variational_composite_with_sf_forward_shape_is_in_features() -> None:
    """End-to-end forward with S&F still reconstructs to in_features."""
    cfg = _base_cfg()
    cfg["stitching_and_fusion_layer"] = "Feedforward"
    composite = build_variational_composite(cfg)
    composite.eval()
    x = torch.randn(2, 7, 6)
    out = composite(x)
    assert out.reconstruction is not None
    assert out.reconstruction.shape == (2, 7, 6)


def test_variational_composite_with_sf_gradients_flow_to_bridges() -> None:
    """Both pre_encoder and post_decoder bridges receive gradients."""
    cfg = _base_cfg()
    cfg["stitching_and_fusion_layer"] = "Feedforward"
    composite = build_variational_composite(cfg)
    composite.train()
    x = torch.randn(2, 7, 6)
    out = composite(x)
    assert out.reconstruction is not None
    loss = out.reconstruction.pow(2).sum()
    loss.backward()
    assert composite.pre_encoder is not None
    assert composite.post_decoder is not None
    for p in composite.pre_encoder.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    for p in composite.post_decoder.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_variational_autoencoder_with_sf_wires_bridges() -> None:
    """Stage 1 autoencoder respects the same S&F wiring."""
    cfg = _base_cfg()
    cfg["stitching_and_fusion_layer"] = "Feedforward"
    ae = build_variational_autoencoder(cfg)
    assert isinstance(ae.pre_encoder, FeedforwardStitchingFusion)
    assert isinstance(ae.post_decoder, FeedforwardStitchingFusion)
    ae.eval()
    x = torch.randn(2, 5, 6)
    out = ae(x)
    assert out.reconstruction.shape == (2, 5, 6)


def test_copy_autoencoder_weights_copies_sf_bridges() -> None:
    """Stage 1 → Stage 2 handoff copies pre/post bridge weights."""
    from neural_data_decoding.models.composite import copy_autoencoder_weights

    cfg = _base_cfg()
    cfg["stitching_and_fusion_layer"] = "Feedforward"
    src = build_variational_autoencoder(cfg)
    dst = build_variational_composite(cfg)

    # Mutate src bridge weights to a known value before copy.
    assert src.pre_encoder is not None
    with torch.no_grad():
        src.pre_encoder.linear.weight.fill_(0.5)
        src.pre_encoder.linear.bias.fill_(-0.25)
    copy_autoencoder_weights(src, dst)
    assert dst.pre_encoder is not None
    assert torch.equal(
        dst.pre_encoder.linear.weight, src.pre_encoder.linear.weight,
    )
    assert torch.equal(
        dst.pre_encoder.linear.bias, src.pre_encoder.linear.bias,
    )


def test_unknown_sf_type_in_cfg_raises_at_build_time() -> None:
    """Bad stitching_and_fusion_layer value fails at build, not at forward."""
    cfg = _base_cfg()
    cfg["stitching_and_fusion_layer"] = "BogusVariant"
    with pytest.raises(ValueError, match="Unknown stitching_and_fusion_layer"):
        build_variational_composite(cfg)
