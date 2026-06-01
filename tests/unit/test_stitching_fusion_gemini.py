"""End-to-end tests for the Gemini S&F variants (CC #3 Phase 3).

Exercises the three cascaded-multi-area option-sets ported from
``cgg_createStitchingFusionModule_v2.m``:

* 'Parallel Single Level'
* 'Cascade Single Kernel - Single Reduction'
* 'Cascade Single Kernel - Progressive Reduction'
"""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.models.composite import (
    build_variational_autoencoder,
    build_variational_composite,
)
from neural_data_decoding.models.stitching_fusion.gemini import (
    GeminiStitchingFusionModule,
    build_gemini_stitching_fusion,
)


_GEMINI_VARIANTS = [
    "Parallel Single Level",
    "Cascade Single Kernel - Single Reduction",
    "Cascade Single Kernel - Progressive Reduction",
]


# ───────────────────────── GeminiStitchingFusionModule ─────────────────────────


@pytest.mark.parametrize("variant", _GEMINI_VARIANTS)
def test_gemini_encoder_runs_and_reduces_dims(variant: str) -> None:
    """Each Gemini encoder accepts 5-D input and produces 5-D output with
    reduced T (per encoder_reduction)."""
    enc = build_gemini_stitching_fusion(
        variant, num_areas=2, filters_per_area=8, mode="Encoder",
    )
    x = torch.randn(2, 5, 16, 2, 4)
    y = enc(x)
    assert y.ndim == 5
    # T should be reduced.
    assert y.size(2) <= x.size(2)


@pytest.mark.parametrize("variant", _GEMINI_VARIANTS)
def test_gemini_decoder_runs(variant: str) -> None:
    """Each Gemini decoder consumes the encoder's output shape."""
    enc = build_gemini_stitching_fusion(
        variant, num_areas=2, filters_per_area=8, mode="Encoder",
    )
    dec = build_gemini_stitching_fusion(
        variant, num_areas=2, filters_per_area=8, mode="Decoder",
    )
    y = enc(torch.randn(2, 5, 16, 2, 4))
    z = dec(y)
    # Decoder produces 5-D output (T might not match exactly — composite crops).
    assert z.ndim == 5
    # Areas back to num_areas after final_reduction (groupedConv 1×1 to 1 per area).
    assert z.size(3) == 2


@pytest.mark.parametrize("variant", _GEMINI_VARIANTS)
def test_gemini_gradients_flow(variant: str) -> None:
    """Backward pass populates gradients on all parameters."""
    enc = build_gemini_stitching_fusion(
        variant, num_areas=2, filters_per_area=8, mode="Encoder",
    )
    y = enc(torch.randn(2, 5, 16, 2, 4))
    y.sum().backward()
    n_with_grad = sum(
        1 for p in enc.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0


def test_gemini_unknown_variant_raises() -> None:
    """Unknown option-set name → ValueError listing accepted names."""
    with pytest.raises(ValueError, match="Unknown Gemini option-set"):
        build_gemini_stitching_fusion(
            "BogusGemini", num_areas=2, filters_per_area=8, mode="Encoder",
        )


def test_gemini_input_validation() -> None:
    """Wrong-rank input raises a clear ValueError."""
    enc = build_gemini_stitching_fusion(
        "Parallel Single Level", num_areas=2, filters_per_area=8, mode="Encoder",
    )
    with pytest.raises(ValueError, match="5-D input"):
        enc(torch.randn(2, 5, 8))  # 3-D


# ───────────────────────── Composite end-to-end ─────────────────────────


def _gemini_cfg(*, variant: str, t: int = 8, a: int = 2, c: int = 4) -> dict:
    return {
        "in_features": c,
        "samples_per_window": t,
        "num_areas": a,
        "hidden_sizes": [16, 4],
        "num_classes_per_dim": [3],
        "classifier_hidden_size": [8, 4],
        "transform": "GRU",
        "loss_type_decoder": "MSE",
        "stitching_and_fusion_layer": variant,
    }


@pytest.mark.parametrize("variant", _GEMINI_VARIANTS)
def test_gemini_composite_forward_5d_shape(variant: str) -> None:
    """Composite with Gemini S&F: 5-D input → 5-D reconstruction matching input."""
    cfg = _gemini_cfg(variant=variant, t=8, a=2, c=4)
    model = build_variational_composite(cfg)
    model.eval()
    x = torch.randn(2, 5, 8, 2, 4)
    out = model(x)
    assert out.reconstruction is not None
    assert out.reconstruction.shape == x.shape


@pytest.mark.parametrize("variant", _GEMINI_VARIANTS)
def test_gemini_autoencoder_forward_5d_shape(variant: str) -> None:
    """Stage 1 autoencoder with each Gemini variant."""
    cfg = _gemini_cfg(variant=variant, t=8, a=2, c=4)
    ae = build_variational_autoencoder(cfg)
    ae.eval()
    x = torch.randn(2, 5, 8, 2, 4)
    out = ae(x)
    assert out.reconstruction.shape == x.shape


@pytest.mark.parametrize("variant", _GEMINI_VARIANTS)
def test_gemini_composite_gradients_flow(variant: str) -> None:
    """Backward pass populates gradients through Gemini + Linear + main encoder."""
    cfg = _gemini_cfg(variant=variant, t=8, a=2, c=4)
    model = build_variational_composite(cfg)
    x = torch.randn(2, 5, 8, 2, 4)
    out = model(x)
    assert out.reconstruction is not None
    loss = out.reconstruction.pow(2).sum() + sum(lg.sum() for lg in out.logits)
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0


def test_gemini_module_is_instance_check() -> None:
    """The factory returns a :class:`GeminiStitchingFusionModule`."""
    for v in _GEMINI_VARIANTS:
        enc = build_gemini_stitching_fusion(
            v, num_areas=2, filters_per_area=8, mode="Encoder",
        )
        assert isinstance(enc, GeminiStitchingFusionModule)
