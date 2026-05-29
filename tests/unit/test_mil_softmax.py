"""Unit tests for :class:`MILSoftmaxLayer` (multi-axis softmax)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neural_data_decoding.models.layers.mil_softmax import MILSoftmaxLayer


def test_joint_softmax_sums_to_one_over_softmax_axes() -> None:
    """For each non-softmax index, the softmaxed values sum to 1."""
    layer = MILSoftmaxLayer(dims=(0, 2))  # softmax over (C, T) of (C, B, T)
    x = torch.randn(3, 2, 4)
    z = layer(x)
    # Per batch element, the joint (C, T) distribution sums to 1.
    per_batch = z.sum(dim=(0, 2))
    assert torch.allclose(per_batch, torch.ones(2), atol=1e-6)


def test_single_axis_matches_standard_softmax() -> None:
    """A joint softmax over a single axis is the same as F.softmax on that axis."""
    layer = MILSoftmaxLayer(dims=(2,))
    x = torch.randn(2, 3, 4)
    expected = F.softmax(x, dim=2)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_output_shape_matches_input() -> None:
    """The layer preserves tensor shape (softmax is shape-invariant)."""
    layer = MILSoftmaxLayer(dims=(0, 2))
    x = torch.randn(3, 5, 4)
    assert layer(x).shape == x.shape


def test_from_formats_picks_matching_axes() -> None:
    """``from_formats`` maps format tags to numeric axes (find+ismember)."""
    layer = MILSoftmaxLayer.from_formats(
        softmax_format="SCT", tensor_format="CBT"
    )
    assert layer.dims == (0, 2)


def test_from_formats_case_insensitive() -> None:
    """Tag matching is case-insensitive on both arguments."""
    layer = MILSoftmaxLayer.from_formats(
        softmax_format="sct", tensor_format="cbt"
    )
    assert layer.dims == (0, 2)


def test_from_formats_ssctb_full() -> None:
    """SSCTB / 'SCT' picks every non-batch axis."""
    layer = MILSoftmaxLayer.from_formats(
        softmax_format="SCT", tensor_format="SSCTB"
    )
    assert layer.dims == (0, 1, 2, 3)


def test_from_formats_no_match_raises() -> None:
    """When no axis matches, fail loud rather than degenerate to identity."""
    with pytest.raises(ValueError, match="No axis"):
        MILSoftmaxLayer.from_formats(
            softmax_format="X", tensor_format="CBT"
        )


def test_empty_dims_rejected() -> None:
    """Constructing with no dims is a programming error."""
    with pytest.raises(ValueError, match="non-empty"):
        MILSoftmaxLayer(dims=())


def test_gradients_flow_through_layer() -> None:
    """A backward through the layer reaches the input.

    Note: ``z.sum()`` over a joint-softmax output is a constant (the batch
    size), so its gradient is zero — that's the math, not a bug. Use a
    single-element output instead so the upstream gradient is non-trivial.
    """
    layer = MILSoftmaxLayer(dims=(0, 2))
    x = torch.randn(3, 2, 4, requires_grad=True)
    z = layer(x)
    z[0, 0, 0].backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_output_is_finite_for_large_inputs() -> None:
    """Large logits don't blow up (the max-subtraction trick keeps exp finite)."""
    layer = MILSoftmaxLayer(dims=(0,))
    x = torch.tensor([[1000.0, 1001.0, 999.0]])  # would overflow without max-shift
    z = layer(x)
    assert torch.isfinite(z).all()
    assert torch.allclose(z.sum(dim=0), torch.ones(3), atol=1e-6)
