"""Tests for the spatial-divisor and crop-pyramid utilities (CC #3 Phase 2)."""

from __future__ import annotations

import pytest

from neural_data_decoding.models.stitching_fusion.divisors import (
    find_optimal_divisors,
    get_crop_amounts,
)


# ───────────────────────── find_optimal_divisors ─────────────────────────


def test_find_optimal_divisors_docstring_example() -> None:
    """Docstring example: ``find_optimal_divisors(120, 10, 15)``."""
    x, y = find_optimal_divisors(120, 10, 15)
    # Product must divide 120; want largest product, tie-broken by ratio.
    assert 120 % (x * y) == 0
    assert x <= 10 and y <= 15
    # Largest product = 120 (the input itself); 10*12=120 satisfies all.
    assert x * y == 120


def test_find_optimal_divisors_trivial_when_no_pair_fits() -> None:
    """If only x=y=1 satisfies the divisibility, return (1, 1)."""
    # 7 is prime; only divisors are 1 and 7. With b=3, c=3, only x*y=1 works.
    x, y = find_optimal_divisors(7, 3, 3)
    assert (x, y) == (1, 1)


def test_find_optimal_divisors_prefers_balanced_ratio_on_tie() -> None:
    """When two pairs tie on product, the one closer to b/c wins."""
    # a=12, b=6, c=6 → target ratio 1.0
    # Candidates with product 12: (2,6) ratio=0.33, (3,4) ratio=0.75,
    #                              (4,3) ratio=1.33, (6,2) ratio=3.0
    # Closest to 1.0 is (3,4) at |0.75-1.0|=0.25 (vs (4,3) at 0.33).
    x, y = find_optimal_divisors(12, 6, 6)
    assert (x, y) == (3, 4)


def test_find_optimal_divisors_respects_bounds() -> None:
    """x and y never exceed their bounds even if a larger product is divisible."""
    # a=24 has divisor pairs (1,24), (2,12), (3,8), (4,6); bound b=3 caps x.
    x, y = find_optimal_divisors(24, 3, 8)
    assert x <= 3 and y <= 8
    # Best legal pair: (3, 8) product 24.
    assert (x, y) == (3, 8)


def test_find_optimal_divisors_negative_inputs_raise() -> None:
    with pytest.raises(ValueError, match="positive"):
        find_optimal_divisors(0, 10, 10)
    with pytest.raises(ValueError, match="positive"):
        find_optimal_divisors(10, 0, 10)


# ───────────────────────── get_crop_amounts ─────────────────────────


def test_get_crop_amounts_zero_layers_is_identity() -> None:
    """With num_layers=0, upsample stack is just the input; no crops."""
    crops, ups = get_crop_amounts((8, 6), stride=2, num_layers=0)
    assert ups == [(8, 6)]
    assert crops == []


def test_get_crop_amounts_one_layer_even_dims() -> None:
    """Stride-2 over (8, 6) for 1 layer: downsample to (4, 3); no crop needed.

    ``upsample_sizes`` is bottom-up (smallest first); ``crop_sizes`` is the
    reversed bottom-up sequence (matches MATLAB's final ``flipud``).
    """
    crops, ups = get_crop_amounts((8, 6), stride=2, num_layers=1)
    # downsample: (8,6) → (4,3) ; upsample bottom-up: [(4,3), (8,6)]
    assert ups == [(4, 3), (8, 6)]
    # (4,3)*2 - (8,6) = (0,0)
    assert crops == [(0, 0)]


def test_get_crop_amounts_one_layer_odd_dims_needs_crop() -> None:
    """Odd input dim → ceil makes upsample land above the original → crop != 0."""
    # (7, 5) with stride 2 → ceil(7/2)=4, ceil(5/2)=3 → (4,3)
    # Upsample bottom-up: [(4,3), (7,5)]
    # crop[0] = (4,3)*2 - (7,5) = (8,6) - (7,5) = (1,1)
    crops, ups = get_crop_amounts((7, 5), stride=2, num_layers=1)
    assert ups == [(4, 3), (7, 5)]
    assert crops == [(1, 1)]


def test_get_crop_amounts_two_layers_pyramid_even() -> None:
    """Two-layer pyramid with stride 2 over (8, 8) — perfect divides, no crops."""
    crops, ups = get_crop_amounts((8, 8), stride=2, num_layers=2)
    # downsample: (8,8) → (4,4) → (2,2)
    # upsample bottom-up: [(2,2), (4,4), (8,8)]
    assert ups == [(2, 2), (4, 4), (8, 8)]
    # crop[0]=(2,2)*2-(4,4)=(0,0); crop[1]=(4,4)*2-(8,8)=(0,0); reversed → [(0,0),(0,0)]
    assert crops == [(0, 0), (0, 0)]


def test_get_crop_amounts_two_layers_asymmetric_distinguishes_ordering() -> None:
    """Two-layer (7, 5) — non-trivial crops at both layers, distinguishes order.

    Pre-reversal crops (in upsample-bottom-up order):
      crop[0] = (2,2)*2 - (4,3) = (0,1)  -- the innermost upsample step
      crop[1] = (4,3)*2 - (7,5) = (1,1)  -- the outermost upsample step

    After ``flipud`` the outermost crop is first, matching MATLAB's
    convention where ``CropSizes{1}`` is applied at the topmost upsample.
    """
    crops, ups = get_crop_amounts((7, 5), stride=2, num_layers=2)
    assert ups == [(2, 2), (4, 3), (7, 5)]
    assert crops == [(1, 1), (0, 1)]


def test_get_crop_amounts_stride_must_be_positive() -> None:
    with pytest.raises(ValueError, match="stride"):
        get_crop_amounts((4, 4), stride=0, num_layers=1)


def test_get_crop_amounts_negative_layers_raises() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        get_crop_amounts((4, 4), stride=2, num_layers=-1)
