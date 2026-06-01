"""Spatial-dim helpers ported from MATLAB's ``cgg_findOptimalDivisors.m``
and ``cgg_getCropAmount.m`` — used by the Default and Gemini S&F variants
to partition a flat channel count into a 2D spatial grid and to compute
crop / upsample sizes through a strided pyramid.
"""

from __future__ import annotations

import math


def find_optimal_divisors(a: int, b: int, c: int) -> tuple[int, int]:
    """Find ``(x, y)`` such that ``x * y`` divides ``a`` evenly, with the
    largest possible product and the ``x/y`` ratio closest to ``b/c``.

    Port of ``cgg_findOptimalDivisors.m``. Used by
    ``cgg_constructStitchingAndFusionNetwork`` (the Gemini path, line 45)
    to pick the spatial grid size for a target channel count.

    Constraints
    -----------
    * ``x <= b``
    * ``y <= c``
    * ``a % (x * y) == 0``

    Optimization
    ------------
    1. Maximize ``x * y``.
    2. Tie-break by minimizing ``|x/y - b/c|``.

    Parameters
    ----------
    a
        Channel count to be partitioned. Must be positive.
    b
        Upper bound on the first divisor.
    c
        Upper bound on the second divisor.

    Returns
    -------
    tuple[int, int]
        ``(best_x, best_y)``. Both default to 1 when no non-trivial pair
        satisfies the constraints (matching the MATLAB initialization).

    Raises
    ------
    ValueError
        If any of ``a``, ``b``, ``c`` is not a positive integer.
    """
    if a < 1 or b < 1 or c < 1:
        raise ValueError(
            f"find_optimal_divisors requires positive a/b/c; got a={a}, b={b}, c={c}.",
        )

    best_x = 1
    best_y = 1
    max_product = 0
    min_ratio_diff = math.inf
    target_ratio = b / c

    for x in range(1, b + 1):
        if a % x != 0:
            continue
        max_remaining = a // x
        for y in range(1, c + 1):
            if max_remaining % y != 0:
                continue
            current_product = x * y
            current_diff = abs((x / y) - target_ratio)
            if current_product > max_product:
                max_product = current_product
                min_ratio_diff = current_diff
                best_x, best_y = x, y
            elif current_product == max_product and current_diff < min_ratio_diff:
                min_ratio_diff = current_diff
                best_x, best_y = x, y

    return best_x, best_y


def get_crop_amounts(
    spatial_sizes: tuple[int, ...], stride: int, num_layers: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Compute the per-layer crop amounts and upsample sizes for a strided
    pyramid.

    Port of ``cgg_getCropAmount.m``. Used by the convolutional S&F variants
    to size transposed-conv upsamplers so the decoder reconstructs to the
    original spatial dims exactly.

    For ``num_layers`` strided downsamplings of ``spatial_sizes`` by
    ``stride``, returns ``(crop_sizes, upsample_sizes)``:

    * ``upsample_sizes`` is the spatial-shape pyramid **bottom-up**
      (smallest first, original-shape last): ``upsample_sizes[0]`` is the
      most-reduced shape, ``upsample_sizes[-1]`` is ``spatial_sizes``.
      Matches MATLAB's ``flipud(DownSampleSizes)`` ordering.
    * ``crop_sizes`` are the per-layer corrections, also bottom-up after
      MATLAB's final ``flipud``. ``crop_sizes`` is computed pre-flip as
      ``upsample_sizes[i] * stride - upsample_sizes[i + 1]`` and then
      reversed.

    Parameters
    ----------
    spatial_sizes
        Original spatial dims, e.g. ``(H, W)``.
    stride
        Per-layer downsample stride; ``>= 1``.
    num_layers
        Number of strided layers.

    Returns
    -------
    tuple[list, list]
        ``(crop_sizes, upsample_sizes)``.

    Raises
    ------
    ValueError
        On ``stride < 1`` or ``num_layers < 0``.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}.")
    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0; got {num_layers}.")

    downsample_sizes: list[tuple[int, ...]] = [tuple(spatial_sizes)]
    for _ in range(num_layers):
        prev = downsample_sizes[-1]
        downsample_sizes.append(
            tuple(math.ceil(s / stride) for s in prev),
        )

    upsample_sizes = list(reversed(downsample_sizes))
    crop_sizes_bottom_up = [
        tuple(
            upsample_sizes[i][k] * stride - upsample_sizes[i + 1][k]
            for k in range(len(upsample_sizes[i]))
        )
        for i in range(num_layers)
    ]
    crop_sizes = list(reversed(crop_sizes_bottom_up))
    return crop_sizes, upsample_sizes


__all__ = [
    "find_optimal_divisors",
    "get_crop_amounts",
]
