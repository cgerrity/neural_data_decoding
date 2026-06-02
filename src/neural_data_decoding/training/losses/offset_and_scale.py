"""Learnable offset/scale augmentation loss — port of ``cgg_lossOffsetAndScale.m``.

The MATLAB pipeline supports a learnable augmentation pathway: a
decoder branch outputs per-trial-window estimates of a scale factor
and an offset, and the loss compares these to **targets derived from
the input data itself**:

* ``T_Scale = range(X, spatial_axis) - 1`` (for the default
  ``AugmentEquation='mX+b+X'``).
* ``T_Offset = median(X, spatial_axis)``.

The loss is ``0.5 * l2loss(Y_Scale, T_Scale, Mask=Mask_NaN) +
0.5 * l2loss(Y_Offset, T_Offset, Mask=Mask_NaN)`` where
``Mask_NaN = ~any(isnan(X), spatial_axis)`` flags any spatial slice
containing a removed channel.

Auto-activation (Critical Note #32)
-----------------------------------
The MATLAB loss orchestrator (``cgg_lossComponents.m`` lines 368-375)
only invokes this loss when the decoder's layer graph contains layers
named ``reshape_offset_Augment`` or ``reshape_scale_Augment`` AND
``WeightOffsetAndScale`` is not NaN. The Python equivalent inspects
the decoder for the analogous module
(:class:`~neural_data_decoding.models.layers.offset_scale.LearnableOffsetScale`)
and skips the loss term entirely otherwise.

For our 5-D ``(B, W, T, A, C)`` data layout the analog of MATLAB's
``SpatialDimensions(end)`` is the per-area channel axis ``C``: the
augmentation targets reduce across ``C`` per ``(B, W, T, A)`` slot.
"""

from __future__ import annotations

from typing import Literal

import torch


_AugmentEquation = Literal["mX+b+X", "m(X+b)"]


def offset_and_scale_targets(
    x: torch.Tensor,
    *,
    augment_equation: _AugmentEquation = "mX+b+X",
    spatial_dim: int = -1,
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``T_Scale`` and ``T_Offset`` directly from input ``x``.

    Mirrors ``cgg_lossOffsetAndScale.m`` lines 40-50.

    Parameters
    ----------
    x
        Input tensor (the encoder's input, NOT NaN-zeroed — NaNs are
        meaningful here for the mask).
    augment_equation
        ``'mX+b+X'`` (default; MATLAB ``cgg_lossOffsetAndScale.m`` line
        24) gives scale = range(x) - 1, offset = median(x).
        ``'m(X+b)'`` gives scale = range(x), offset = median(x) /
        (scale + epsilon).
    spatial_dim
        Axis to reduce over (the per-area channel axis ``C`` for our
        canonical 5-D layout). Defaults to ``-1``.
    epsilon
        Numerical stability term for the ``'m(X+b)'`` divisor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(T_Scale, T_Offset)``, both with ``spatial_dim`` removed
        (i.e. reduced shape).
    """
    # range = max - min along the channel axis. We need NaN-aware
    # max/min — but the MATLAB code doesn't explicitly mask before
    # reduction; the mask is applied AFTER, on the loss. So we mirror
    # that: take naive max/min/median across C. The mask handles the
    # NaN positions by excluding them from the loss.
    # nanmax/nanmin/nanmedian avoid NaN-propagation if any NaNs slip in.
    x_max = torch.amax(x, dim=spatial_dim)
    x_min = torch.amin(x, dim=spatial_dim)
    x_range = x_max - x_min
    # MATLAB ``median`` averages the two middle values for even-length
    # vectors; PyTorch's ``torch.median`` returns the lower middle. Use
    # ``torch.quantile(0.5)`` for MATLAB parity.
    x_median = torch.quantile(x, 0.5, dim=spatial_dim)

    if augment_equation == "m(X+b)":
        t_scale = x_range
        t_offset = x_median / (t_scale + epsilon)
    else:  # 'mX+b+X' or otherwise
        t_scale = x_range - 1.0
        t_offset = x_median
    return t_scale, t_offset


def offset_and_scale_loss(
    x: torch.Tensor,
    y_scale: torch.Tensor,
    y_offset: torch.Tensor,
    *,
    augment_equation: _AugmentEquation = "mX+b+X",
    spatial_dim: int = -1,
) -> torch.Tensor:
    """The augmentation auxiliary loss.

    Computes ``0.5 * l2loss(Y_Scale, T_Scale, Mask=Mask_NaN) + 0.5 *
    l2loss(Y_Offset, T_Offset, Mask=Mask_NaN)`` per
    ``cgg_lossOffsetAndScale.m`` lines 73, 95, 98.

    Both ``l2loss`` calls use MATLAB's batch-size normalization
    semantics (Critical Note #38) — ``0.5 * Σ(mask · (y - t)²) /
    batch_size``. ``batch_size`` here is the size of the
    ``(b, w, t, a)`` axes after reducing over ``spatial_dim``.

    Parameters
    ----------
    x
        The original input (may contain NaN at removed-channel
        positions). Used both for computing the targets and the
        NaN mask.
    y_scale, y_offset
        Decoder branch outputs; must have the same shape as the
        targets (``x`` reduced over ``spatial_dim``).
    augment_equation, spatial_dim
        See :func:`offset_and_scale_targets`.

    Returns
    -------
    torch.Tensor
        Scalar (0-D) loss, differentiable w.r.t. ``y_scale`` and
        ``y_offset``.

    Raises
    ------
    ValueError
        If ``y_scale`` or ``y_offset`` shape doesn't match the
        reduced ``x`` shape.
    """
    t_scale, t_offset = offset_and_scale_targets(
        x, augment_equation=augment_equation, spatial_dim=spatial_dim,
    )
    if y_scale.shape != t_scale.shape:
        raise ValueError(
            f"y_scale shape {tuple(y_scale.shape)} does not match the "
            f"target shape {tuple(t_scale.shape)} (input reduced over "
            f"spatial_dim={spatial_dim}).",
        )
    if y_offset.shape != t_offset.shape:
        raise ValueError(
            f"y_offset shape {tuple(y_offset.shape)} does not match the "
            f"target shape {tuple(t_offset.shape)}.",
        )
    # Mask: True where NO NaN appears in the spatial slice (matches
    # MATLAB's ~any(isnan(X), SpatialDimensions(end))).
    mask = ~torch.any(torch.isnan(x), dim=spatial_dim)
    # Replace NaN in targets (from NaN-containing slices) with 0 so the
    # subtraction is well-defined; the mask zeros their contribution.
    t_scale = torch.where(mask, t_scale, torch.zeros_like(t_scale))
    t_offset = torch.where(mask, t_offset, torch.zeros_like(t_offset))
    diff_scale = torch.where(
        mask, y_scale - t_scale, torch.zeros_like(y_scale),
    )
    diff_offset = torch.where(
        mask, y_offset - t_offset, torch.zeros_like(y_offset),
    )
    # batch_size = the size of axis 0 of the input (matches the
    # MATLAB l2loss convention from Critical Note #38).
    batch_size = x.shape[0]
    loss_scale = 0.5 * (diff_scale**2).sum() / batch_size
    loss_offset = 0.5 * (diff_offset**2).sum() / batch_size
    return loss_scale + loss_offset


__all__ = [
    "offset_and_scale_loss",
    "offset_and_scale_targets",
]
