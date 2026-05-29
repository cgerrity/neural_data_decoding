"""Multi-axis softmax for Multiple Instance Learning — port of ``cgg_softmaxLayer.m``.

Standard softmax normalizes over a single axis (the class axis). The MIL
formulation in this pipeline instead normalizes **jointly over several
axes simultaneously** — Space, Channel, and Time (``'SCT'``). The
probability mass sums to 1 over the whole (spatial × class × time) grid
per trial, so the layer does attention-over-instances (time, space) and
classification (channel) in one joint distribution. This is the MIL
"pooling": each (class, time[, space]) cell competes for probability mass.

Critical Note #10 — "MIL pooling is multi-axis softmax across
Space-Channel-Time; ``cgg_softmaxLayer.m`` computes softmax across these
axes simultaneously. Match this." This module reproduces that exactly.

MATLAB algorithm (``cgg_softmaxLayer.predict``)
-----------------------------------------------
1. ``dimsToOperate = find(ismember(dims(X), SoftmaxFormat))`` — the numeric
   indices of axes whose format tag is in the requested format string.
2. ``maxX = max(X, [], dimsToOperate)`` (detached for stability).
3. ``expX = exp(X - maxX)``.
4. ``sumExpX = sum(expX, dimsToOperate)``.
5. ``Z = expX ./ sumExpX``.

PyTorch carries no dlarray format strings, so the axes to operate over are
given explicitly (``dims``) or derived from a layout string via
:meth:`MILSoftmaxLayer.from_formats`, which mirrors the
``find(ismember(...))`` mapping.

Examples
--------
>>> import torch
>>> # Classifier output laid out (channel, batch, time) — softmax over (C, T).
>>> layer = MILSoftmaxLayer.from_formats(softmax_format="SCT", tensor_format="CBT")
>>> layer.dims
(0, 2)
>>> x = torch.randn(3, 2, 4)            # (C=3, B=2, T=4)
>>> z = layer(x)
>>> # Joint distribution over (C, T) per batch element sums to 1.
>>> bool(torch.allclose(z.sum(dim=(0, 2)), torch.ones(2), atol=1e-6))
True
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class MILSoftmaxLayer(nn.Module):
    """Softmax computed jointly over a set of axes (MIL pooling).

    Parameters
    ----------
    dims
        The axes to normalize over **together**. For a classifier output
        laid out ``(channel, batch, time)`` with MATLAB's ``'SCT'`` format,
        this is ``(0, 2)`` — channel and time. Use
        :meth:`from_formats` to derive these from layout strings.

    Attributes
    ----------
    dims : tuple of int
        The joint-softmax axes.
    """

    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        if not dims:
            raise ValueError("dims must be a non-empty sequence of axes.")
        self.dims = tuple(int(d) for d in dims)

    @classmethod
    def from_formats(
        cls, *, softmax_format: str, tensor_format: str
    ) -> "MILSoftmaxLayer":
        """Build a layer by matching format tags, mirroring MATLAB.

        Reproduces ``find(ismember(dims(X), SoftmaxFormat))`` (0-indexed):
        the operate-over axes are the positions in ``tensor_format`` whose
        tag appears in ``softmax_format``.

        Parameters
        ----------
        softmax_format
            The format tags to softmax over, e.g. ``"SCT"`` (Space,
            Channel, Time). Case-insensitive.
        tensor_format
            The layout of the tensor this layer will receive, e.g.
            ``"CBT"`` or ``"SSCTB"``. Case-insensitive.

        Returns
        -------
        MILSoftmaxLayer
            Configured to operate over the matching axes.

        Raises
        ------
        ValueError
            If no axis in ``tensor_format`` matches ``softmax_format``.
        """
        sf = softmax_format.upper()
        dims = [i for i, tag in enumerate(tensor_format.upper()) if tag in sf]
        if not dims:
            raise ValueError(
                f"No axis of tensor_format {tensor_format!r} matches "
                f"softmax_format {softmax_format!r}."
            )
        return cls(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the joint multi-axis softmax.

        Parameters
        ----------
        x
            Logits tensor. Softmax is computed jointly over ``self.dims``;
            all other axes are treated as independent (batch-like).

        Returns
        -------
        torch.Tensor
            Probabilities of the same shape as ``x``; values over
            ``self.dims`` sum to 1 for each fixed index of the remaining
            axes.
        """
        max_x = torch.amax(x, dim=self.dims, keepdim=True)
        exp_x = torch.exp(x - max_x)
        sum_exp = exp_x.sum(dim=self.dims, keepdim=True)
        return exp_x / sum_exp

    def extra_repr(self) -> str:
        """Show the joint-softmax axes in the module's repr."""
        return f"dims={self.dims}"


__all__ = ["MILSoftmaxLayer"]
