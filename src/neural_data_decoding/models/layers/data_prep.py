"""Per-window flatten/unflatten layers for the 5D ``(B, W, T, A, C)`` trial layout.

Pipeline shape contract
-----------------------
Every trial in the data pipeline has shape ``(W, T, A, C)``:

* **W** — number of windows (the GRU/recurrent sequence axis).
* **T** — samples per window (within-window time; MATLAB
  ``InputSize(2)``).
* **A** — number of areas (probes); MATLAB ``InputSize(3)``.
* **C** — channels per area; MATLAB ``InputSize(1)``.

Batched: ``(B, W, T, A, C)``.

The GRU/LSTM/Feedforward encoders consume a per-window 1-D feature
vector, so the composite flattens the within-window dims via
:class:`FlattenPerWindow` before the encoder and unflattens the
decoder's reconstruction back to ``(W, T, A, C)`` via
:class:`UnflattenPerWindow` so the reconstruction loss compares against
the original 5D target element-wise.

The convolutional encoder (CC #3 Phase 2, future) operates BEFORE the
flatten — it sees ``(B, W, T, A, C)`` and produces per-window features
by 2-D conv over the ``(T, C)`` plane with ``A`` as the conv channel
axis (mirroring MATLAB's ``[1, n]`` 2-D kernels per
``cgg_constructConvolutionalCoder.m``).

Singleton dims (``T=1``, ``A=1``) are valid — that case is degenerate
1-D conv along ``C`` only, equivalent to the Feedforward bridge.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FlattenPerWindow(nn.Module):
    """Collapse the within-window dims into a single feature axis.

    Maps ``(B, W, T, A, C) → (B, W, T * A * C)``. Used to feed
    per-window data into the GRU/LSTM/Feedforward encoder which expects
    a 1-D feature vector per sequence step.

    The flatten order is ``(T, A, C)`` raster-major (PyTorch's default
    ``reshape``); the matching :class:`UnflattenPerWindow` reverses it
    exactly so the decoder's output projects cleanly back to the 5D
    target.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, W, T, A, C)`` → ``(B, W, T * A * C)``.

        Accepts both the canonical 5-D layout and a 3-D fallback
        ``(B, W, F)`` (treated as the singleton case ``T = A = 1``,
        ``C = F``) for backwards compatibility with callers that built
        raw tensors before the data-restructure landed.

        Parameters
        ----------
        x
            Tensor with shape ``(B, W, T, A, C)`` (5-D, canonical) or
            ``(B, W, F)`` (3-D, backwards-compat passthrough).

        Returns
        -------
        torch.Tensor
            3-D tensor ``(B, W, F)`` where ``F = T * A * C`` for 5-D
            input, or the same tensor for 3-D input.

        Raises
        ------
        ValueError
            If ``x.ndim`` is neither 3 nor 5.
        """
        if x.ndim == 3:
            return x
        if x.ndim != 5:
            raise ValueError(
                f"FlattenPerWindow expects a 5-D input (B, W, T, A, C) or "
                f"a 3-D fallback (B, W, F); got shape {tuple(x.shape)}.",
            )
        b, w = x.shape[0], x.shape[1]
        return x.reshape(b, w, -1)


class UnflattenPerWindow(nn.Module):
    """Restore the within-window dims after the decoder.

    Maps ``(B, W, T * A * C) → (B, W, T, A, C)``. Reverses
    :class:`FlattenPerWindow` exactly so the reconstruction loss can
    compare against the original 5-D target.

    Parameters
    ----------
    t
        Samples per window (within-window time).
    a
        Number of areas.
    c
        Channels per area.
    """

    def __init__(self, *, t: int, a: int, c: int) -> None:
        super().__init__()
        if t < 1 or a < 1 or c < 1:
            raise ValueError(
                f"UnflattenPerWindow requires positive t/a/c; got t={t}, a={a}, c={c}.",
            )
        self.t = t
        self.a = a
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, W, T * A * C)`` → ``(B, W, T, A, C)``.

        When the builder was configured with ``t = a = 1`` (the
        backwards-compat singleton case), this is the identity on a
        ``(B, W, C)`` tensor — useful so callers can pass 3-D tensors
        through the composite without explicit reshape.

        Parameters
        ----------
        x
            3-D tensor whose last axis size must equal ``t * a * c``.

        Returns
        -------
        torch.Tensor
            5-D tensor ``(B, W, T, A, C)``, or the input unchanged when
            ``t = a = 1`` (in which case the 5-D shape would be
            ``(B, W, 1, 1, C)`` and we return ``(B, W, C)`` instead so
            tests checking 3-D shapes still work).

        Raises
        ------
        ValueError
            If ``x.ndim != 3`` or the trailing axis doesn't match
            ``t * a * c``.
        """
        if x.ndim != 3:
            raise ValueError(
                f"UnflattenPerWindow expects a 3-D input (B, W, F); "
                f"got shape {tuple(x.shape)}.",
            )
        # Singleton case (T = A = 1): passthrough — the 5-D shape would
        # be (B, W, 1, 1, C) and we'd rather return (B, W, C) so existing
        # 3-D callers see no shape change. Skip the trailing-axis
        # validation entirely (`c` is a placeholder in this mode).
        if self.t == 1 and self.a == 1:
            return x
        expected = self.t * self.a * self.c
        if x.shape[-1] != expected:
            raise ValueError(
                f"UnflattenPerWindow trailing axis ({x.shape[-1]}) must equal "
                f"t * a * c = {self.t} * {self.a} * {self.c} = {expected}.",
            )
        b, w = x.shape[0], x.shape[1]
        return x.reshape(b, w, self.t, self.a, self.c)


__all__ = [
    "FlattenPerWindow",
    "UnflattenPerWindow",
]
