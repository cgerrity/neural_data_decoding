"""Learnable offset/scale decoder block — port of MATLAB's augmentation head.

The MATLAB pipeline includes an optional decoder branch (built via
``cgg_generateAugmentBlock.m``) that outputs per-window estimates of a
scale factor and an offset matching the input's statistics
(``range(X) - 1`` and ``median(X)`` for the default
``AugmentEquation='mX+b+X'``). The loss
(``cgg_lossOffsetAndScale.m``, ported as
:func:`~neural_data_decoding.training.losses.offset_and_scale.offset_and_scale_loss`)
matches these outputs to the targets derived from the input.

Critical Note #32 auto-activation: the MATLAB loss orchestrator
inspects the decoder's layer graph for the named augmentation layers
(``reshape_offset_Augment`` / ``reshape_scale_Augment``). The Python
equivalent checks the decoder module tree for an instance of
:class:`LearnableOffsetScale` via ``isinstance`` so the loss only
fires when the topology is present (no separate config flag needed
beyond placing the module in the decoder).

Pythonic design
---------------
The MATLAB block is built out of fully-connected layers (~250 hidden,
per the default) plus a final reshape into the input-matching shape.
Our Pythonic port wraps two parallel ``nn.Linear`` heads (one for
scale, one for offset) projecting from the latent ``z`` into per-
window-reduced statistics. The reduced shape is ``(B, W, T, A)`` after
collapsing the per-area channel axis ``C``; the heads operate
per-window with a Linear producing ``T * A`` outputs, reshaped to
``(B, W, T, A)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnableOffsetScale(nn.Module):
    """Decoder-side augmentation head producing ``(Y_Scale, Y_Offset)``.

    Two parallel ``nn.Linear`` heads project from the latent ``z``
    (shape ``(B, W, latent)``) to the per-window-reduced output shape
    ``(B, W, T, A)`` — matching the targets the augmentation loss
    derives from input ``x`` reduced over the per-area channel axis
    ``C``.

    Auto-activation invariant: place this module anywhere in the
    decoder; the loss orchestrator detects it via ``isinstance`` and
    invokes the loss term. Without this module in the decoder tree,
    the augmentation loss term is a no-op (returns ``None`` from the
    composite forward).

    Parameters
    ----------
    latent_dim
        Trailing-axis size of the input ``z`` tensor (the bottleneck
        / sampling output).
    samples_per_window
        ``T`` — within-window time samples in the data layout.
    num_areas
        ``A`` — number of areas in the data layout.
    hidden_dim
        Optional hidden size for the FC heads. Defaults to
        ``samples_per_window * num_areas`` (mirroring MATLAB's
        ``HiddenSizeAugment=250`` default, but sized to the actual
        output instead of a fixed 250). Pass an explicit value to
        match a specific MATLAB configuration.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        samples_per_window: int,
        num_areas: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or samples_per_window < 1 or num_areas < 1:
            raise ValueError(
                "latent_dim, samples_per_window, and num_areas must all "
                f"be >= 1; got {latent_dim}, {samples_per_window}, {num_areas}.",
            )
        self.latent_dim = latent_dim
        self.samples_per_window = samples_per_window
        self.num_areas = num_areas
        out_dim = samples_per_window * num_areas
        h = hidden_dim if hidden_dim is not None else out_dim
        self.scale_head = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(
        self, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, W, latent_dim) → ((B, W, T, A), (B, W, T, A))``.

        Returns the ``(scale, offset)`` per-window-reduced estimates
        matching the augmentation-loss target shape.
        """
        if z.ndim != 3:
            raise ValueError(
                f"LearnableOffsetScale expects 3-D z (B, W, latent); "
                f"got shape {tuple(z.shape)}.",
            )
        b, w, _ = z.shape
        scale = self.scale_head(z).reshape(
            b, w, self.samples_per_window, self.num_areas,
        )
        offset = self.offset_head(z).reshape(
            b, w, self.samples_per_window, self.num_areas,
        )
        return scale, offset


def find_learnable_offset_scale(
    module: nn.Module,
) -> LearnableOffsetScale | None:
    """Walk a module tree and return the first :class:`LearnableOffsetScale`.

    Mirrors the MATLAB pattern of inspecting the decoder's layer graph
    for the auto-activation layers (Critical Note #32). Returns
    ``None`` when the module tree contains no augmentation head.
    """
    for child in module.modules():
        if isinstance(child, LearnableOffsetScale):
            return child
    return None


__all__ = [
    "LearnableOffsetScale",
    "find_learnable_offset_scale",
]
