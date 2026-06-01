"""Per-window 2-D convolutional encoder/decoder — port of
``cgg_constructConvolutionalCoder.m`` on the new 5-D ``(B, W, T, A, C)``
data path.

MATLAB ↔ Python mapping
-----------------------
MATLAB data layout per trial: ``(C, T, A, W, B)`` — Channel × WindowedTime
× Area × Window × Batch. Convolutional layers operate **per window** with
``[1, n]`` 2-D kernels: 1 along ``C``, ``n`` along ``T``. The conv treats
the ``(C, T)`` plane as the 2-D spatial axes and ``A`` as the conv's
channel axis (i.e. each conv filter sees all ``A`` areas at each spatial
location).

In PyTorch, with our data layout ``(B, W, T, A, C)``, the per-window
data is ``(T, A, C)``. We reshape to ``(B*W, A, C, T)`` to feed
:class:`torch.nn.Conv2d` (``(N, C_in, H, W)`` format), apply the conv
with kernel ``(1, kernel_t)``, and reshape the output back to
``(B, W, T_out, A_out, C)``.

WantSplitAreas
--------------
MATLAB's ``WantSplitAreas=true`` runs an independent conv per area (no
cross-area mixing). PyTorch maps this to grouped convolution with
``groups=A`` — each input area channel produces its own filter bank.

For ``WantSplitAreas=false`` the conv runs over all ``A`` channels
jointly, mixing across areas.

Block structure
---------------
Per :class:`_PerWindowConvBlock`:

1. Conv2d (or ConvTranspose2d for decoder) with kernel ``(1, kernel_t)``
   and stride ``(1, stride_t)`` along ``T``.
2. Optional dropout.
3. Optional activation (ReLU / Leaky ReLU / GeLU / Softplus-for-SoftSign).
4. Optional normalization (BatchNorm2d / InstanceNorm2d / GroupNorm).

Per :class:`_ConvLevel`:

* ``repetitions_per_block`` blocks (only the last carries the stride).
* Optional ResNet residual skip via a 1×1 conv to match shape, then
  add and apply a trailing activation.

Per :class:`PerWindowConvolutionalCoder`:

* Multi-level stack of ``_ConvLevel`` modules.
* Decoder mode reverses the level order and uses transposed convs.
* Optional time-axis cropping to recover the encoder's input ``T``
  exactly (mirrors MATLAB's ``cropLayer(CropAmount)``).

Scope
-----
This module implements the operations exercised by the
``'Default'`` S&F option-set plus enough flexibility for the
``'Convolutional'`` regular-encoder branch (future). Out of scope:
multi-filter cascading (``NumFilters > 1`` paths with combination
blocks), augment blocks (``WantLearnableOffset`` / ``WantLearnableScale``),
and the Gemini cascade variants (CC #3 Phase 3).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


def _activation_for(name: str) -> Optional[nn.Module]:
    """Map MATLAB activation strings to PyTorch modules."""
    n = name.strip()
    if n == "ReLU":
        return nn.ReLU()
    if n == "Leaky ReLU":
        return nn.LeakyReLU()
    if n == "SoftSign":
        # MATLAB's 'SoftSign' label actually instantiates a softplusLayer
        # (Critical Note #37 in PLAN.md).
        return nn.Softplus()
    if n == "GeLU":
        return nn.GELU()
    return None


def _normalization_for(want_normalization: bool | str, channels: int) -> Optional[nn.Module]:
    """Pick a 2-D normalization layer matching ``cgg_selectNormalizationLayer``."""
    if not want_normalization:
        return None
    if want_normalization is True or want_normalization == "Batch":
        return nn.BatchNorm2d(channels)
    if want_normalization == "Instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if want_normalization == "Layer":
        return nn.GroupNorm(1, channels)
    return None


class _PerWindowConvBlock(nn.Module):
    """One conv block applied per-window. Encoder or decoder mode.

    Operates on ``(B*W, A, C, T)`` tensors. Conv kernel is ``(1, kernel_t)``
    so it filters along ``T`` only — matching MATLAB's ``[1, n]`` kernel
    pattern (1 along InputSize(1)=C, n along InputSize(2)=T).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_t: int,
        coder: Literal["Encoder", "Decoder"],
        stride_t: int,
        groups: int,
        activation: str,
        dropout: float,
        want_normalization: bool | str,
        is_last_depth: bool,
        crop_t: int = 0,
    ) -> None:
        super().__init__()
        self.coder = coder
        self.is_last_depth = is_last_depth
        self.crop_t = crop_t

        # Non-last depths in a multi-rep level don't carry the stride.
        effective_stride = stride_t if is_last_depth else 1
        # "Padding='same'" along T for odd kernels with stride 1; for
        # strided downsamples we use floor(kernel/2).
        pad_t = kernel_t // 2

        if coder == "Encoder":
            self.conv: nn.Module = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_t),
                stride=(1, effective_stride),
                padding=(0, pad_t),
                groups=groups,
            )
        else:
            # Decoder: transposed conv along T for upsampling.
            tconv_kernel_t = effective_stride * 2
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, tconv_kernel_t),
                stride=(1, effective_stride),
                padding=(0, (tconv_kernel_t - effective_stride) // 2),
                groups=groups,
            )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        # Mid-block activation only when NOT last depth (the level scope
        # applies the trailing activation after the optional ResNet add).
        self.act_mid = _activation_for(activation) if not is_last_depth else None
        self.norm = _normalization_for(want_normalization, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B*W, A, C, T)`` → ``(B*W, A_out, C, T_out)``."""
        x = self.conv(x)
        if self.coder == "Decoder" and self.crop_t > 0:
            x = x[..., : x.size(-1) - self.crop_t]
        if self.dropout is not None:
            x = self.dropout(x)
        if self.act_mid is not None:
            x = self.act_mid(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class _ConvLevel(nn.Module):
    """One pyramid level: ``repetitions_per_block`` blocks + optional ResNet add."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_t: int,
        coder: Literal["Encoder", "Decoder"],
        stride_t: int,
        groups: int,
        activation: str,
        dropout: float,
        want_normalization: bool | str,
        repetitions_per_block: int,
        want_resnet: bool,
        crop_t: int = 0,
    ) -> None:
        super().__init__()
        self.want_resnet = want_resnet
        self.coder = coder

        blocks: list[_PerWindowConvBlock] = []
        for ridx in range(repetitions_per_block):
            is_last = ridx == repetitions_per_block - 1
            blocks.append(
                _PerWindowConvBlock(
                    in_channels=in_channels if ridx == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_t=kernel_t,
                    coder=coder,
                    stride_t=stride_t,
                    groups=groups,
                    activation=activation,
                    dropout=dropout,
                    want_normalization=want_normalization,
                    is_last_depth=is_last,
                    crop_t=crop_t if is_last else 0,
                ),
            )
        self.blocks = nn.ModuleList(blocks)

        if want_resnet:
            # 1×1 conv to match the post-block shape (and stride for
            # encoder; transposed-conv for decoder).
            if coder == "Decoder":
                tconv_kernel = stride_t * 2
                self.residual_proj: Optional[nn.Module] = nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, tconv_kernel),
                    stride=(1, stride_t),
                    padding=(0, (tconv_kernel - stride_t) // 2),
                    groups=groups,
                )
            else:
                self.residual_proj = nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(1, stride_t),
                    padding=0,
                    groups=groups,
                )
            self.act_final = _activation_for(activation)
        else:
            self.residual_proj = None
            self.act_final = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        for blk in self.blocks:
            out = blk(out)
        if self.want_resnet and self.residual_proj is not None:
            skip = self.residual_proj(identity)
            t_min = min(out.size(-1), skip.size(-1))
            out = out[..., :t_min] + skip[..., :t_min]
            if self.act_final is not None:
                out = self.act_final(out)
        return out


class PerWindowConvolutionalCoder(nn.Module):
    """Multi-level per-window 2-D conv encoder/decoder.

    Operates on the new 5-D ``(B, W, T, A, C)`` data path. Per-window
    conv blocks treat the ``(C, T)`` plane as the 2-D spatial axes and
    ``A`` as the conv's channel axis — semantically identical to MATLAB's
    ``convolution2dLayer([1, kernel_t], NumFilters)`` operating on
    ``InputSize=[C, T, A]`` data.

    Parameters
    ----------
    num_areas
        Number of areas (``A``). Becomes the input channel count of the
        first conv layer.
    filter_hidden_sizes
        Per-level output area-channel counts. Encoder applies them in
        order; decoder reverses. With ``want_split_areas=True``, values
        are rounded up to multiples of ``num_areas``.
    kernel_t
        Time-axis kernel length (the ``n`` in MATLAB's ``[1, n]``).
    coder
        ``'Encoder'`` (downsamples T) or ``'Decoder'`` (upsamples T).
    stride_t
        Time-axis stride for down/up-sampling.
    want_split_areas
        ``True`` → grouped convolution with ``groups=num_areas`` (each
        area independent; mirrors MATLAB's ``WantSplitAreas=true``).
        ``False`` → single conv mixes across areas.
    want_resnet
        Per-level residual skip connection.
    repetitions_per_block
        Conv blocks per level; only the last carries stride.
    activation
        ``'ReLU'`` / ``'Leaky ReLU'`` / ``'SoftSign'`` / ``'GeLU'``.
    dropout
        Per-block dropout probability.
    want_normalization
        ``False`` / ``True`` (BatchNorm) / ``'Batch'`` / ``'Instance'`` /
        ``'Layer'``.
    crop_amounts_t
        Optional per-level time-axis crop amounts (decoder only).
        Mirrors MATLAB's ``cropLayer`` for exact T reconstruction.

    Notes
    -----
    The ``C`` (channels-per-area) axis is **not** convolved over — kernels
    are 1 along that axis. This matches MATLAB's typical use of
    ``[1, n]`` kernels and the empirical fact that per-area channels
    rarely have meaningful spatial structure across them. Future
    extensions could add kernel size > 1 along ``C`` for true 2-D
    convolution.
    """

    def __init__(
        self,
        *,
        num_areas: int,
        filter_hidden_sizes: list[int],
        kernel_t: int,
        coder: Literal["Encoder", "Decoder"],
        stride_t: int = 2,
        want_split_areas: bool = True,
        want_resnet: bool = True,
        repetitions_per_block: int = 1,
        activation: str = "ReLU",
        dropout: float = 0.0,
        want_normalization: bool | str = False,
        crop_amounts_t: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        if num_areas < 1:
            raise ValueError(f"num_areas must be >= 1; got {num_areas}.")
        if any(h < 1 for h in filter_hidden_sizes):
            raise ValueError(
                f"filter_hidden_sizes must be positive ints; got {filter_hidden_sizes}.",
            )

        groups = num_areas if want_split_areas else 1
        # Each level's output channels must be a multiple of groups for
        # grouped convolution to be valid.
        levels = [
            max(groups, (h // groups) * groups) for h in filter_hidden_sizes
        ] if want_split_areas else list(filter_hidden_sizes)
        if coder == "Decoder":
            levels = list(reversed(levels))

        if crop_amounts_t is None:
            crop_amounts_t = [0] * len(levels)
        if len(crop_amounts_t) != len(levels):
            raise ValueError(
                f"crop_amounts_t length ({len(crop_amounts_t)}) must match "
                f"number of levels ({len(levels)}).",
            )

        # Build the level stack. For the decoder, the input channel count
        # at level 0 is the FIRST (i.e. last-encoder) level's output =
        # levels[0] after reversal. For the encoder, it's num_areas.
        if coder == "Encoder":
            prev = num_areas
        else:
            prev = levels[0]

        built: list[_ConvLevel] = []
        # Decoder targets: from level 0 -> levels[1] -> ... -> num_areas.
        if coder == "Decoder":
            targets = list(levels[1:]) + [num_areas]
        else:
            targets = list(levels)

        for lidx, out_c in enumerate(targets):
            built.append(
                _ConvLevel(
                    in_channels=prev,
                    out_channels=out_c,
                    kernel_t=kernel_t,
                    coder=coder,
                    stride_t=stride_t,
                    groups=groups,
                    activation=activation,
                    dropout=dropout,
                    want_normalization=want_normalization,
                    repetitions_per_block=repetitions_per_block,
                    want_resnet=want_resnet,
                    crop_t=crop_amounts_t[lidx],
                ),
            )
            prev = out_c
        self.levels = nn.ModuleList(built)
        self.num_areas = num_areas
        self.coder = coder
        # The expected input A axis depends on direction: the encoder
        # takes raw num_areas; the decoder takes the bottleneck A count
        # (filter_hidden_sizes[-1] after the encoder).
        self.in_areas_expected = num_areas if coder == "Encoder" else levels[0]
        self.out_areas = prev  # final A_out after all levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, W, T, A, C)`` → ``(B, W, T', A_out, C)``.

        Internally reshapes to ``(B*W, A, C, T)`` for ``Conv2d``,
        applies the level stack, and reshapes back to 5-D.
        """
        if x.ndim != 5:
            raise ValueError(
                f"PerWindowConvolutionalCoder expects 5-D input (B, W, T, A, C); "
                f"got shape {tuple(x.shape)}.",
            )
        b, w, t, a, c = x.shape
        if a != self.in_areas_expected:
            raise ValueError(
                f"Input A axis ({a}) does not match expected "
                f"in_areas={self.in_areas_expected} for {self.coder!r} mode.",
            )
        # (B, W, T, A, C) -> (B*W, A, C, T)
        z = x.permute(0, 1, 3, 4, 2).reshape(b * w, a, c, t)
        for level in self.levels:
            z = level(z)
        # (B*W, A_out, C, T_out) -> (B, W, T_out, A_out, C)
        a_out, c_out, t_out = z.shape[1], z.shape[2], z.shape[3]
        z = z.reshape(b, w, a_out, c_out, t_out)
        return z.permute(0, 1, 4, 2, 3).contiguous()


__all__ = [
    "PerWindowConvolutionalCoder",
]
