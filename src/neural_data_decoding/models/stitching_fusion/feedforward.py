"""Feedforward Stitching+Fusion bridge and the unified S&F dispatcher.

Implements the ``'Feedforward'`` option-set from
``PARAMETERS_cgg_constructStitchingAndFusionNetwork.m`` (lines 94-130):
a single per-timestep ``Linear`` projection on each side of the
encoder/decoder. ``Transform='Feedforward'``, ``Activation='None'``,
``Dropout=0``, ``WantNormalization=false`` — the MATLAB block reduces
to a bare ``fullyConnectedLayer(out_features)``.

The same module hosts :func:`build_stitching_fusion`, the factory that
dispatches on the MATLAB option-set string. Phase 2 (``'Default'``)
wraps :class:`~neural_data_decoding.models.stitching_fusion.
convolutional.PerWindowConvolutionalCoder` with the appropriate
``Linear`` projection at the boundary. Phase 3 (Gemini variants) is
still pending.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from neural_data_decoding.models.stitching_fusion.convolutional import (
    PerWindowConvolutionalCoder,
)


class FeedforwardStitchingFusion(nn.Module):
    """Single-``Linear`` cross-area fusion bridge.

    Matches the ``'Feedforward'`` option-set in
    ``PARAMETERS_cgg_constructStitchingAndFusionNetwork.m`` (lines 94-130).
    The MATLAB block is a bare ``fullyConnectedLayer(out_features)`` —
    no activation, no dropout, no normalization.

    Parameters
    ----------
    in_features
        Channel count of the incoming tensor.
    out_features
        Channel count after fusion.
    """

    def __init__(self, *, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the per-window linear projection.

        Accepts ``(B, W, T, A, C)`` (5-D, canonical) by flattening the
        within-window dims internally, or ``(B, W, F)`` (3-D fallback)
        for backwards compat.

        Parameters
        ----------
        x
            Tensor shaped ``(B, W, T, A, C)`` or ``(B, W, F)`` where
            ``F = T*A*C`` matches ``self.in_features``.

        Returns
        -------
        torch.Tensor
            Tensor shaped ``(B, W, out_features)``.
        """
        if x.ndim == 5:
            b, w = x.shape[0], x.shape[1]
            x = x.reshape(b, w, -1)
        return self.linear(x)


def build_stitching_fusion(
    network_type: str,
    *,
    in_features: int,
    cross_area_fusion_size: int,
    mode: Literal["Encoder", "Decoder"],
    samples_per_window: int = 1,
    num_areas: int = 1,
) -> nn.Module:
    """Dispatch to the named S&F builder based on MATLAB's option-set string.

    Mirrors ``cgg_constructStitchingAndFusionNetwork.m``'s switch on the
    ``StitchingAndFusionNetworkType`` argument.

    Parameters
    ----------
    network_type
        One of the five option-set names from
        ``PARAMETERS_cgg_constructStitchingAndFusionNetwork.m``:
        ``'Feedforward'``, ``'Default'``, ``'Parallel Single Level'``,
        ``'Cascade Single Kernel - Single Reduction'``,
        ``'Cascade Single Kernel - Progressive Reduction'``.
    in_features
        Raw input channels (``mode='Encoder'``) **or** the encoder's
        ``in_features`` count that the decoder output must project back to
        (``mode='Decoder'``).
    cross_area_fusion_size
        The unified fusion-space channel count (= ``hidden_sizes[0] * 2``
        per the MATLAB convention in
        ``cgg_constructNetworkArchitecture.m`` line 125).
    mode
        ``'Encoder'`` builds the pre-encoder bridge (raw ``in_features`` →
        ``cross_area_fusion_size``); ``'Decoder'`` builds the post-decoder
        bridge (``cross_area_fusion_size`` → raw ``in_features``).

    Returns
    -------
    nn.Module
        The bridge module.

    Raises
    ------
    NotImplementedError
        For the ``'Default'`` and Gemini variants pending in Phases 2-3.
    ValueError
        For an unrecognized ``network_type`` string.
    """
    if network_type == "Feedforward":
        if mode == "Encoder":
            return FeedforwardStitchingFusion(
                in_features=in_features, out_features=cross_area_fusion_size,
            )
        return FeedforwardStitchingFusion(
            in_features=cross_area_fusion_size, out_features=in_features,
        )
    if network_type == "Default":
        return _build_default_stitching_fusion(
            channels_per_area=in_features // max(1, samples_per_window * num_areas)
            if (samples_per_window * num_areas) > 0
            else in_features,
            num_areas=num_areas,
            cross_area_fusion_size=cross_area_fusion_size,
            mode=mode,
        )
    if network_type in {
        "Parallel Single Level",
        "Cascade Single Kernel - Single Reduction",
        "Cascade Single Kernel - Progressive Reduction",
    }:
        raise NotImplementedError(
            f"Stitching+Fusion variant {network_type!r} is pending (CC #3 "
            "Phase 3 — Gemini cascade variants). 'Feedforward' and "
            "'Default' are implemented.",
        )
    raise ValueError(
        f"Unknown stitching_and_fusion_layer: {network_type!r}. Expected "
        "one of {'', 'Feedforward', 'Default', 'Parallel Single Level', "
        "'Cascade Single Kernel - Single Reduction', "
        "'Cascade Single Kernel - Progressive Reduction'}.",
    )


# Defaults from PARAMETERS_cgg_constructStitchingAndFusionNetwork.m
# lines 57-93 ('Default' case): WantSplitAreas=true,
# HiddenSizeAutoEncoder=16, Stride=4, RepetitionsPerBlock=2,
# Activation='Leaky ReLU', WantResnet=true. The 'Default' S&F variant
# also uses the convolutional cross-area encoder/decoder per
# cgg_constructStitchingAndFusionNetwork.m lines 84-129.
_DEFAULT_FILTER_HIDDEN = 16
_DEFAULT_KERNEL_T = 5
_DEFAULT_STRIDE_T = 4
_DEFAULT_REPETITIONS = 2
_DEFAULT_ACTIVATION = "Leaky ReLU"


class _DefaultStitchingFusionBridge(nn.Module):
    """``'Default'`` S&F bridge: per-window conv + ``Linear`` projection.

    Encoder mode mirrors ``cgg_constructStitchingAndFusionNetwork.m``
    lines 84-105: per-area convolutional cross-area encoder followed by
    a ``CrossAreaFusionLayer`` (``fullyConnectedLayer(CrossAreaFusionSize)``).

    Decoder mode mirrors lines 106-129: a leading ``Linear`` from the
    fusion size into a per-window-reshape-ready vector, then the
    convolutional decoder unwinds the per-window 2-D conv stack.

    Parameters
    ----------
    num_areas
        Number of input areas (``A``). With ``num_areas=1`` the
        grouped conv degenerates to standard 2-D.
    in_features_per_area
        Channels per area on the input side (``C``; raw multi-area
        input has ``A * C`` total channels).
    cross_area_fusion_size
        Output channel count of the fusion bridge (``Linear`` projection
        target; derived from ``hidden_sizes[0] * 2`` per MATLAB).
    mode
        ``'Encoder'`` or ``'Decoder'``.
    """

    def __init__(
        self,
        *,
        num_areas: int,
        in_features_per_area: int,
        cross_area_fusion_size: int,
        mode: Literal["Encoder", "Decoder"],
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_areas = num_areas
        self.in_features_per_area = in_features_per_area
        self.cross_area_fusion_size = cross_area_fusion_size
        # Single-level conv (matches HiddenSizeAutoEncoder=16 in MATLAB).
        self.conv = PerWindowConvolutionalCoder(
            num_areas=num_areas,
            filter_hidden_sizes=[_DEFAULT_FILTER_HIDDEN],
            kernel_t=_DEFAULT_KERNEL_T,
            coder=mode,
            stride_t=_DEFAULT_STRIDE_T,
            want_split_areas=True,
            want_resnet=True,
            repetitions_per_block=_DEFAULT_REPETITIONS,
            activation=_DEFAULT_ACTIVATION,
        )
        # The Linear projects between the flattened conv output and
        # cross_area_fusion_size. We size it lazily in forward() because
        # T_out depends on the input length, which we don't know at
        # construction time.
        self._linear: Optional[nn.Linear] = None
        # Flat dim of the conv input side (excluding T which the conv
        # reshapes / strides). The Linear's *other* end is fusion-sized.
        self._other_total_features = (
            in_features_per_area * (
                num_areas if mode == "Decoder" else _DEFAULT_FILTER_HIDDEN
            )
        )

    def _ensure_linear(self, flat_dim: int, device: torch.device) -> nn.Linear:
        if self._linear is None or self._linear.in_features != flat_dim:
            if self.mode == "Encoder":
                self._linear = nn.Linear(flat_dim, self.cross_area_fusion_size).to(device)
            else:
                self._linear = nn.Linear(self.cross_area_fusion_size, flat_dim).to(device)
            # Register so it's saved with state_dict.
            self.add_module("linear", self._linear)
        return self._linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder: ``(B, W, T, A, C)`` → ``(B, W, cross_area_fusion_size)``.

        Decoder: ``(B, W, cross_area_fusion_size)`` → ``(B, W, T, A, C)``.
        """
        if self.mode == "Encoder":
            if x.ndim != 5:
                raise ValueError(
                    f"Default S&F Encoder expects 5-D (B, W, T, A, C); "
                    f"got {tuple(x.shape)}.",
                )
            # Apply per-window conv: (B, W, T, A, C) → (B, W, T', A_out, C).
            y = self.conv(x)
            b, w, t_out, a_out, c = y.shape
            flat = y.reshape(b, w, t_out * a_out * c)
            linear = self._ensure_linear(flat.size(-1), flat.device)
            return linear(flat)
        # Decoder
        if x.ndim != 3:
            raise ValueError(
                f"Default S&F Decoder expects 3-D (B, W, cross_area_fusion_size); "
                f"got {tuple(x.shape)}.",
            )
        b, w, _ = x.shape
        # We need to know (T_in_conv, A_in_conv, C) to unflatten. The
        # conv decoder takes (B, W, T_in, A_in, C) where A_in is the
        # bottleneck A count = filter_hidden_sizes[-1] (here
        # _DEFAULT_FILTER_HIDDEN), and T_in is the post-strided length.
        # We pick T_in=1 (minimal) — the conv decoder's transposed-conv
        # stride will expand it. A_in is the conv's expected input A.
        a_in = self.conv.in_areas_expected
        c = self.in_features_per_area
        t_in = 1
        flat_dim = t_in * a_in * c
        linear = self._ensure_linear(flat_dim, x.device)
        y = linear(x).reshape(b, w, t_in, a_in, c)
        return self.conv(y)


def _build_default_stitching_fusion(
    *,
    channels_per_area: int,
    num_areas: int,
    cross_area_fusion_size: int,
    mode: Literal["Encoder", "Decoder"],
) -> nn.Module:
    """Construct the ``'Default'`` S&F bridge — picks encoder/decoder by mode.

    The bridge's :class:`PerWindowConvolutionalCoder` operates per-window
    on the ``(T, A, C)`` axes with ``A`` as the conv channel and
    ``[1, n]`` kernels filtering only along ``T`` (no cross-channel
    mixing along ``C``).
    """
    return _DefaultStitchingFusionBridge(
        num_areas=num_areas,
        in_features_per_area=channels_per_area,
        cross_area_fusion_size=cross_area_fusion_size,
        mode=mode,
    )


__all__ = [
    "FeedforwardStitchingFusion",
    "build_stitching_fusion",
]
