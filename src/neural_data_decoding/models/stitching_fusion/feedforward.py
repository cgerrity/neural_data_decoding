"""Feedforward Stitching+Fusion bridge and the unified S&F dispatcher.

Implements the ``'Feedforward'`` option-set from
``PARAMETERS_cgg_constructStitchingAndFusionNetwork.m`` (lines 94-130):
a single per-timestep ``Linear`` projection on each side of the
encoder/decoder. ``Transform='Feedforward'``, ``Activation='None'``,
``Dropout=0``, ``WantNormalization=false`` — the MATLAB block reduces
to a bare ``fullyConnectedLayer(out_features)``.

The same module hosts :func:`build_stitching_fusion`, the factory that
dispatches on the MATLAB option-set string. Phase 2 (Default) and Phase 3
(Gemini variants) will register their builders here too.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


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
        """Apply the per-timestep linear projection.

        Parameters
        ----------
        x
            Tensor shaped ``(batch, time, in_features)``.

        Returns
        -------
        torch.Tensor
            Tensor shaped ``(batch, time, out_features)``.
        """
        return self.linear(x)


def build_stitching_fusion(
    network_type: str,
    *,
    in_features: int,
    cross_area_fusion_size: int,
    mode: Literal["Encoder", "Decoder"],
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
    if network_type in {
        "Default",
        "Parallel Single Level",
        "Cascade Single Kernel - Single Reduction",
        "Cascade Single Kernel - Progressive Reduction",
    }:
        raise NotImplementedError(
            f"Stitching+Fusion variant {network_type!r} is pending (CC #3 "
            "Phases 2-3). 'Feedforward' is the only variant implemented "
            "in Phase 1.",
        )
    raise ValueError(
        f"Unknown stitching_and_fusion_layer: {network_type!r}. Expected "
        "one of {'', 'Feedforward', 'Default', 'Parallel Single Level', "
        "'Cascade Single Kernel - Single Reduction', "
        "'Cascade Single Kernel - Progressive Reduction'}.",
    )


__all__ = [
    "FeedforwardStitchingFusion",
    "build_stitching_fusion",
]
