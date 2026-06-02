"""Convolutional / ResNet / Multi-Filter encoder builders.

Pythonic adapters that present the per-window 2-D conv stack
(:class:`~neural_data_decoding.models.stitching_fusion.convolutional.PerWindowConvolutionalCoder`)
and the multi-scale cascaded fusion module
(:class:`~neural_data_decoding.models.stitching_fusion.gemini.GeminiStitchingFusionModule`)
as standard 3-D-in / 3-D-out encoders so they fit the same composite
slot as :class:`~neural_data_decoding.models.encoder.SimpleSequenceEncoder`.

Why an adapter
--------------
The composite's contract is: encoder receives 3-D ``(B, W, F)`` and
returns 3-D ``(B, W, F_out)``. The per-window conv builders operate on
the explicit 5-D ``(B, W, T, A, C)`` layout because their kernels are
``(1, kernel_t)`` over the ``(C, T)`` plane (mirroring MATLAB's
``[1, n]`` 2-D convolutions — kernels never cross the ``C`` axis).
This adapter unflattens the composite's 3-D input to 5-D, runs the
conv stack, and re-flattens.

This is a Pythonic deviation from MATLAB's
``cgg_constructConvolutionalCoder.m`` topology — MATLAB builds the
conv layers directly on ``dlarray`` formatted as ``"CBTSS"`` and
relies on implicit reshape via ``functionLayer``. We keep the
functionally identical operations (same kernel, stride, grouping,
activation) but make the reshape boundaries explicit and isolated to
the adapter.

Registered architectures
------------------------
* ``'Convolutional'`` — single-filter, no ResNet (mirrors MATLAB
  ``cgg_constructNetworkArchitecture.m`` ``'Convolutional'`` case
  with ``FilterSizes={[4,20]}, Stride=2, Activation='Leaky ReLU',
  WantResnet=false``).
* ``'Resnet'`` — single-filter, ResNet residual connections
  (``WantResnet=true``; otherwise identical to ``'Convolutional'``).
* ``'Multi-Filter Convolutional'`` — three parallel kernel sizes
  ``[3, 5, 7]`` (uses the Gemini ``'Parallel Single Level'`` module
  which implements the same parallel-multi-scale pattern).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from neural_data_decoding.models.registry import register_encoder
from neural_data_decoding.models.stitching_fusion.convolutional import (
    PerWindowConvolutionalCoder,
)
from neural_data_decoding.models.stitching_fusion.gemini import (
    build_gemini_stitching_fusion,
)


def _infer_channels_per_area(
    in_features: int, samples_per_window: int, num_areas: int,
) -> int:
    """Derive ``C`` from the flat ``in_features = T * A * C`` contract."""
    denom = samples_per_window * num_areas
    if denom < 1:
        raise ValueError(
            "samples_per_window and num_areas must both be >= 1; got "
            f"samples_per_window={samples_per_window}, num_areas={num_areas}.",
        )
    if in_features % denom != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by "
            f"samples_per_window * num_areas ({samples_per_window} * "
            f"{num_areas} = {denom}) so the channels-per-area count C is "
            "an integer.",
        )
    return in_features // denom


def _kernel_t_for(
    samples_per_window: int, kernel_t: int | None,
) -> int:
    """Pick a temporal kernel size that doesn't exceed the available T axis.

    MATLAB derives kernels from ``FilterSizePercent`` and ``InputSize``
    (e.g. ``ceil(T * 0.3)`` for the default ``Convolutional``). The
    Python adapter accepts an explicit ``kernel_t`` override; the
    fallback is the larger of ``3`` or ``ceil(T * 0.3)``.
    """
    if kernel_t is not None:
        if kernel_t < 1:
            raise ValueError(f"kernel_t must be >= 1; got {kernel_t}.")
        return min(kernel_t, samples_per_window)
    return max(1, min(samples_per_window, max(3, (samples_per_window + 2) // 3)))


class ConvolutionalEncoder(nn.Module):
    """3-D-in / 3-D-out adapter wrapping :class:`PerWindowConvolutionalCoder`.

    Implements the MATLAB ``'Convolutional'`` and ``'Resnet'``
    architectures (cgg_constructNetworkArchitecture.m lines 191-228) on
    the Python 5-D ``(B, W, T, A, C)`` data path.

    The adapter:

    1. Accepts the composite's standard 3-D input ``(B, W, T*A*C)``.
    2. Unflattens to ``(B, W, T, A, C)`` using ``samples_per_window`` /
       ``num_areas`` from construction.
    3. Runs the per-window 2-D conv stack — kernels ``(1, kernel_t)``,
       filter along ``T`` only (never crosses the per-area ``C`` axis),
       optional ``groups=A`` for split-areas.
    4. Re-flattens the conv output to 3-D for the composite.

    Parameters
    ----------
    in_features
        Flat feature count per window (``T * A * C``).
    samples_per_window
        ``T`` — within-window time samples.
    num_areas
        ``A`` — number of areas (probes).
    filter_hidden_sizes
        Per-level output filter counts (per area, when split-areas).
    kernel_t
        Temporal kernel size. ``None`` → auto-pick from
        :func:`_kernel_t_for`.
    stride_t
        Down-sample stride along ``T``.
    want_split_areas
        ``True`` → grouped conv per area (no cross-area mixing).
    want_resnet
        ``True`` → ``'Resnet'`` architecture; ``False`` →
        ``'Convolutional'``.
    activation
        ``'ReLU'`` / ``'Leaky ReLU'`` / ``'GeLU'`` / ``'SoftSign'``.
    dropout, want_normalization, repetitions_per_block
        Pass-through to :class:`PerWindowConvolutionalCoder`.

    Attributes
    ----------
    out_features : int
        Flat feature count per window after the conv stack. Used by
        the composite's bottleneck for sizing.
    """

    def __init__(
        self,
        *,
        in_features: int,
        samples_per_window: int,
        num_areas: int,
        filter_hidden_sizes: list[int],
        kernel_t: int | None = None,
        stride_t: int = 2,
        want_split_areas: bool = False,
        want_resnet: bool = False,
        activation: str = "Leaky ReLU",
        dropout: float = 0.0,
        want_normalization: bool | str = False,
        repetitions_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.samples_per_window = samples_per_window
        self.num_areas = num_areas
        self.channels_per_area = _infer_channels_per_area(
            in_features, samples_per_window, num_areas,
        )
        self.in_features = in_features

        effective_kernel_t = _kernel_t_for(samples_per_window, kernel_t)
        self.conv = PerWindowConvolutionalCoder(
            num_areas=num_areas,
            filter_hidden_sizes=filter_hidden_sizes,
            kernel_t=effective_kernel_t,
            coder="Encoder",
            stride_t=stride_t,
            want_split_areas=want_split_areas,
            want_resnet=want_resnet,
            repetitions_per_block=repetitions_per_block,
            activation=activation,
            dropout=dropout,
            want_normalization=want_normalization,
        )
        self.out_features = self._compute_out_features()

    def _compute_out_features(self) -> int:
        """Run the conv stack on a dummy tensor to learn the output flat dim."""
        dummy = torch.zeros(
            1, 1,
            self.samples_per_window, self.num_areas, self.channels_per_area,
        )
        with torch.no_grad():
            y = self.conv(dummy)
        # (1, 1, T_out, A_out, C) → flat dim = T_out * A_out * C.
        return int(y.shape[2] * y.shape[3] * y.shape[4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, W, T*A*C) → (B, W, T_out*A_out*C)``.

        Parameters
        ----------
        x
            3-D tensor ``(B, W, in_features)`` from the composite's
            flatten step.

        Returns
        -------
        torch.Tensor
            3-D tensor ``(B, W, out_features)`` ready for the bottleneck.
        """
        if x.ndim != 3:
            raise ValueError(
                f"ConvolutionalEncoder expects 3-D input (B, W, F); "
                f"got shape {tuple(x.shape)}.",
            )
        b, w, _ = x.shape
        x5 = x.reshape(
            b, w,
            self.samples_per_window, self.num_areas, self.channels_per_area,
        )
        y5 = self.conv(x5)
        return y5.reshape(b, w, -1)


class MultiFilterConvolutionalEncoder(nn.Module):
    """3-D-in / 3-D-out adapter wrapping the Gemini parallel-multi-scale module.

    Implements the MATLAB ``'Multi-Filter Convolutional'`` architecture
    (cgg_constructNetworkArchitecture.m lines 249-266) on the 5-D data
    path. Routes through
    :class:`~neural_data_decoding.models.stitching_fusion.gemini.GeminiStitchingFusionModule`'s
    ``'Parallel Single Level'`` variant — three parallel ``[1, k]``
    temporal kernels (default ``k ∈ [3, 5, 7]``) summed via the
    cascade-bypass addition. Per the Pythonic design principle, no
    standalone multi-filter builder is added since the Gemini module
    already implements the pattern.

    Parameters
    ----------
    in_features
        Flat feature count per window (``T * A * C``).
    samples_per_window
        ``T``.
    num_areas
        ``A``.
    filters_per_area
        Per-area output filter count for the Gemini module (mirrors
        MATLAB ``FilterHiddenSizes(1)``).

    Attributes
    ----------
    out_features : int
        Flat feature count per window after the multi-filter stack.
    """

    _GEMINI_VARIANT = "Parallel Single Level"

    def __init__(
        self,
        *,
        in_features: int,
        samples_per_window: int,
        num_areas: int,
        filters_per_area: int = 8,
    ) -> None:
        super().__init__()
        self.samples_per_window = samples_per_window
        self.num_areas = num_areas
        self.channels_per_area = _infer_channels_per_area(
            in_features, samples_per_window, num_areas,
        )
        self.in_features = in_features
        self.gemini = build_gemini_stitching_fusion(
            self._GEMINI_VARIANT,
            num_areas=num_areas,
            filters_per_area=filters_per_area,
            mode="Encoder",
        )
        self.out_features = self._compute_out_features()

    def _compute_out_features(self) -> int:
        dummy = torch.zeros(
            1, 1,
            self.samples_per_window, self.num_areas, self.channels_per_area,
        )
        with torch.no_grad():
            y = self.gemini(dummy)
        return int(y.shape[2] * y.shape[3] * y.shape[4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"MultiFilterConvolutionalEncoder expects 3-D input "
                f"(B, W, F); got shape {tuple(x.shape)}.",
            )
        b, w, _ = x.shape
        x5 = x.reshape(
            b, w,
            self.samples_per_window, self.num_areas, self.channels_per_area,
        )
        y5 = self.gemini(x5)
        return y5.reshape(b, w, -1)


# ───────────────────────── Registry entries ─────────────────────────


def _conv_cfg(cfg: Mapping[str, Any]) -> dict:
    """Common cfg-key resolution for the conv encoder builders."""
    try:
        in_features = int(cfg["in_features"])
        hidden_sizes = [int(h) for h in cfg.get("hidden_sizes", [16])]
    except KeyError as exc:
        raise KeyError(
            f"Convolutional encoder builder: missing cfg key {exc}",
        ) from exc
    return {
        "in_features": in_features,
        "samples_per_window": int(cfg.get("samples_per_window", 1)),
        "num_areas": int(cfg.get("num_areas", 1)),
        "filter_hidden_sizes": hidden_sizes,
        "kernel_t": cfg.get("kernel_t"),  # None → auto-pick
        "stride_t": int(cfg.get("stride", 2)),
        "want_split_areas": bool(cfg.get("want_split_areas", False)),
        "activation": str(cfg.get("activation", "Leaky ReLU")),
        "dropout": float(cfg.get("dropout", 0.0)),
        "want_normalization": cfg.get("want_normalization", False),
        "repetitions_per_block": int(cfg.get("repetitions_per_block", 1)),
    }


def build_convolutional_encoder(cfg: Mapping[str, Any]) -> ConvolutionalEncoder:
    """Builder for the ``'Convolutional'`` architecture."""
    kwargs = _conv_cfg(cfg)
    return ConvolutionalEncoder(want_resnet=False, **kwargs)


def build_resnet_encoder(cfg: Mapping[str, Any]) -> ConvolutionalEncoder:
    """Builder for the ``'Resnet'`` architecture."""
    kwargs = _conv_cfg(cfg)
    return ConvolutionalEncoder(want_resnet=True, **kwargs)


def build_multi_filter_convolutional_encoder(
    cfg: Mapping[str, Any],
) -> MultiFilterConvolutionalEncoder:
    """Builder for the ``'Multi-Filter Convolutional'`` architecture."""
    try:
        in_features = int(cfg["in_features"])
    except KeyError as exc:
        raise KeyError(
            f"Multi-filter conv encoder builder: missing cfg key {exc}",
        ) from exc
    hidden_sizes = [int(h) for h in cfg.get("hidden_sizes", [8])]
    return MultiFilterConvolutionalEncoder(
        in_features=in_features,
        samples_per_window=int(cfg.get("samples_per_window", 1)),
        num_areas=int(cfg.get("num_areas", 1)),
        filters_per_area=hidden_sizes[0],
    )


register_encoder("Convolutional")(build_convolutional_encoder)
register_encoder("Resnet")(build_resnet_encoder)
register_encoder("Multi-Filter Convolutional")(
    build_multi_filter_convolutional_encoder,
)


__all__ = [
    "ConvolutionalEncoder",
    "MultiFilterConvolutionalEncoder",
    "build_convolutional_encoder",
    "build_multi_filter_convolutional_encoder",
    "build_resnet_encoder",
]
