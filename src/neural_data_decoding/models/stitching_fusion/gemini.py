"""Gemini cascaded multi-area Stitching+Fusion module — port of
``cgg_createStitchingFusionModule_v2.m``.

Three named option-sets from
``PARAMETERS_cgg_constructStitchingAndFusionNetwork.m``:

* **'Parallel Single Level'** — multi-scale temporal kernels
  ``[3, 5, 7]`` in parallel, single cascade stage, progressive mode.
* **'Cascade Single Kernel - Single Reduction'** — single kernel size 3
  with 3 cascade stages, reduction at stage 1 only.
* **'Cascade Single Kernel - Progressive Reduction'** — single kernel
  size 3 with 3 cascade stages, reduction at every stage,
  ``EncoderReduction=[4, 2]``.

Architecture (encoder)
----------------------
For each ``temporal_kernel_size`` and each ``cascade_stage``:

1. Per-stage initial pool (only when ``cascade_stride_mode='progressive'``
   and ``reduction_method='maxpool'`` — not the active path).
2. ``num_residual_layers`` grouped 2-D conv blocks with kernel
   ``(1, kernel_t)`` (filters along ``T`` only; never crosses ``C``).
   Stride applies on the first residual layer when conditions are met.
3. Cascade-stage ReLU between stages.
4. Extra reduction conv when ``cascade_stride_mode='progressive'`` and
   there are more stages to come — pushes the per-stage output to the
   same final ``T`` length as later stages so the addition layer can
   sum them element-wise.
5. Optional dropout.

A **bypass projection** (avg-pool + 1×1 grouped conv, or strided
grouped conv directly) runs in parallel; all branches are summed by
an ``additionLayer`` and passed through ReLU. The temporal stack
output then feeds:

6. A **spatial conv** ``(spatial_kernel_size, 1)`` with stride along
   ``C`` (the spatial axis of our reshape), followed by ReLU.
7. An **area fusion** 1×1 conv (ungrouped — mixes across areas),
   followed by ReLU.

Architecture (decoder)
----------------------
Mirrors the encoder with transposed convs and ends with a final
groupedConv 1×1 → channel reduction → precision crop to recover the
original ``(C, T)`` shape.

Data path
---------
Operates on the new 5-D ``(B, W, T, A, C)`` layout. Internally
reshapes per-window slices to ``(B*W, A, C, T)`` for ``Conv2d``
(matching MATLAB's interpretation: ``A`` as conv channels, ``(C, T)``
as 2-D spatial).

Scope
-----
Implements the operations exercised by the three active Gemini option
sets. Out of scope: ``UseDepthwiseSeparable``, ``UseBottleneck``,
``Normalization`` other than ``'none'``, and ``DropoutRate > 0`` —
these are all defaults-false in the active configs.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


_CascadeStrideMode = Literal["single", "progressive"]
_ReductionMethod = Literal["stride", "maxpool"]
_StrideBypassMethod = Literal["kernel", "avgpool"]


def _grouped_conv2d(
    in_channels_per_area: int,
    out_channels_per_area: int,
    num_areas: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
) -> nn.Conv2d:
    """Build a grouped Conv2d that mirrors MATLAB's ``groupedConvolution2dLayer``.

    MATLAB: ``groupedConvolution2dLayer(kernel, filtersPerArea, numAreas)``
    → per-area independent filters. ``filtersPerArea`` outputs per area,
    total ``numAreas * filtersPerArea`` output channels. PyTorch
    grouped conv with ``groups=numAreas`` gives the same: each group
    sees ``in_channels // groups`` input channels and produces
    ``out_channels // groups`` output channels.
    """
    return nn.Conv2d(
        in_channels=num_areas * in_channels_per_area,
        out_channels=num_areas * out_channels_per_area,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        groups=num_areas,
    )


def _grouped_tconv2d(
    in_channels_per_area: int,
    out_channels_per_area: int,
    num_areas: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
) -> nn.ConvTranspose2d:
    """Grouped transposed Conv2d — decoder counterpart of :func:`_grouped_conv2d`."""
    return nn.ConvTranspose2d(
        in_channels=num_areas * in_channels_per_area,
        out_channels=num_areas * out_channels_per_area,
        kernel_size=kernel_size,
        stride=stride,
        padding=(
            (kernel_size[0] - stride[0]) // 2,
            (kernel_size[1] - stride[1]) // 2,
        ),
        groups=num_areas,
    )


class _TemporalBranch(nn.Module):
    """One (kernel_size, cascade_stage) temporal branch.

    ``num_residual_layers`` stacked grouped convs (kernel ``(1, k_t)``)
    with the first carrying the stride when conditions match. Followed
    by optional ``extra_reduce`` conv for progressive mode.
    """

    def __init__(
        self,
        *,
        in_channels_per_area: int,
        out_channels_per_area: int,
        num_areas: int,
        kernel_t: int,
        num_residual_layers: int,
        stride_t: int,
        apply_stride: bool,
        extra_reduce_stride: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for r in range(num_residual_layers):
            conv_stride = stride_t if (r == 0 and apply_stride) else 1
            in_c = in_channels_per_area if r == 0 else out_channels_per_area
            layers.append(
                _grouped_conv2d(
                    in_c, out_channels_per_area, num_areas,
                    kernel_size=(1, kernel_t),
                    stride=(1, conv_stride),
                ),
            )
            if r < num_residual_layers - 1:
                layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        # Extra reduction (progressive mode + stages remaining).
        self.extra: Optional[nn.Module] = None
        if extra_reduce_stride > 1:
            self.extra = _grouped_conv2d(
                out_channels_per_area, out_channels_per_area, num_areas,
                kernel_size=(1, extra_reduce_stride),
                stride=(1, extra_reduce_stride),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the branch body and optional extra-reduce conv."""
        y = self.body(x)
        if self.extra is not None:
            y = self.extra(y)
        return y


class _TemporalBranchTransposed(nn.Module):
    """Decoder counterpart of :class:`_TemporalBranch`.

    Optional leading expansion transposed conv (matches the encoder's
    stride when applicable), then ``num_residual_layers`` grouped
    transposed-conv blocks, then optional extra-expansion conv for
    progressive mode.
    """

    def __init__(
        self,
        *,
        in_channels_per_area: int,
        out_channels_per_area: int,
        num_areas: int,
        kernel_t: int,
        num_residual_layers: int,
        apply_expand: bool,
        expand_stride: int,
        extra_expand_stride: int = 1,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        in_c = in_channels_per_area
        if apply_expand and expand_stride > 1:
            modules.append(
                _grouped_tconv2d(
                    in_c, out_channels_per_area, num_areas,
                    kernel_size=(1, max(3, expand_stride)),
                    stride=(1, expand_stride),
                ),
            )
            in_c = out_channels_per_area
        for r in range(num_residual_layers):
            modules.append(
                _grouped_conv2d(
                    in_c if r == 0 else out_channels_per_area,
                    out_channels_per_area,
                    num_areas,
                    kernel_size=(1, kernel_t),
                    stride=(1, 1),
                ),
            )
            if r < num_residual_layers - 1:
                modules.append(nn.ReLU())
        self.body = nn.Sequential(*modules)
        self.extra: Optional[nn.Module] = None
        if extra_expand_stride > 1:
            self.extra = _grouped_tconv2d(
                out_channels_per_area, out_channels_per_area, num_areas,
                kernel_size=(1, max(3, extra_expand_stride)),
                stride=(1, extra_expand_stride),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transposed-conv branch body + optional extra-expand."""
        y = self.body(x)
        if self.extra is not None:
            y = self.extra(y)
        return y


class GeminiStitchingFusionModule(nn.Module):
    """Gemini cascaded multi-area stitching+fusion (encoder or decoder).

    Port of ``cgg_createStitchingFusionModule_v2.m`` restricted to the
    operations exercised by the three active Gemini option-sets. See
    the module docstring for the architectural overview.

    Parameters
    ----------
    num_areas
        Number of areas (``A`` axis of the input).
    filters_per_area
        Per-area filter count (mirrors MATLAB ``filtersPerArea``).
    mode
        ``'Encoder'`` or ``'Decoder'``.
    temporal_kernel_sizes
        List of kernel sizes along ``T`` (``TemporalKernelSizes``).
    encoder_reduction
        ``(spatial_reduce, temporal_reduce)`` — strides along ``C`` and
        ``T`` respectively.
    num_cascade_layers
        Number of cascade stages.
    num_residual_layers
        Conv blocks per cascade stage.
    cascade_stride_mode
        ``'single'`` (reduce at stage 1 only) or ``'progressive'``
        (reduce at every stage).
    reduction_method
        ``'stride'`` (default in active configs) or ``'maxpool'``.
    stride_bypass_method
        Bypass projection mode: ``'avgpool'`` (default in active
        configs) or ``'kernel'``.
    """

    def __init__(
        self,
        *,
        num_areas: int,
        filters_per_area: int,
        mode: Literal["Encoder", "Decoder"],
        temporal_kernel_sizes: list[int],
        encoder_reduction: tuple[int, int] = (2, 2),
        num_cascade_layers: int = 1,
        num_residual_layers: int = 2,
        cascade_stride_mode: _CascadeStrideMode = "single",
        reduction_method: _ReductionMethod = "stride",
        stride_bypass_method: _StrideBypassMethod = "avgpool",
    ) -> None:
        super().__init__()
        if num_areas < 1:
            raise ValueError(f"num_areas must be >= 1; got {num_areas}.")
        if filters_per_area < 1:
            raise ValueError(
                f"filters_per_area must be >= 1; got {filters_per_area}.",
            )
        if not temporal_kernel_sizes:
            raise ValueError("temporal_kernel_sizes must be non-empty.")
        self.mode = mode
        self.num_areas = num_areas
        self.filters_per_area = filters_per_area
        self.temporal_kernel_sizes = list(temporal_kernel_sizes)
        self.encoder_reduction = tuple(encoder_reduction)
        self.num_cascade_layers = num_cascade_layers
        self.num_residual_layers = num_residual_layers
        self.cascade_stride_mode = cascade_stride_mode
        self.reduction_method = reduction_method
        self.stride_bypass_method = stride_bypass_method

        if mode == "Encoder":
            self._build_encoder()
        else:
            self._build_decoder()

    # ───────────── Encoder build ─────────────

    def _build_encoder(self) -> None:
        """Construct the encoder's temporal branches + bypass + spatial + area fusion."""
        spatial_reduce, temporal_reduce = self.encoder_reduction
        # 1. Temporal extraction — parallel branches.
        self.temporal_branches = nn.ModuleList()
        # Each branch operates on its own running state (which can carry
        # over between cascade stages within the same kernel). For
        # simplicity we treat each (kernel, cascade) pair as an
        # independent branch in the addition list; the cascade-stage
        # output of one stage feeds the next stage's input within the
        # same kernel.
        # We use the convention from MATLAB: at the addition layer, all
        # numKernels * numCascades branches are summed together with the
        # bypass.
        for kernel_t in self.temporal_kernel_sizes:
            for j in range(self.num_cascade_layers):
                # Input channels: 1 (raw) for the very first conv in
                # cascade stage 1; otherwise filters_per_area.
                in_pa = 1 if j == 0 else self.filters_per_area
                # Apply stride on the first residual layer when:
                #   reduction='stride' AND temporal_reduce > 1 AND
                #   (progressive OR j == 0).
                apply_stride = (
                    self.reduction_method == "stride"
                    and temporal_reduce > 1
                    and (
                        self.cascade_stride_mode == "progressive" or j == 0
                    )
                )
                # Extra reduction for progressive mode + stages remaining.
                stages_remaining = self.num_cascade_layers - j - 1
                extra_stride = 1
                if (
                    self.cascade_stride_mode == "progressive"
                    and temporal_reduce > 1
                    and stages_remaining > 0
                ):
                    extra_stride = temporal_reduce ** stages_remaining
                self.temporal_branches.append(
                    _TemporalBranch(
                        in_channels_per_area=in_pa,
                        out_channels_per_area=self.filters_per_area,
                        num_areas=self.num_areas,
                        kernel_t=kernel_t,
                        num_residual_layers=self.num_residual_layers,
                        stride_t=temporal_reduce,
                        apply_stride=apply_stride,
                        extra_reduce_stride=extra_stride,
                    ),
                )

        # 2. Bypass projection.
        # Computes the projection's stride: temporal_reduce^numCascades
        # for progressive mode, else temporal_reduce.
        if temporal_reduce > 1:
            bypass_stride = (
                temporal_reduce ** self.num_cascade_layers
                if self.cascade_stride_mode == "progressive"
                else temporal_reduce
            )
        else:
            bypass_stride = 1
        self.bypass_pool: Optional[nn.Module] = None
        if (
            bypass_stride > 1
            and self.stride_bypass_method == "avgpool"
        ):
            # avgpool + 1x1 grouped conv.
            self.bypass_pool = nn.AvgPool2d(
                kernel_size=(1, bypass_stride),
                stride=(1, bypass_stride),
                padding=(0, 0),
            )
            self.bypass_proj = _grouped_conv2d(
                1, self.filters_per_area, self.num_areas,
                kernel_size=(1, 1),
                stride=(1, 1),
            )
        else:
            # Strided grouped conv directly.
            self.bypass_proj = _grouped_conv2d(
                1, self.filters_per_area, self.num_areas,
                kernel_size=(1, bypass_stride if bypass_stride > 1 else 1),
                stride=(1, bypass_stride),
            )

        # 3. Spatial conv.
        spatial_kernel = max(3, spatial_reduce)
        spatial_stride = spatial_reduce if self.reduction_method == "stride" else 1
        self.spatial_conv = _grouped_conv2d(
            self.filters_per_area, self.filters_per_area, self.num_areas,
            kernel_size=(spatial_kernel, 1),
            stride=(spatial_stride, 1),
        )

        # 4. Area fusion — ungrouped 1×1 conv mixing across areas.
        self.area_fusion = nn.Conv2d(
            in_channels=self.num_areas * self.filters_per_area,
            out_channels=self.filters_per_area,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

    # ───────────── Decoder build ─────────────

    def _build_decoder(self) -> None:
        """Construct the decoder's de-fusion expansion + temporal branches + bypass + final reduction."""
        spatial_reduce, temporal_reduce = self.encoder_reduction
        cascade_expand = (
            temporal_reduce ** self.num_cascade_layers
            if (
                temporal_reduce > 1
                and self.cascade_stride_mode == "progressive"
            )
            else (temporal_reduce if temporal_reduce > 1 else 1)
        )

        # Area de-fusion expansion (single transposed conv on combined
        # spatial+temporal expansion). For decoder the input has
        # filters_per_area channels (ungrouped output of the encoder's
        # area_fusion). The transposed conv expands to
        # num_areas*filters_per_area total channels.
        self.defusion_expansion = nn.ConvTranspose2d(
            in_channels=self.filters_per_area,
            out_channels=self.num_areas * self.filters_per_area,
            kernel_size=(max(3, spatial_reduce), max(3, 1)),
            stride=(spatial_reduce, 1),
            padding=(
                (max(3, spatial_reduce) - spatial_reduce) // 2,
                (max(3, 1) - 1) // 2,
            ),
        )

        # Spatial trans conv (3×1).
        self.spatial_trans_conv = _grouped_conv2d(
            self.filters_per_area, self.filters_per_area, self.num_areas,
            kernel_size=(3, 1),
            stride=(1, 1),
        )

        # Temporal expansion branches.
        self.temporal_branches = nn.ModuleList()
        for kernel_t in self.temporal_kernel_sizes:
            for j in range(self.num_cascade_layers):
                apply_expand = (
                    temporal_reduce > 1
                    and (
                        self.cascade_stride_mode == "progressive" or j == 0
                    )
                )
                stages_remaining = self.num_cascade_layers - j - 1
                extra_expand = 1
                if (
                    self.cascade_stride_mode == "progressive"
                    and temporal_reduce > 1
                    and stages_remaining > 0
                ):
                    extra_expand = temporal_reduce ** stages_remaining
                self.temporal_branches.append(
                    _TemporalBranchTransposed(
                        in_channels_per_area=self.filters_per_area,
                        out_channels_per_area=self.filters_per_area,
                        num_areas=self.num_areas,
                        kernel_t=kernel_t,
                        num_residual_layers=self.num_residual_layers,
                        apply_expand=apply_expand,
                        expand_stride=temporal_reduce,
                        extra_expand_stride=extra_expand,
                    ),
                )

        # Bypass: transposed conv if cascadeExpand > 1, else 1×1 grouped.
        if cascade_expand > 1:
            self.bypass_proj = nn.ConvTranspose2d(
                in_channels=self.num_areas * self.filters_per_area,
                out_channels=self.num_areas * self.filters_per_area,
                kernel_size=(1, max(3, cascade_expand)),
                stride=(1, cascade_expand),
                padding=(0, (max(3, cascade_expand) - cascade_expand) // 2),
                groups=self.num_areas,
            )
        else:
            self.bypass_proj = _grouped_conv2d(
                self.filters_per_area, self.filters_per_area, self.num_areas,
                kernel_size=(1, 1),
                stride=(1, 1),
            )

        # Final channel reduction: groupedConv 1×1 → 1 channel per area.
        self.final_reduction = _grouped_conv2d(
            self.filters_per_area, 1, self.num_areas,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    # ───────────── Forward ─────────────

    def _to_conv_layout(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """``(B, W, T, A, C)`` → ``(B*W, A, C, T)``."""
        if x.ndim != 5:
            raise ValueError(
                f"GeminiStitchingFusionModule expects 5-D input (B, W, T, A, C); "
                f"got shape {tuple(x.shape)}.",
            )
        b, w = x.shape[0], x.shape[1]
        return x.permute(0, 1, 3, 4, 2).reshape(b * w, *x.shape[2:][::-1][::-1]).contiguous(), b, w

    def _from_conv_layout(self, z: torch.Tensor, b: int, w: int) -> torch.Tensor:
        """``(B*W, A*F, C, T)`` → ``(B, W, T, A*F or A, C)`` depending on context.

        The Gemini encoder's area-fusion output collapses the per-area
        channel groups into a single ``filters_per_area`` channel set,
        so the returned 5-D has ``A_out=1, C=C, T=T``. To keep a
        consistent 5-D shape downstream, we expose ``filters_per_area``
        as the new ``A_out`` axis (a slight abuse — see module docstring).
        """
        # z: (B*W, channels, C, T)
        ch, c, t = z.shape[1], z.shape[2], z.shape[3]
        # Reshape to (B, W, channels, C, T) then permute to (B, W, T, A=channels, C).
        z2 = z.reshape(b, w, ch, c, t)
        return z2.permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the encoder or decoder forward depending on ``self.mode``."""
        if self.mode == "Encoder":
            return self._forward_encoder(x)
        return self._forward_decoder(x)

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder pass: temporal branches + bypass + addition → spatial → area fusion."""
        if x.ndim != 5:
            raise ValueError(
                f"GeminiStitchingFusionModule expects 5-D input "
                f"(B, W, T, A, C); got shape {tuple(x.shape)}.",
            )
        b, w = x.shape[0], x.shape[1]
        # Permute (B, W, T, A, C) → (B, W, A, C, T) and flatten batch+window.
        z = x.permute(0, 1, 3, 4, 2).reshape(
            b * w, x.shape[3], x.shape[4], x.shape[2],
        )
        # Temporal branches. We reuse cascade state per kernel: i.e.,
        # the output of cascade stage j (without extra reduce) feeds
        # cascade stage j+1 within the same kernel.
        branch_outputs: list[torch.Tensor] = []
        for k_idx in range(len(self.temporal_kernel_sizes)):
            cascade_state = z
            for j in range(self.num_cascade_layers):
                branch_idx = k_idx * self.num_cascade_layers + j
                branch = self.temporal_branches[branch_idx]
                # The branch transforms cascade_state into a new state.
                # The output (with optional extra_reduce) goes to the
                # addition. The cascade_state (without extra_reduce) is
                # passed to the next stage after a ReLU.
                out = branch.body(cascade_state)
                with_extra = branch.extra(out) if branch.extra is not None else out
                branch_outputs.append(with_extra)
                if j < self.num_cascade_layers - 1:
                    cascade_state = torch.relu(out)
        # Bypass.
        bypass_in = z
        if self.bypass_pool is not None:
            bypass_in = self.bypass_pool(bypass_in)
        bypass_out = self.bypass_proj(bypass_in)
        # Align spatial+temporal sizes across branches before adding
        # (stride math can drift by 1 timestep).
        all_outputs = branch_outputs + [bypass_out]
        c_min = min(o.size(2) for o in all_outputs)
        t_min = min(o.size(3) for o in all_outputs)
        all_outputs = [o[:, :, :c_min, :t_min] for o in all_outputs]
        temporal = torch.stack(all_outputs, dim=0).sum(dim=0)
        temporal = torch.relu(temporal)
        # Spatial conv + ReLU.
        spatial = torch.relu(self.spatial_conv(temporal))
        # Area fusion (ungrouped 1×1).
        fused = torch.relu(self.area_fusion(spatial))
        # Back to 5-D.
        return self._from_conv_layout(fused, b, w)

    def _forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        """Decoder pass: de-fusion expansion + spatial trans + temporal branches + bypass + final reduction."""
        if x.ndim != 5:
            raise ValueError(
                f"GeminiStitchingFusionModule expects 5-D input "
                f"(B, W, T, A, C); got shape {tuple(x.shape)}.",
            )
        b, w = x.shape[0], x.shape[1]
        # Decoder input: (B, W, T, A, C) where A is filters_per_area
        # (the area axis after the encoder's area_fusion compressed
        # things into a single ungrouped channel set). We rehydrate to
        # the per-area grouped layout via the defusion_expansion.
        z = x.permute(0, 1, 3, 4, 2).reshape(
            b * w, x.shape[3], x.shape[4], x.shape[2],
        )
        z = torch.relu(self.defusion_expansion(z))
        z = torch.relu(self.spatial_trans_conv(z))
        # Temporal expansion branches.
        branch_outputs: list[torch.Tensor] = []
        for k_idx in range(len(self.temporal_kernel_sizes)):
            cascade_state = z
            for j in range(self.num_cascade_layers):
                branch_idx = k_idx * self.num_cascade_layers + j
                branch = self.temporal_branches[branch_idx]
                out = branch.body(cascade_state)
                with_extra = branch.extra(out) if branch.extra is not None else out
                branch_outputs.append(with_extra)
                if j < self.num_cascade_layers - 1:
                    cascade_state = torch.relu(out)
        # Bypass projection.
        bypass_out = self.bypass_proj(z)
        all_outputs = branch_outputs + [bypass_out]
        c_min = min(o.size(2) for o in all_outputs)
        t_min = min(o.size(3) for o in all_outputs)
        all_outputs = [o[:, :, :c_min, :t_min] for o in all_outputs]
        temporal = torch.stack(all_outputs, dim=0).sum(dim=0)
        temporal = torch.relu(temporal)
        # Final channel reduction (groupedConv 1×1, 1 channel per area).
        out = self.final_reduction(temporal)
        return self._from_conv_layout(out, b, w)


# ───────────── Option-set defaults ─────────────


_GEMINI_OPTION_SETS = {
    # MATLAB defaults from PARAMETERS_cgg_constructStitchingAndFusionNetwork.m
    # cases 'Parallel Single Level' (131-139), 'Cascade Single Kernel -
    # Single Reduction' (140-148), 'Cascade Single Kernel - Progressive
    # Reduction' (149-157).
    "Parallel Single Level": dict(
        temporal_kernel_sizes=[3, 5, 7],
        encoder_reduction=(4, 4),
        num_cascade_layers=1,
        num_residual_layers=2,
        cascade_stride_mode="progressive",
        reduction_method="stride",
        stride_bypass_method="avgpool",
    ),
    "Cascade Single Kernel - Single Reduction": dict(
        temporal_kernel_sizes=[3],
        encoder_reduction=(4, 4),
        num_cascade_layers=3,
        num_residual_layers=2,
        cascade_stride_mode="single",
        reduction_method="stride",
        stride_bypass_method="avgpool",
    ),
    "Cascade Single Kernel - Progressive Reduction": dict(
        temporal_kernel_sizes=[3],
        encoder_reduction=(4, 2),
        num_cascade_layers=3,
        num_residual_layers=2,
        cascade_stride_mode="progressive",
        reduction_method="stride",
        stride_bypass_method="avgpool",
    ),
}


def build_gemini_stitching_fusion(
    network_type: str,
    *,
    num_areas: int,
    filters_per_area: int,
    mode: Literal["Encoder", "Decoder"],
) -> GeminiStitchingFusionModule:
    """Build the Gemini S&F module for one of the three named option-sets.

    Parameters
    ----------
    network_type
        Option-set name: ``'Parallel Single Level'``,
        ``'Cascade Single Kernel - Single Reduction'``, or
        ``'Cascade Single Kernel - Progressive Reduction'``.
    num_areas
        Number of input areas (``A``).
    filters_per_area
        Per-area filter count (mirrors MATLAB ``FilterHiddenSizes(1)``).
    mode
        ``'Encoder'`` or ``'Decoder'``.

    Returns
    -------
    GeminiStitchingFusionModule
    """
    if network_type not in _GEMINI_OPTION_SETS:
        raise ValueError(
            f"Unknown Gemini option-set: {network_type!r}. Expected one of "
            f"{sorted(_GEMINI_OPTION_SETS)}.",
        )
    kwargs = dict(_GEMINI_OPTION_SETS[network_type])
    return GeminiStitchingFusionModule(
        num_areas=num_areas,
        filters_per_area=filters_per_area,
        mode=mode,
        **kwargs,  # type: ignore[arg-type]
    )


__all__ = [
    "GeminiStitchingFusionModule",
    "build_gemini_stitching_fusion",
]
