"""Stitching + Fusion bridges — port of ``cgg_constructStitchingAndFusionNetwork.m``.

Cross-area fusion bridges that sit **before** the encoder (pre-encoder —
projects raw multi-area input into a unified fusion space) and **after**
the decoder (post-decoder — projects the decoder's reconstruction back
into the raw multi-area output space).

When configured, the encoder operates in ``cross_area_fusion_size``-channel
space instead of ``in_features``; the post-decoder bridge converts that
back to ``in_features`` for the reconstruction loss.

Five MATLAB option-sets are defined in
``PARAMETERS_cgg_constructStitchingAndFusionNetwork.m``:

* **Feedforward** — a single per-timestep ``Linear`` projection on each
  side (see :mod:`.feedforward`; Phase 1).
* **Default** — convolutional cross-area fusion with ``WantSplitAreas``,
  Leaky ReLU activations (Phase 2 — pending).
* **Parallel Single Level**, **Cascade Single Kernel - Single
  Reduction**, **Cascade Single Kernel - Progressive Reduction** — Gemini
  cascaded multi-area fusion variants (Phase 3 — pending).

CrossAreaFusionSize convention
------------------------------
Per ``cgg_constructNetworkArchitecture.m`` line 125, the fusion space
dimension is derived as ``HiddenSizeAutoEncoder(1) * 2``; the Python
caller computes the same from ``hidden_sizes[0] * 2`` and passes it via
``cross_area_fusion_size``.
"""

from neural_data_decoding.models.stitching_fusion.convolutional import (
    PerWindowConvolutionalCoder,
)
from neural_data_decoding.models.stitching_fusion.feedforward import (
    FeedforwardStitchingFusion,
    build_stitching_fusion,
)

__all__ = [
    "FeedforwardStitchingFusion",
    "PerWindowConvolutionalCoder",
    "build_stitching_fusion",
]
