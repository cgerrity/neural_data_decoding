"""Architecture-string registry — port of ``cgg_constructNetworkArchitecture.m``.

The MATLAB pipeline selects an entire encoder/decoder topology via a single
``cfg.ModelName`` string. The `case` branches in
``PARAMETERS_cgg_constructNetworkArchitecture.m`` each set ~10 flag fields
(``IsSimple``, ``Transform``, ``Dropout``, ``WantNormalization``,
``Activation``, ``IsVariational``, ``needReshape``,
``OutputFullyConnected``, ``BottleNeckDepth``, plus conv-only fields
like ``FilterSizes``, ``WantSplitAreas``, ``Stride``,
``DownSampleMethod``, ``UpSampleMethod``, ``WantResnet``,
``FinalActivation``).

This module ports that name → flag-bundle mapping to Python as an
:class:`ArchitectureSpec` dataclass plus a registry indexed by the
exact MATLAB ``ModelName`` string. Per Critical Note #14, milestones
A/B/C populated only the variants their pipelines exercised; CC.1
extends the registry to cover the full SLURM-sweep parameter space:

SLURM sweep (``SLURMPARAMETERS_cgg_runAutoEncoder_v2.m`` lines
145-179): ``'Feedforward'``, ``'LSTM'``, ``'Convolutional'``,
``'Resnet'``, ``'Multi-Filter Convolutional'``, ``'Logistic
Regression'``, ``'PCA'``.

Active production (``PARAMETERS_OPTIMAL_*.m``): ``'GRU'``.

Per the project directive, all SLURM/production architectures must be
obtainable by parameter switching against the existing Python
builders. The registry resolves a name to a spec; the spec's flag
combination then drives the existing composite/encoder/decoder
construction (no new builder code per architecture). The ``'PCA'``
entry is backed by the CC.2 PCA backbone (``PCAEncoder`` in
``models/layers/pca.py``, registered as the ``'PCA'`` encoder builder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Per-architecture flag bundle from ``cgg_constructNetworkArchitecture.m``.

    The MATLAB ``cfg`` struct ends up with every local variable in the
    function as a field (``w = whos; for a = 1:length(w); cfg.(w(a).name)
    = eval(w(a).name); end``); we capture the same fields as a frozen
    dataclass.

    Fields are grouped: the Simple-branch common fields are required;
    the Convolutional-only fields default to ``None`` and are only set
    for ``IsSimple=false`` architectures.

    Attributes
    ----------
    is_simple
        MATLAB ``IsSimple``. ``True`` → use the Simple GRU/LSTM/FF
        builder; ``False`` → use the Convolutional builder.
    is_variational
        MATLAB ``IsVariational``. Enables the variational
        composite (encoder + sampling + decoder + classifier).
    transform
        MATLAB ``Transform``. One of ``'Feedforward'``, ``'GRU'``,
        ``'LSTM'``, ``'PCA'``. For Convolutional architectures this
        names the bottleneck / classifier transform — the conv
        operations are external.
    activation
        MATLAB ``Activation``. ``'ReLU'`` / ``'Leaky ReLU'`` /
        ``'Softplus'`` / ``'GeLU'`` / ``''``.
    dropout
        MATLAB ``Dropout`` (block dropout probability).
    want_normalization
        MATLAB ``WantNormalization``. ``False`` / ``True`` (BatchNorm)
        / ``'Batch'`` / ``'Instance'`` / ``'Layer'``.
    need_reshape
        MATLAB ``needReshape``. Whether to add the decoder-output
        ``Function_Decoder`` reshape (CBTSS layout).
    output_fully_connected
        MATLAB ``OutputFullyConnected``. Whether the decoder's last
        layer is an FC projection to flat-output dim.
    bottleneck_depth
        MATLAB ``BottleNeckDepth``. Number of stacked bottleneck blocks.
    final_activation
        MATLAB ``FinalActivation`` — used only by Convolutional
        architectures (``'Convolutional'``, ``'Tanh'``, ``'Sigmoid'``,
        ``'None'``).
    filter_sizes
        MATLAB ``FilterSizes``. Either a list of ints (multi-filter,
        e.g. ``[3, 5, 7]``) or a single int. Convolutional-only.
    filter_size_percent
        MATLAB ``FilterSizePercent``. Multiplier(s) for filter sizes
        as a fraction of input spatial dims. Convolutional-only.
    want_split_areas
        MATLAB ``WantSplitAreas``. Per-area grouped conv vs. shared.
    stride
        MATLAB ``Stride``. Down/up-sample factor along the temporal axis.
    down_sample_method
        MATLAB ``DownSampleMethod``. ``'MaxPool'`` / ``'Same - Stride'``.
    up_sample_method
        MATLAB ``UpSampleMethod``. ``'Transpose Convolution'`` / ``'None'``.
    want_resnet
        MATLAB ``WantResnet``. Per-level residual connection.
    encoder_output_type
        MATLAB ``EncoderOutputType``. ``'Deterministic'`` / ``'Stochastic'``.
    bottleneck_normalization
        MATLAB ``BottleNeckNormalization``. Defaults to ``'Layer'``.
    repetitions_per_block
        MATLAB ``RepetitionsPerBlock``. Conv blocks per pyramid level.
    want_pre_activation
        MATLAB ``WantPreActivation``. Move activation before convolution.
    want_post_decoder_convolution
        MATLAB ``WantPostDecoderConvolution``. Trailing decoder conv.
    want_pre_decoder_convolution
        MATLAB ``WantPreDecoderConvolution``. Leading decoder conv.
    want_learnable_offset
        MATLAB ``WantLearnableOffset``. Enable learnable additive
        augmentation (gated; ports ``cgg_lossOffsetAndScale.m`` —
        CC.6).
    want_learnable_scale
        MATLAB ``WantLearnableScale``. Enable learnable multiplicative
        augmentation (CC.6).
    """

    # Simple-branch common fields (required).
    is_simple: bool
    is_variational: bool
    transform: str
    activation: str
    dropout: float
    want_normalization: Any  # bool | str
    need_reshape: bool
    output_fully_connected: bool
    bottleneck_depth: int

    # Decoder activation choice (Convolutional only; ignored by Simple).
    final_activation: Optional[str] = None

    # Convolutional-branch fields (only set when ``is_simple=False``).
    filter_sizes: Optional[list] = None
    filter_size_percent: Optional[list] = None
    want_split_areas: Optional[bool] = None
    stride: Optional[int] = None
    down_sample_method: Optional[str] = None
    up_sample_method: Optional[str] = None
    want_resnet: Optional[bool] = None

    # Module-wide defaults (overrideable per architecture).
    encoder_output_type: str = "Deterministic"
    bottleneck_normalization: str = "Layer"
    repetitions_per_block: int = 1
    want_pre_activation: bool = False
    want_post_decoder_convolution: bool = False
    want_pre_decoder_convolution: bool = False
    want_learnable_offset: bool = False
    want_learnable_scale: bool = False


# ───────────────────────── Registry ─────────────────────────


# Architecture entries. Names match MATLAB ``cfg.ModelName`` exactly
# (case-sensitive). Scope: the 7 SLURM-sweep variants
# (SLURMPARAMETERS_cgg_runAutoEncoder_v2.m lines 145-179) plus
# ``'GRU'`` (active production in PARAMETERS_OPTIMAL_*.m). Additional
# entries can be added when SLURM sweep coverage expands.
_ARCH_SPECS: dict[str, ArchitectureSpec] = {
    "Logistic Regression": ArchitectureSpec(
        is_simple=True, is_variational=False, transform="Feedforward",
        activation="", dropout=0.0, want_normalization=False,
        need_reshape=True, output_fully_connected=False, bottleneck_depth=1,
    ),
    "Feedforward": ArchitectureSpec(
        is_simple=True, is_variational=False, transform="Feedforward",
        activation="ReLU", dropout=0.0, want_normalization=False,
        need_reshape=True, output_fully_connected=True, bottleneck_depth=1,
    ),
    "GRU": ArchitectureSpec(
        is_simple=True, is_variational=False, transform="GRU",
        activation="", dropout=0.0, want_normalization=False,
        need_reshape=True, output_fully_connected=True, bottleneck_depth=1,
    ),
    "LSTM": ArchitectureSpec(
        is_simple=True, is_variational=False, transform="LSTM",
        activation="", dropout=0.0, want_normalization=False,
        need_reshape=True, output_fully_connected=True, bottleneck_depth=1,
    ),
    # CC.2 — done. The PCA architecture uses ``Transform='PCA'``, backed
    # by the ``PCAEncoder`` backbone (``models/layers/pca.py``, registered
    # as the ``'PCA'`` encoder builder via ``build_pca_encoder``);
    # resolving this entry returns the spec and composite construction
    # succeeds.
    "PCA": ArchitectureSpec(
        is_simple=False, is_variational=False, transform="PCA",
        activation="", dropout=0.0, want_normalization=False,
        need_reshape=False, output_fully_connected=False, bottleneck_depth=1,
    ),
    # Convolutional variants — buildable via the
    # PerWindowConvolutionalCoder (CC.5 Phase 2) by selecting
    # appropriate parameter combinations. No new builder code required.
    "Convolutional": ArchitectureSpec(
        is_simple=False, is_variational=False, transform="Feedforward",
        activation="Leaky ReLU", dropout=0.0, want_normalization=False,
        need_reshape=False, output_fully_connected=False, bottleneck_depth=1,
        final_activation="Convolutional",
        filter_sizes=[[4, 20]], filter_size_percent=[0.3],
        want_split_areas=False, stride=2,
        down_sample_method="MaxPool",
        up_sample_method="Transpose Convolution",
        want_resnet=False,
    ),
    "Resnet": ArchitectureSpec(
        is_simple=False, is_variational=False, transform="Feedforward",
        activation="Leaky ReLU", dropout=0.0, want_normalization=False,
        need_reshape=False, output_fully_connected=False, bottleneck_depth=1,
        final_activation="Convolutional",
        filter_sizes=[[4, 20]], filter_size_percent=[0.3],
        want_split_areas=False, stride=2,
        down_sample_method="MaxPool",
        up_sample_method="Transpose Convolution",
        want_resnet=True,  # only difference from 'Convolutional'
    ),
    # Multi-filter — buildable via the GeminiStitchingFusionModule's
    # parallel multi-scale branches (CC.5 Phase 3). The Gemini
    # 'Parallel Single Level' variant with TemporalKernelSizes=[3,5,7]
    # mirrors this architecture's multi-filter pattern. No new builder
    # code required.
    "Multi-Filter Convolutional": ArchitectureSpec(
        is_simple=False, is_variational=False, transform="Feedforward",
        activation="Leaky ReLU", dropout=0.0, want_normalization=False,
        need_reshape=False, output_fully_connected=False, bottleneck_depth=1,
        final_activation="Convolutional",
        filter_sizes=[3, 5, 7], filter_size_percent=[0.2, 0.3, 0.4],
        want_split_areas=False, stride=2,
        down_sample_method="MaxPool",
        up_sample_method="Transpose Convolution",
        want_resnet=False,
    ),
}


def resolve_architecture(name: str) -> ArchitectureSpec:
    """Resolve a MATLAB ``ModelName`` string to its flag bundle.

    Mirrors the ``switch`` in
    ``PARAMETERS_cgg_constructNetworkArchitecture.m``.

    Parameters
    ----------
    name
        Exact MATLAB ``ModelName`` value (case-sensitive). See
        :func:`list_architectures` for the registered names.

    Returns
    -------
    ArchitectureSpec
        The architecture's flag bundle.

    Raises
    ------
    ValueError
        If ``name`` is not registered. The error lists known names to
        aid discoverability.
    """
    if name not in _ARCH_SPECS:
        raise ValueError(
            f"Unknown architecture {name!r}. Registered: "
            f"{sorted(_ARCH_SPECS)}.",
        )
    return _ARCH_SPECS[name]


def list_architectures() -> list[str]:
    """List all registered ``ModelName`` strings, sorted alphabetically."""
    return sorted(_ARCH_SPECS)


def has_architecture(name: str) -> bool:
    """``True`` if ``name`` is a registered architecture."""
    return name in _ARCH_SPECS


__all__ = [
    "ArchitectureSpec",
    "has_architecture",
    "list_architectures",
    "resolve_architecture",
]
