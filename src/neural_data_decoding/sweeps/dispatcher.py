"""Sweep dispatcher — Python port of ``SLURMPARAMETERS_cgg_runAutoEncoder_v2.m``.

The MATLAB sweep file is a two-level table indexed by
``(SLURMChoice, SLURMIDX)``. ``SLURMChoice`` (1-15) selects which
parameter family is being varied; ``SLURMIDX`` (1-10) picks a specific
override bundle within that family. The "otherwise" branch is the
no-override base run across all folds.

This module flattens both axes into a **single integer sweep index**
1..N so a SLURM array can dispatch one config per array task without
needing to know about the MATLAB choice/idx pairing. Each
:class:`SweepEntry` carries the original MATLAB description, the cfg
override bundle (translated to Python field names), and a ``notes``
tuple flagging partial-support fields or known caveats.

Field-name translation
----------------------
The MATLAB fields use CamelCase; the Python cfg uses snake_case.
:data:`_MATLAB_TO_PYTHON_FIELD` is the authoritative map (covers every
field that appears in any sweep entry). See
``configs/base.yaml`` for the full list of valid Python cfg fields and
``src/neural_data_decoding/sweeps/parameter_coverage.py`` for per-field
support status.

Convention for dynamic MATLAB values
------------------------------------
Some MATLAB entries reference ``cfg.HiddenSizes(end)`` — the bottleneck
dim of the resolved base config (250 in the Optimal/base default). The
Python entries store the **resolved** literal value (e.g. ``250``)
rather than a callback. Users who run a non-default base config can
re-override via the CLI ``--override`` flag.

Examples
--------
>>> from neural_data_decoding.sweeps import lookup
>>> entry = lookup(1)                    # First sweep entry
>>> entry.description
'Feedforward Network'
>>> entry.overrides["model_name"]
'Feedforward'
>>> entry.matlab_choice, entry.matlab_idx
(1, 1)
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


# MATLAB → Python cfg field translation. Every key that appears in a
# sweep entry's overrides dict MUST appear here (so a CamelCase typo
# in an entry definition is caught at module import time).
_MATLAB_TO_PYTHON_FIELD: dict[str, str] = {
    "Fold": "fold",
    "ModelName": "model_name",
    "DataWidth": "data_width",
    "WindowStride": "window_stride",
    "HiddenSizes": "hidden_sizes",
    "InitialLearningRate": "initial_learning_rate",
    "WeightReconstruction": "weight_reconstruction",
    "WeightKL": "weight_kl",
    "WeightClassification": "weight_classification",
    "MiniBatchSize": "mini_batch_size",
    "Subset": "subset",
    "Target": "target",
    "Epoch": "epoch",
    "WeightedLoss": "weighted_loss",
    "GradientThreshold": "gradient_threshold",
    "ClassifierName": "classifier_name",
    "ClassifierHiddenSize": "classifier_hidden_size",
    "STDChannelOffset": "std_channel_offset",
    "STDWhiteNoise": "std_white_noise",
    "STDRandomWalk": "std_random_walk",
    "NumEpochsAutoEncoder": "num_epochs_autoencoder",
    "NumEpochsFull": "num_epochs_full",
    "Optimizer": "optimizer",
    "Normalization": "normalization",
    "LossType_Decoder": "loss_type_decoder",
    "LossType_Classifier": "loss_type_classifier",
    "maxworkerMiniBatchSize": "max_worker_mini_batch_size",
    "L2Factor": "l2_factor",
    "Dropout": "dropout",
    "WantNormalization": "want_normalization",
    "Activation": "activation",
    "IsVariational": "is_variational",
    "BottleNeckDepth": "bottle_neck_depth",
    "WantSaveOptimalNet": "want_save_optimal_net",
    "EncoderOutputType": "encoder_output_type",
    "GradientClipType": "gradient_clip_type",
    "MultipleInstanceLearningType": "multiple_instance_learning_type",
    "DynamicParameterSet": "dynamic_parameter_set",
    "StitchingAndFusionLayer": "stitching_and_fusion_layer",
    "StartEndPercent": "start_end_percent",
    "wantStratifiedPartition": "want_stratified_partition",
    "STDTimeShift": "std_time_shift",
    "WantSeparateTimeShift": "want_separate_time_shift",
    "WeightOffsetAndScale": "weight_offset_and_scale",
    "RescaleLossEpoch": "rescale_loss_epoch",
    "WeightConfidence": "weight_confidence",
    "ConfidenceType": "confidence_type",
}

# The MATLAB-resolved value of ``cfg.HiddenSizes(end)`` when running
# against the Optimal/base config (``hidden_sizes: [1000, 500, 250]``).
# Several SC 1 entries set ``HiddenSizes = [8, 16, 32, cfg.HiddenSizes(end)]``.
_BOTTLENECK_DIM_DEFAULT = 250

# Conventional partial-support note strings (de-duplicated).
_NOTE_BOTTLENECK_GT_1 = (
    "bottle_neck_depth > 1: cfg passes through but the variational head "
    "is single-stage; behavior may differ from MATLAB."
)
_NOTE_NORMALIZATION_STRING = (
    "Specific data-normalization string variants are only partially "
    "supported in Python (see sweeps/parameter_coverage.py)."
)
_NOTE_LOSS_TYPE_NONE = (
    "loss_type_decoder='None' disables the reconstruction loss kernel "
    "entirely — only safe with a non-zero classification weight."
)


@dataclass(frozen=True, slots=True)
class SweepEntry:
    """One sweep configuration — flat index + override bundle.

    Parameters
    ----------
    sweep_index
        1-based flat index across the whole sweep table; what a SLURM
        array task receives.
    matlab_choice
        MATLAB ``SLURMChoice`` (1-15).
    matlab_idx
        MATLAB ``SLURMIDX`` (1-based; usually 1-10).
    description
        MATLAB ``Description`` string — used in run banners and SLURM
        output filenames.
    overrides
        Python cfg field → value bundle. Keys are snake_case (see
        ``configs/base.yaml``); values are the override that will be
        merged on top of the base config.
    notes
        Caveat strings for partial-support fields, MATLAB FIXME flags,
        or known behavior differences. Empty tuple when the entry is
        fully supported.
    """

    sweep_index: int
    matlab_choice: int
    matlab_idx: int
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


# ----------------------------------------------------------------------
# Raw entry definitions, organized by MATLAB SLURMChoice block.
#
# Each row is ``(choice, idx, description, overrides_with_matlab_keys, notes)``.
# Overrides are written with MATLAB CamelCase keys here so the side-by-
# side mapping to SLURMPARAMETERS_cgg_runAutoEncoder_v2.m is obvious;
# the module-init pass translates them to Python snake_case via
# :data:`_MATLAB_TO_PYTHON_FIELD`.
# ----------------------------------------------------------------------


def _build_raw_entries() -> list[tuple[int, int, str, dict[str, Any], tuple[str, ...]]]:
    """Return the raw sweep entries in MATLAB SLURMChoice/SLURMIDX order.

    Factored into a function so the long literal definition stays out
    of the module top-level (keeps interrogate / pyright reports clean
    and makes the dispatcher easier to skim).
    """
    raw: list[tuple[int, int, str, dict[str, Any], tuple[str, ...]]] = []

    # ── SLURMChoice 1: encoder architecture + classifier + AE epochs ──
    raw += [
        (1, 1, "Feedforward Network", {
            "ModelName": "Feedforward", "WantNormalization": True,
        }, ()),
        (1, 2, "LSTM Network", {"ModelName": "LSTM"}, ()),
        (1, 3, "Convolutional Network - Gradient Accumulation size 25", {
            "ModelName": "Convolutional",
            "maxworkerMiniBatchSize": 25,
            "HiddenSizes": [8, 16, 32, _BOTTLENECK_DIM_DEFAULT],
            "WantNormalization": "Instance",
        }, ()),
        (1, 4, "Resnet Network - Gradient Accumulation size 20", {
            "ModelName": "Resnet",
            "maxworkerMiniBatchSize": 20,
            "HiddenSizes": [8, 16, 32, _BOTTLENECK_DIM_DEFAULT],
            "WantNormalization": "Instance",
        }, ()),
        (1, 5, "Multi-Filter Network - Gradient Accumulation size 25", {
            "ModelName": "Multi-Filter Convolutional",
            "maxworkerMiniBatchSize": 25,
            "HiddenSizes": [8, 16, 32, _BOTTLENECK_DIM_DEFAULT],
            "WantNormalization": "Instance",
        }, ()),
        (1, 6, "Self-supervised epochs - 10", {"NumEpochsAutoEncoder": 10}, ()),
        (1, 7, "Self-supervised epochs - 50", {"NumEpochsAutoEncoder": 50}, ()),
        (1, 8, "Self-supervised epochs - 100", {"NumEpochsAutoEncoder": 100}, ()),
        (1, 9, "Classifier - GRU", {"ClassifierName": "Deep GRU - Dropout 0.5"}, ()),
        (1, 10, "Classifier - Feedforward", {
            "ClassifierName": "Deep Feedforward - Dropout 0.5",
        }, ()),
    ]

    # ── SLURMChoice 2: L2 sweep + Norm + accumulation ──
    raw += [
        (2, 1, "L2 Factor - 1", {"L2Factor": 1.0}, ()),
        (2, 2, "L2 Factor - 1e-1", {"L2Factor": 1e-1}, ()),
        (2, 3, "L2 Factor - 1e-2", {"L2Factor": 1e-2}, ()),
        (2, 4, "L2 Factor - 1e-3", {"L2Factor": 1e-3}, ()),
        (2, 5, "L2 Factor - 1e-5", {"L2Factor": 1e-5}, ()),
        (2, 6, "L2 Factor - 1e-6", {"L2Factor": 1e-6}, ()),
        (2, 7, "L2 Factor - 1e-7", {"L2Factor": 1e-7}, ()),
        (2, 8, "Layer Normalization", {"WantNormalization": True}, ()),
        (2, 9, "Gradient Accumulation size 1", {"maxworkerMiniBatchSize": 1}, ()),
        (2, 10, "Gradient Accumulation size 50", {"maxworkerMiniBatchSize": 50}, ()),
    ]

    # ── SLURMChoice 3: DataWidth + WindowStride + WeightedLoss ──
    raw += [
        (3, 1, "Data Width 200", {"DataWidth": 200, "WindowStride": 100}, ()),
        (3, 2, "Data Width 50", {"DataWidth": 50, "WindowStride": 25}, ()),
        (3, 3, "Data Width 20", {"DataWidth": 20, "WindowStride": 10}, ()),
        (3, 4, "Data Width 10", {"DataWidth": 10, "WindowStride": 5}, ()),
        (3, 5, "Data Width 4", {"DataWidth": 4, "WindowStride": 2}, ()),
        (3, 6, "Stride 1 - with Gradient Accumulation size 10", {
            "WindowStride": 1, "maxworkerMiniBatchSize": 10,
        }, ()),
        (3, 7, "Stride 25", {"WindowStride": 25}, ()),
        (3, 8, "Stride 75", {"WindowStride": 75}, ()),
        (3, 9, "Stride 100", {"WindowStride": 100}, ()),
        (3, 10, "Unweighted Loss", {"WeightedLoss": ""}, ()),
    ]

    # ── SLURMChoice 4: encoder HiddenSizes variants ──
    raw += [
        (4, 1, "Hidden Sizes - [2000,1000,500] - 3 layers ~ Higher", {
            "HiddenSizes": [2000, 1000, 500],
        }, ()),
        (4, 2, "Hidden Sizes - [4000,2000,1000] - 3 layers ~ Much Higher", {
            "HiddenSizes": [4000, 2000, 1000],
        }, ()),
        (4, 3, "Hidden Sizes - [500,250,100] - 3 layers ~ Lower", {
            "HiddenSizes": [500, 250, 100],
        }, ()),
        (4, 4, "Hidden Sizes - [2000,1000,500,250] - 4 layers ~ Higher", {
            "HiddenSizes": [2000, 1000, 500, 250],
        }, ()),
        (4, 5, "Hidden Sizes - [1000,500,250,100] - 4 layers ~ Lower", {
            "HiddenSizes": [1000, 500, 250, 100],
        }, ()),
        (4, 6, "Hidden Sizes - [4000,2000,1000,500,250] - 5 layers ~ Higher", {
            "HiddenSizes": [4000, 2000, 1000, 500, 250],
        }, ()),
        (4, 7, "Hidden Sizes - [1000,500,250,100,50] - 5 layers ~ Lower", {
            "HiddenSizes": [1000, 500, 250, 100, 50],
        }, ()),
        (4, 8, "Hidden Sizes - [500,250] - 2 layers ~ Lower", {
            "HiddenSizes": [500, 250],
        }, ()),
        (4, 9, "Hidden Sizes - [1000,500] - 2 layers ~ Higher", {
            "HiddenSizes": [1000, 500],
        }, ()),
        (4, 10, "Hidden Sizes - 1000 - 1 layer", {"HiddenSizes": [1000]}, ()),
    ]

    # ── SLURMChoice 5: ClassifierHiddenSize variants ──
    raw += [
        (5, 1, "Classifier Hidden Sizes - [500,250,100] - 3 layers ~ Higher", {
            "ClassifierHiddenSize": [500, 250, 100],
        }, ()),
        (5, 2, "Classifier Hidden Sizes - [100,50,25] - 3 layers ~ Lower", {
            "ClassifierHiddenSize": [100, 50, 25],
        }, ()),
        (5, 3, "Classifier Hidden Sizes - [50,25,10] - 3 layers ~ Much Lower", {
            "ClassifierHiddenSize": [50, 25, 10],
        }, ()),
        (5, 4, "Classifier Hidden Sizes - [500,250,100,50] - 4 layers ~ Higher", {
            "ClassifierHiddenSize": [500, 250, 100, 50],
        }, ()),
        (5, 5, "Classifier Hidden Sizes - [250,100,50,25] - 4 layers ~ Lower", {
            "ClassifierHiddenSize": [250, 100, 50, 25],
        }, ()),
        (5, 6, "Classifier Hidden Sizes - [1000,500,250,100,50] - 5 layers ~ Higher", {
            "ClassifierHiddenSize": [1000, 500, 250, 100, 50],
        }, ()),
        (5, 7, "Classifier Hidden Sizes - [250,100,50,25,10] - 5 layers ~ Higher", {
            "ClassifierHiddenSize": [250, 100, 50, 25, 10],
        }, ()),
        (5, 8, "Classifier Hidden Sizes - [250,100] - 2 layers ~ Higher", {
            "ClassifierHiddenSize": [250, 100],
        }, ()),
        (5, 9, "Classifier Hidden Sizes - [100,50] - 2 layers ~ Lower", {
            "ClassifierHiddenSize": [100, 50],
        }, ()),
        (5, 10, "Classifier Hidden Sizes - 250 - 1 layer", {
            "ClassifierHiddenSize": [250],
        }, ()),
    ]

    # ── SLURMChoice 6: MiniBatchSize + InitialLearningRate ──
    raw += [
        (6, 1, "Mini-Batch Size - 10", {"MiniBatchSize": 10}, ()),
        (6, 2, "Mini-Batch Size - 25", {"MiniBatchSize": 25}, ()),
        (6, 3, "Mini-Batch Size - 50", {"MiniBatchSize": 50}, ()),
        (6, 4, "Mini-Batch Size - 200", {
            "MiniBatchSize": 200, "maxworkerMiniBatchSize": 200,
        }, ()),
        (6, 5, "Mini-Batch Size - 400", {
            "MiniBatchSize": 400, "maxworkerMiniBatchSize": 400,
        }, ()),
        (6, 6, "Initial Learnging Rate - 5e-2", {"InitialLearningRate": 5e-2}, ()),
        (6, 7, "Initial Learnging Rate - 5e-3", {"InitialLearningRate": 5e-3}, ()),
        (6, 8, "Initial Learnging Rate - 1e-3", {"InitialLearningRate": 1e-3}, ()),
        (6, 9, "Initial Learnging Rate - 5e-4", {"InitialLearningRate": 5e-4}, ()),
        (6, 10, "Initial Learnging Rate - 1e-4", {"InitialLearningRate": 1e-4}, ()),
    ]

    # ── SLURMChoice 7: WeightReconstruction / WeightKL / WeightClassification ──
    raw += [
        (7, 1, "Reconstruction Weight - 1", {"WeightReconstruction": 1}, ()),
        (7, 2, "KL Weight - 1", {"WeightKL": 1}, ()),
        (7, 3, "Classification Weight - 1", {"WeightClassification": 1}, ()),
        (7, 4, "Reconstruction Weight - 2", {"WeightReconstruction": 2}, ()),
        (7, 5, "Reconstruction Weight - 10", {"WeightReconstruction": 10}, ()),
        (7, 6, "Reconstruction Weight - 100", {"WeightReconstruction": 100}, ()),
        (7, 7, "Reconstruction Weight - 1000", {"WeightReconstruction": 1000}, ()),
        (7, 8, "KL Weight - 1e-4", {"WeightKL": 1e-4}, ()),
        (7, 9, "KL Weight - 1e-5", {"WeightKL": 1e-5}, ()),
        (7, 10, "KL Weight - 1e-6", {"WeightKL": 1e-6}, ()),
    ]

    # ── SLURMChoice 8: Optimizer + Normalization + LossType_Decoder ──
    raw += [
        (8, 1, "Optimizer - SGD", {"Optimizer": "SGD"}, ()),
        (8, 2, "Data Normalization - None", {"Normalization": "None"}, ()),
        (8, 3, "Data Normalization - Channel - Z-Score - Global - MinMax - [-1,1]",
         {"Normalization": "Channel - Z-Score - Global - MinMax - [-1,1]"},
         (_NOTE_NORMALIZATION_STRING,)),
        (8, 4, "Data Normalization - Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered",
         {"Normalization": "Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered"},
         (_NOTE_NORMALIZATION_STRING,)),
        (8, 5, "Data Normalization - Channel - Z-Score",
         {"Normalization": "Channel - Z-Score"},
         (_NOTE_NORMALIZATION_STRING,)),
        (8, 6, "Data Normalization - Global - MinMax - [-1,1]",
         {"Normalization": "Global - MinMax - [-1,1]"},
         (_NOTE_NORMALIZATION_STRING,)),
        (8, 7, "Decoder Loss Type - MAE", {"LossType_Decoder": "MAE"}, ()),
        (8, 8, "No Decoder", {"LossType_Decoder": "None"}, (_NOTE_LOSS_TYPE_NONE,)),
        (8, 9, "Not Variational", {"IsVariational": False}, ()),
        (8, 10, "Gradient Accumulation size 25", {"maxworkerMiniBatchSize": 25}, ()),
    ]

    # ── SLURMChoice 9: Dropout + BottleNeckDepth + GradientThreshold ──
    raw += [
        (9, 1, "Dropout - 0", {"Dropout": 0.0}, ()),
        (9, 2, "Dropout - 0.25", {"Dropout": 0.25}, ()),
        (9, 3, "Dropout - 0.9", {"Dropout": 0.9}, ()),
        (9, 4, "Dropout - 0.75", {"Dropout": 0.75}, ()),
        (9, 5, "Bottleneck Depth - 2", {"BottleNeckDepth": 2}, (_NOTE_BOTTLENECK_GT_1,)),
        (9, 6, "Bottleneck Depth - 3", {"BottleNeckDepth": 3}, (_NOTE_BOTTLENECK_GT_1,)),
        (9, 7, "Gradient Threshold - 0.1", {"GradientThreshold": 0.1}, ()),
        (9, 8, "Gradient Threshold - 1", {"GradientThreshold": 1}, ()),
        (9, 9, "Gradient Threshold - 10", {"GradientThreshold": 10}, ()),
        (9, 10, "Gradient Threshold - 1000", {"GradientThreshold": 1000}, ()),
    ]

    # ── SLURMChoice 10: accumulation + bottleneck + gradient + LR + weight triples ──
    raw += [
        (10, 1, "Gradient Accumulation size 10", {"maxworkerMiniBatchSize": 10}, ()),
        (10, 2, "Bottleneck Depth - 4", {"BottleNeckDepth": 4}, (_NOTE_BOTTLENECK_GT_1,)),
        (10, 3, "Gradient Threshold - 10000", {"GradientThreshold": 10000}, ()),
        (10, 4, "Gradient Threshold - 0.01", {"GradientThreshold": 0.01}, ()),
        (10, 5, "Initial Learnging Rate - 0.1", {"InitialLearningRate": 0.1}, ()),
        (10, 6, "Initial Learnging Rate - 0.5", {"InitialLearningRate": 0.5}, ()),
        (10, 7, "Weights Ratio - 1:1e-2:1e-2 (R:C:K)", {
            "WeightReconstruction": 1, "WeightKL": 1e-2, "WeightClassification": 1e-2,
        }, ()),
        (10, 8, "Weights Ratio - 1e-4:1e-6:1e-6 (R:C:K)", {
            "WeightReconstruction": 1e-4, "WeightKL": 1e-6, "WeightClassification": 1e-6,
        }, ()),
        (10, 9, "Weights Ratio - 1:10:1e-4 (R:C:K)", {
            "WeightReconstruction": 1, "WeightKL": 1e-4, "WeightClassification": 10,
        }, ()),
        (10, 10, "Weights Ratio - 1:100:1e-4 (R:C:K)", {
            "WeightReconstruction": 1, "WeightKL": 1e-4, "WeightClassification": 100,
        }, ()),
    ]

    # ── SLURMChoice 11: model variants + curriculum + MIL ──
    raw += [
        (11, 1, "Logistic Regression", {"ModelName": "Logistic Regression"}, ()),
        (11, 2, "Hidden Sizes - [250,100,50] - 3 layers ~ Much Lower", {
            "HiddenSizes": [250, 100, 50],
        }, ()),
        (11, 3, "Small Network with Large Classification Weight", {
            "HiddenSizes": [250],
            "ClassifierHiddenSize": [100],
            "WeightReconstruction": 1,
            "WeightKL": 1e-4,
            "WeightClassification": 10_000,
        }, ()),
        (11, 4, "PCA", {"ModelName": "PCA"}, ()),
        (11, 5, "Stochastic Encoder", {"EncoderOutputType": "Stochastic"}, ()),
        (11, 6, "Stochastic Encoder with Global Gradient Clip", {
            "EncoderOutputType": "Stochastic", "GradientClipType": "Global",
        }, ()),
        (11, 7, "Multiple Instance Learning", {
            "MultipleInstanceLearningType": "MIL",
        }, ()),
        (11, 8, "Soft Three-Stage Curriculum with Multiple Instance Learning", {
            "MultipleInstanceLearningType": "MIL",
            "DynamicParameterSet": "Soft Three-Stage Curriculum",
        }, ()),
        (11, 9, "Soft Three-Stage Curriculum", {
            "MultipleInstanceLearningType": "None",
            "DynamicParameterSet": "Soft Three-Stage Curriculum",
        }, ()),
        (11, 10, "Feedforward Network with Soft Three-Stage Curriculum and Multiple Instance Learning", {
            "ModelName": "Feedforward",
            "WantNormalization": True,
            "MultipleInstanceLearningType": "MIL",
            "DynamicParameterSet": "Soft Three-Stage Curriculum",
        }, ()),
    ]

    # ── SLURMChoice 12: WeightClassification / WeightKL / WeightReconstruction ──
    raw += [
        (12, 1, "Classification Weight - 2", {"WeightClassification": 2}, ()),
        (12, 2, "Classification Weight - 10", {"WeightClassification": 10}, ()),
        (12, 3, "Classification Weight - 100", {"WeightClassification": 100}, ()),
        (12, 4, "Classification Weight - 1000", {"WeightClassification": 1000}, ()),
        (12, 5, "Classification Weight - 0.1", {"WeightClassification": 0.1}, ()),
        (12, 6, "KL Weight - 0.1", {"WeightKL": 0.1}, ()),
        (12, 7, "KL Weight - 0.01", {"WeightKL": 0.01}, ()),
        (12, 8, "KL Weight - 1e-3", {"WeightKL": 1e-3}, ()),
        (12, 9, "KL Weight - 10", {"WeightKL": 10}, ()),
        (12, 10, "Reconstruction Weight - 0.1", {"WeightReconstruction": 0.1}, ()),
    ]

    # ── SLURMChoice 13: NumEpochsAutoEncoder + WeightClassification + WeightKL ──
    # MATLAB lines 697-727. Entries 1 & 2 are duplicates of 'Self-supervised
    # epochs - 10' in the MATLAB source — preserved as-is for index parity.
    raw += [
        (13, 1, "Self-supervised epochs - 10", {"NumEpochsAutoEncoder": 10}, ()),
        (13, 2, "Self-supervised epochs - 10", {"NumEpochsAutoEncoder": 10}, (
            "Duplicate of (13, 1) in the MATLAB source — preserved for index parity.",
        )),
        (13, 3, "Classification Weight - 100", {"WeightClassification": 100}, ()),
        (13, 4, "Classification Weight - 1000", {"WeightClassification": 1000}, ()),
        (13, 5, "Classification Weight - 0.1", {"WeightClassification": 0.1}, ()),
        (13, 6, "KL Weight - 0.1", {"WeightKL": 0.1}, ()),
        (13, 7, "KL Weight - 0.01", {"WeightKL": 0.01}, ()),
        (13, 8, "KL Weight - 1e-3", {"WeightKL": 1e-3}, ()),
        (13, 9, "KL Weight - 10", {"WeightKL": 10}, ()),
        (13, 10, "Reconstruction Weight - 0.1", {"WeightReconstruction": 0.1}, ()),
    ]

    # ── SLURMChoice 14: Target + Stitching+Fusion + StartEndPercent + Classifier ──
    raw += [
        (14, 1, "Target is Outcome", {"Target": "Outcome"}, ()),
        (14, 2, "Target is Outcome with Stochastic Encoder with Global Gradient Clip", {
            "Target": "Outcome",
            "EncoderOutputType": "Stochastic",
            "GradientClipType": "Global",
        }, ()),
        (14, 3, "Target is Outcome with Stochastic Encoder with Global Gradient Clip and Multiple Instance Learning", {
            "Target": "Outcome",
            "EncoderOutputType": "Stochastic",
            "GradientClipType": "Global",
            "MultipleInstanceLearningType": "MIL",
        }, ()),
        (14, 4, "Stitching and Fusion Layer", {
            "MultipleInstanceLearningType": "None",
            "StitchingAndFusionLayer": "Default",
        }, ()),
        (14, 5, "Stitching and Fusion Layer with MIL", {
            "MultipleInstanceLearningType": "MIL",
            "StitchingAndFusionLayer": "Default",
        }, ()),
        (14, 6, "Pre-Feedback Data with MIL", {
            "MultipleInstanceLearningType": "MIL",
            "StartEndPercent": [math.nan, 0.5],
        }, ()),
        (14, 7, "Pre-Feedback Data without MIL", {
            "MultipleInstanceLearningType": "None",
            "StartEndPercent": [math.nan, 0.5],
        }, ()),
        (14, 8, "Feedforward Classifier with Soft Three-Stage Curriculum and Multiple Instance Learning", {
            "ClassifierName": "Deep Feedforward - Dropout 0.5",
            "WantNormalization": True,
            "MultipleInstanceLearningType": "MIL",
            "DynamicParameterSet": "Soft Three-Stage Curriculum",
        }, ()),
        (14, 9, "Feedforward Model with Normalization and Classifier with Soft Three-Stage Curriculum and Multiple Instance Learning", {
            "ModelName": "Feedforward",
            "ClassifierName": "Deep Feedforward - Dropout 0.5",
            "WantNormalization": True,
            "MultipleInstanceLearningType": "MIL",
            "DynamicParameterSet": "Soft Three-Stage Curriculum",
        }, ()),
        (14, 10, "Hierarchically Stratified Sampling", {
            "wantStratifiedPartition": True,
        }, ()),
    ]

    # ── SLURMChoice 15: Data augmentation + curriculum variants ──
    # MATLAB defines 7 named entries here (entries 8-10 stay as the
    # 'Base' filler and are NOT exposed as sweep entries).
    raw += [
        (15, 1, "Data Augmentation with separate time shift", {
            "STDWhiteNoise": 0.15 * 0.1,
            "STDRandomWalk": 0.007 * 0.1,
            "STDChannelOffset": 0.3 * 0.1,
            "STDTimeShift": 100,
            "WantSeparateTimeShift": True,
        }, ()),
        (15, 2, "Standard Stratified Sampling", {
            "wantStratifiedPartition": "Standard",
        }, ()),
        (15, 3, "Weighted Loss", {
            "WeightReconstruction": 100,
            "WeightClassification": 10,
            "WeightKL": 1,
        }, ()),
        (15, 4, "Data Augmentation with separate time shift and Weighted Loss", {
            "STDWhiteNoise": 0.15 * 0.1,
            "STDRandomWalk": 0.007 * 0.1,
            "STDChannelOffset": 0.3 * 0.1,
            "STDTimeShift": 100,
            "WantSeparateTimeShift": True,
            "WeightReconstruction": 100,
            "WeightClassification": 10,
            "WeightKL": 1,
        }, ()),
        (15, 5, "Soft Three-Stage Curriculum: Data Augmentation with separate time shift and Weighted Loss", {
            "STDWhiteNoise": 0.15 * 0.1,
            "STDRandomWalk": 0.007 * 0.1,
            "STDChannelOffset": 0.3 * 0.1,
            "STDTimeShift": 100,
            "WantSeparateTimeShift": True,
            "WeightReconstruction": 100,
            "WeightClassification": 10,
            "WeightKL": 1,
            "DynamicParameterSet": "Soft Three-Stage Curriculum",
        }, ()),
        (15, 6, "Soft Two-Stage Curriculum: Data Augmentation with separate time shift and Weighted Loss", {
            "STDWhiteNoise": 0.15 * 0.1,
            "STDRandomWalk": 0.007 * 0.1,
            "STDChannelOffset": 0.3 * 0.1,
            "STDTimeShift": 100,
            "WantSeparateTimeShift": True,
            "WeightReconstruction": 100,
            "WeightClassification": 10,
            "WeightKL": 1,
            "DynamicParameterSet": "Soft Two-Stage Curriculum",
        }, ()),
        (15, 7, "No Dynamic Parameters: Data Augmentation with separate time shift and Weighted Loss", {
            "STDWhiteNoise": 0.15 * 0.1,
            "STDRandomWalk": 0.007 * 0.1,
            "STDChannelOffset": 0.3 * 0.1,
            "STDTimeShift": 100,
            "WantSeparateTimeShift": True,
            "WeightReconstruction": 100,
            "WeightClassification": 10,
            "WeightKL": 1,
            "DynamicParameterSet": "No Dynamic Parameters",
        }, ()),
    ]

    return raw


def _translate_overrides(matlab_overrides: dict[str, Any]) -> dict[str, Any]:
    """Translate MATLAB CamelCase keys → Python snake_case via the field map.

    Raises ``KeyError`` if a key is missing from :data:`_MATLAB_TO_PYTHON_FIELD`
    — that's a typo in the entry definition above and should be caught at
    module-import time.
    """
    out: dict[str, Any] = {}
    for matlab_key, value in matlab_overrides.items():
        if matlab_key not in _MATLAB_TO_PYTHON_FIELD:
            raise KeyError(
                f"Sweep entry references unknown MATLAB field {matlab_key!r}. "
                f"Add it to _MATLAB_TO_PYTHON_FIELD or fix the typo."
            )
        out[_MATLAB_TO_PYTHON_FIELD[matlab_key]] = value
    return out


def _build_entries() -> tuple[SweepEntry, ...]:
    """Assemble the full sweep table — runs at module-import time."""
    raw = _build_raw_entries()
    entries: list[SweepEntry] = []
    for i, (choice, idx, desc, m_overrides, notes) in enumerate(raw):
        entries.append(
            SweepEntry(
                sweep_index=i + 1,
                matlab_choice=choice,
                matlab_idx=idx,
                description=desc,
                overrides=_translate_overrides(m_overrides),
                notes=notes,
            )
        )
    return tuple(entries)


SWEEP_ENTRIES: tuple[SweepEntry, ...] = _build_entries()
"""All sweep entries in flat 1-based order (sweep_index → entry)."""


# Reverse index: (matlab_choice, matlab_idx) → sweep_index, built once.
_CHOICE_INDEX: dict[tuple[int, int], int] = {
    (e.matlab_choice, e.matlab_idx): e.sweep_index for e in SWEEP_ENTRIES
}


def total_sweep_count() -> int:
    """Number of sweep entries (currently 147)."""
    return len(SWEEP_ENTRIES)


def lookup(sweep_index: int) -> SweepEntry:
    """Return the :class:`SweepEntry` for ``sweep_index`` (1-based).

    Raises
    ------
    IndexError
        If ``sweep_index`` is outside ``[1, total_sweep_count()]``.
    """
    if sweep_index < 1 or sweep_index > len(SWEEP_ENTRIES):
        raise IndexError(
            f"sweep_index={sweep_index} out of range [1, {len(SWEEP_ENTRIES)}]."
        )
    return SWEEP_ENTRIES[sweep_index - 1]


def lookup_by_choice(matlab_choice: int, matlab_idx: int) -> SweepEntry:
    """Return the entry with the given ``(SLURMChoice, SLURMIDX)`` pair.

    Raises
    ------
    KeyError
        If no entry matches the pair (e.g. an SC 15 / IDX 8 query —
        SC 15 has only 7 named entries; entries 8-10 were base-filler
        in the MATLAB source and are not exposed here).
    """
    key = (matlab_choice, matlab_idx)
    if key not in _CHOICE_INDEX:
        raise KeyError(
            f"No sweep entry for (SLURMChoice={matlab_choice}, "
            f"SLURMIDX={matlab_idx}). SC 15 has only 7 named entries; "
            f"SC 1-14 each have 10."
        )
    return SWEEP_ENTRIES[_CHOICE_INDEX[key] - 1]


def iter_by_choice() -> Iterator[tuple[int, list[SweepEntry]]]:
    """Yield ``(choice, entries)`` pairs grouped by MATLAB SLURMChoice."""
    last_choice = -1
    current: list[SweepEntry] = []
    for entry in SWEEP_ENTRIES:
        if entry.matlab_choice != last_choice:
            if current:
                yield last_choice, current
            current = []
            last_choice = entry.matlab_choice
        current.append(entry)
    if current:
        yield last_choice, current


__all__ = [
    "SWEEP_ENTRIES",
    "SweepEntry",
    "iter_by_choice",
    "lookup",
    "lookup_by_choice",
    "total_sweep_count",
]
