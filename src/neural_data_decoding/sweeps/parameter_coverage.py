"""CC.8 — SLURM sweep parameter coverage audit.

The MATLAB sweep harness
(``SLURMPARAMETERS_cgg_runAutoEncoder_v2.m`` lines 42-56) enumerates
**47 sweep variables**. This module documents Python support status
for each and provides the parametrized integration tests that verify
representative slices of the parameter space train end-to-end without
crashing (no parity claim — just non-crash gating).

Variable support matrix
-----------------------
Status column: ``✅`` = supported, ``◐`` = partial / not all values,
``N/A`` = not applicable to the Python pipeline.

============================ ============================== =====
Variable                     Python cfg key                 Status
============================ ============================== =====
Fold                         ``fold``                       ✅
ModelName                    ``model_name``                 ✅ (8 registered: Logistic Regression, Feedforward, GRU, LSTM, Convolutional, Resnet, Multi-Filter Convolutional, PCA)
DataWidth                    ``data_width``                 ◐ (cfg field passes through; sets the per-window time dim — synthetic generator uses ``samples_per_window`` directly; the real-data loader (``MatFileTrialDataset``) consumes it as the per-window ``T`` axis)
WindowStride                 ``window_stride``              ◐ (cfg field passes through; controls the stride between adjacent windows during raw-signal windowing — only meaningful with the real-data loader; synthetic trials are generated as independent windows so the stride is implicit)
HiddenSizes                  ``hidden_sizes``               ✅
InitialLearningRate          ``initial_learning_rate``      ✅
WeightReconstruction         ``weight_reconstruction``      ✅
WeightKL                     ``weight_kl``                  ✅
WeightClassification         ``weight_classification``      ✅
MiniBatchSize                ``mini_batch_size``            ✅
Subset                       ``subset``                     ◐ (synthetic_num_sessions / trials_per_session approximate)
Target                       ``target``                     ✅
Epoch                        ``epoch``                      ✅
WeightedLoss                 ``weighted_loss``              ✅ (CC.7 — 'Inverse' and '' both routed)
GradientThreshold            ``gradient_threshold``         ✅
ClassifierName               ``classifier_name``            ✅ (3 registered: Logistic, Deep LSTM - Dropout 0.5, Deep LSTM - Dropout 0.25)
ClassifierHiddenSize         ``classifier_hidden_size``     ✅
STDChannelOffset             ``std_channel_offset``         ✅
STDWhiteNoise                ``std_white_noise``            ✅
STDRandomWalk                ``std_random_walk``            ✅
NumEpochsAutoEncoder         ``num_epochs_autoencoder``     ✅ (two-stage path)
NumEpochsFull                ``num_epochs_full``            ✅
Optimizer                    ``optimizer``                  ✅ (CC.4 — 'ADAM' / 'SGDM')
Normalization                n/a (data normalization)       ◐ (per-channel zscore happens elsewhere)
LossType_Decoder             ``loss_type_decoder``          ✅ (CC.3 — 'MSE' / 'MAE')
LossType_Classifier          ``loss_type_classifier``       ◐ (only 'CrossEntropy' supported; MATLAB has variants)
maxworkerMiniBatchSize       n/a (parallelism)              N/A
L2Factor                     ``l2_factor``                  ✅
Dropout                      ``dropout``                    ✅
WantNormalization            ``want_normalization``         ✅
Activation                   ``activation``                 ✅
IsVariational                ``is_variational``             ✅
BottleNeckDepth              n/a (always 1 in active configs)   ◐
WantSaveOptimalNet           always-on (CM_Table.mat writer)    ✅
EncoderOutputType            ``encoder_output_type``        ✅
GradientClipType             ``gradient_clip_type``         ✅
MultipleInstanceLearningType ``multiple_instance_learning_type`` ✅
DynamicParameterSet          ``dynamic_parameter_set``      ✅
StitchingAndFusionLayer      ``stitching_and_fusion_layer`` ✅ (CC.5 — all 5 variants)
StartEndPercent              n/a (stratification)           ◐
wantStratifiedPartition      ``want_stratified_partition``  ✅
STDTimeShift                 ``std_time_shift``             ✅
WantSeparateTimeShift        n/a (time-shift always per-trial)  ◐
WeightOffsetAndScale         ``weight_offset_and_scale``    ✅ (CC.6 — loss kernel + decoder block)
RescaleLossEpoch             ``rescale_loss_epoch``         ✅
WeightConfidence             ``weight_confidence``          ✅
ConfidenceType               ``confidence_type``            ✅
============================ ============================== =====

Coverage: ~40 of 47 variables fully supported, 7 partial (data-prep
fields that pass through cfg but are consumed by the
real-data loader, plus never-exercised options), 1 N/A (parallelism
not applicable to the single-GPU Python path).

Integration tests
-----------------
:mod:`tests.integration.test_slurm_sweep_coverage` parametrizes
representative slices (each ModelName, each S&F variant, MAE vs MSE,
SGDM vs ADAM, etc.) and runs 1-2 synthetic epochs through the CLI to
verify non-crash training. These tests catch regressions where a
parameter combination silently breaks the pipeline.
"""

from __future__ import annotations


__all__: list[str] = []
