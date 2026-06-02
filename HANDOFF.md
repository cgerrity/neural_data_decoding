# Session handoff — neural_data_decoding

A self-contained snapshot of project state, conventions, and the next step.
Intended for a fresh contributor (human or AI) picking up the work — read
top-to-bottom, then start at "Next up". Last updated 2026-06-01.

## Where the project lives

- **Python project (this repo):** `neural_data_decoding/` — standalone GitHub
  repo `cgerrity/neural_data_decoding`. May be located anywhere on disk; the
  scripts and CLI don't assume a specific parent.
- **MATLAB pipeline being ported:** `Processing_Functions_cgg/` and its
  sibling utility folders. These live in the parent MATLAB repo:
  `/Users/cgerrity/Documents/MATLAB/Neural Data Reading/`. They are not part
  of this Python project; they're referenced for parity testing only.
- **Source-root discovery:** the fixture-generation scripts (Python and
  MATLAB) resolve the MATLAB source via `NDD_MATLAB_SOURCE_ROOT` env var,
  with sensible fallbacks. Set it in your shell rc:
  ```bash
  export NDD_MATLAB_SOURCE_ROOT="/Users/cgerrity/Documents/MATLAB/Neural Data Reading"
  ```

## Environment setup

```bash
cd <path-to>/neural_data_decoding
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
python -m neural_data_decoding --help
```

## Verify the suite

```bash
# Default suite (~3s; MATLAB-gated round-trip tests deselected by addopts).
python -m pytest

# MATLAB-gated parity (needs MATLAB executable + Processing_Functions_cgg
# discoverable via NDD_MATLAB_SOURCE_ROOT). Manual milestone-boundary gate.
python -m pytest -m needs_matlab

# Docstring + docs gates.
interrogate src/                       # must be 100%
mkdocs build --strict -f docs/mkdocs.yml
```

Expected: **680 passed, 4 deselected** by default; **4 passed** under
`-m needs_matlab`; interrogate 100%; mkdocs strict 0 warnings (modulo
the cosmetic Material-team blog notice).

## Where things are

```
neural_data_decoding/
├── src/neural_data_decoding/
│   ├── data/                # Dataset, samplers, stratification, normalization
│   ├── models/              # Encoder, decoder, classifier, bottleneck, composite
│   │   └── layers/          # SamplingLayer, NaNToZero, MILSoftmaxLayer
│   ├── training/            # Lifecycle, loop, checkpoint, losses (ELBO, classification)
│   ├── interop/             # MATLAB ↔ Python: CM_Table, folder hierarchy,
│   │                        # weight converter, matlab -batch runner
│   └── utils/               # Paths, seeding, matlab_axes, matlab_source
├── configs/target_milestone/
│   ├── A_logistic_synthetic.yaml
│   └── B_gru_classifier_synthetic.yaml
├── tests/                   # unit/ + parity/ + fixtures/ (fixtures gitignored)
├── scripts/
│   ├── ndd_add_matlab_paths.m       # MATLAB-side source resolver (helper)
│   ├── prepare_golden_fixtures.py   # Python driver for batch regeneration
│   ├── generate_t2_*.m              # Per-test-class fixture generators
│   └── convert_reference_cm_tables.m
└── docs/
    ├── PLAN.md              # Full migration spec (frozen reference)
    ├── narrative/           # MkDocs source (Milestone F)
    └── api/                 # Sphinx source
```

## Current status (2026-05-29)

| Milestone | State | Parity precision |
|-----------|-------|------------------|
| 0 — Foundation | ✅ Complete | Stratification: element-for-element MATLAB |
| A — Logistic tracer | ✅ Complete + smoke-runnable | CM_Table T4 round-trip |
| B — GRU + Classifier | ✅ Complete + smoke-runnable | T2 encoder ~1e-7; composite ~1e-9 |
| C — Full Optimal VAE | ✅ **Core + curriculum + two-stage + confidence + Eq. 2 CE + MIL + accumulation complete** | VAE-core T2 ~1e-6; confidence kernel ~1e-10; Beta P-controller ~1e-12; curriculum interpolator ~1e-12; MIL+Eq. 2 CE analytical; accumulation gradient parity ~1e-6 |
| CC — Extra-credit features | ✅ **all 8 sub-milestones done** — CC.1 (Conv/Resnet/Multi-Filter encoders) + CC.2 (PCA backbone) + CC.3 (MAE) + CC.4 (SGDM) + CC.5 (S&F all 5 variants) + CC.6 (offset/scale augmentation) + CC.7 (unweighted loss) + CC.8 (SLURM sweep coverage audit + 24 integration tests) | See `sweeps/parameter_coverage.py` for the full 47-variable support matrix |
| D — Cluster deployment | 🚧 In progress — plan locked, implementation underway. See [`docs/MILESTONE_D_PLAN.md`](docs/MILESTONE_D_PLAN.md) for the full plan |  |

Milestone C status — what's done

- **VAE sampling layer** + MATLAB parity (deterministic eval `Z = mu`).
- **NaN→0 input transform** + parity (Critical Note #38a).
- **ELBO** (NaN-masked MSE + KL + per-channel telemetry) — the highest-risk
  silent-parity point was resolved empirically: `cgg_lossELBO_v2`
  normalizes by **batch_size**, not `mask.sum()` (the plan's note was
  wrong; a regression-guard test now pins this).
- **Variational architecture** — `SimpleSequenceDecoder`,
  `VariationalComposite` (encoder → bottleneck(2×latent) → sampling →
  {decoder, classifier}), `build_variational_composite`.
- **MIL softmax pooling** + parity (multi-axis softmax across S/C/T;
  `from_formats` mirrors MATLAB `find(ismember(...))`).
- **Confidence routing + PD-controller** — all five Critical Note #29
  subtleties parity-verified to ~1e-10 against MATLAB:
  multiplicative conjunction, ConfidenceDropout, prediction-to-truth
  interpolation, stop-grad on historical EMA, BatchFraction-governed
  cadence. Eq. 10 batch correction also verified.
- **EMA prior normalization** (Critical Notes #6, #30) —
  `aggregate_normalized_losses` ports `cgg_getLossInformation` +
  `cgg_processLossComponent`: per-component EMA (π=0.9 default), rescale
  to Classification's prior, stop-grad on prior, assembly into
  `Loss_Encoder = Loss_Decoder + Loss_Classifier` (Critical Note #28
  single gradient root).
- **Variational training integration** —
  `train_one_epoch` / `validate` / `fit_supervised` detect
  `VariationalOutput` automatically and thread `LossPriors` state across
  iterations. CLI dispatches `is_variational` configs to
  `build_variational_composite`. `configs/target_milestone/C_optimal_synthetic.yaml`
  end-to-ends:
  ```bash
  python -m neural_data_decoding train --config-name C_optimal_synthetic --fold 1
  ```
- **Validation + Test CM_Tables** — the in-training (`CM_Table_Validation.mat`,
  from final epoch state) and final-results (`CM_Table.mat`, from the
  restored Optimal weights run on the held-out test split) are both
  written. Matches the MATLAB pipeline's convention: validation drives
  model selection during training; test is what downstream analysis
  aggregates for the reported results.
- **Dynamic curriculum schedules** (Milestone C #5) — port of MATLAB's
  three sibling schedules (`LoadParameters` / `WeightParameters` /
  `FreezeParameters`) sharing the `cgg_calculateDynamicValue` +
  `cgg_annealWeight` interpolator. Pythonic single-class
  `Schedule` + factory functions (`make_load_schedule`,
  `make_weight_schedule`, `make_freeze_schedule`) + `CurriculumBundle`.
  YAML preset library (`configs/schedule/*.yaml`); the
  `Soft Three-Stage Curriculum - Shortened` regime is verified to
  ~1e-12 per-epoch against the MATLAB
  `PARAMETERS_cgg_selectDynamicParameters` outputs. The
  off-by-one quirk inside `cgg_annealWeight` (segment-end
  discontinuity) is preserved exactly and pinned by a regression test.
  Dataset reads load magnitudes live in `__getitem__` (Critical
  Note #8); freeze applies via per-module optimizer param groups
  (`build_optimizer_with_module_groups` + `apply_freeze_to_optimizer`);
  `RescaleLossEpoch` cadence (Critical Note #6) honored via a
  per-epoch `update_priors_strategy` derived from the MATLAB
  `mod(Epoch+1, N) == 1` test. Smoke run with
  `C_optimal_synthetic.yaml` (now using the Soft Three-Stage regime)
  shows augmentation, weights, and freeze ticking as expected and
  the train loss collapsing right when the classifier unfreezes
  at epoch 11.
- **SGDM optimizer** (Milestone CC #1) — port of MATLAB's SGDM path
  in `cgg_procUpdateNetworks.m` (which despite the config name
  `'SGD'` uses `sgdmupdate`, i.e. SGD with momentum). Default momentum
  is 0.9 per `cgg_initializeOptimizerVariables.m` line 10. New
  `resolve_optimizer_factory(name)` helper in `training/freezing.py`
  maps `"ADAM"` → `torch.optim.AdamW` and `"SGDM"` →
  `torch.optim.SGD` (with `momentum=SGDM_DEFAULT_MOMENTUM=0.9`). The
  returned factory is compatible with both the standard
  `(params, lr=..., weight_decay=...)` call site and the per-module-
  groups call from `build_optimizer_with_module_groups` where per-group
  `lr`s override the factory default. CLI's `_build_optimizer` (and
  the Stage 1 dispatch in `_dispatch_two_stage`) reads `cfg.optimizer`
  (defaults to `"ADAM"`). Smoke check: `fit_supervised` with SGDM on
  synthetic data drops val_loss from 0.82 → 0.33 in 3 epochs.
- **Stitching+Fusion Phase 3 — Gemini cascade variants**
  (Milestone CC #3 Phase 3) — port of MATLAB's
  ``cgg_createStitchingFusionModule_v2.m`` (659 lines) restricted to
  the operations exercised by the three active Gemini option-sets:
  ``'Parallel Single Level'`` (multi-scale kernels ``[3, 5, 7]``, single
  cascade), ``'Cascade Single Kernel - Single Reduction'`` (kernel 3,
  3 cascades, reduce at stage 1), and ``'Cascade Single Kernel -
  Progressive Reduction'`` (kernel 3, 3 cascades, reduce every stage,
  ``EncoderReduction=[4, 2]``). New ``models/stitching_fusion/gemini.py``
  hosts :class:`GeminiStitchingFusionModule` plus
  :func:`build_gemini_stitching_fusion`. The module's encoder runs
  parallel ``(kernel, cascade)`` branches via grouped Conv2d with
  ``[1, kernel_t]`` kernels along ``T``, a bypass projection
  (avg-pool + 1×1 grouped conv per ``StrideBypassMethod='avgpool'``),
  branch addition + ReLU, a spatial conv along ``C``, then a final
  ungrouped 1×1 ``area_fusion`` mixing across areas. The decoder
  mirrors with transposed convs and ends in a grouped 1×1 channel
  reduction. Composite's reconstruction-shape helper expanded from
  T-only to both ``T`` and ``C`` axes (renamed
  ``_match_time_length_5d`` → ``_match_shape_5d``) so the transposed-
  conv stride-math drift doesn't break the loss. Dispatcher's
  ``_GeminiStitchingFusionBridge`` wraps the Gemini module with the
  same boundary ``Linear`` projection pattern as the Default bridge,
  giving a consistent ``(B, W, CAF)`` 3-D contract at the composite
  slot. 21 new tests cover all three variants (encoder/decoder
  shapes, gradient flow, end-to-end composite + autoencoder). Smoke
  run with ``stitching_and_fusion_layer: "Parallel Single Level"``
  completes 3 synthetic epochs end-to-end. All 5 S&F option-sets are
  now buildable via ``cfg.stitching_and_fusion_layer``.
- **Stitching+Fusion Phase 2 — Default variant (per-window 2-D conv)**
  (Milestone CC #3 Phase 2) — port of MATLAB's convolutional cross-area
  encoder/decoder used by the ``'Default'`` S&F option-set. New
  ``models/stitching_fusion/convolutional.py`` provides
  :class:`PerWindowConvolutionalCoder` which operates per-window on the
  ``(T, A, C)`` axes: reshape ``(B, W, T, A, C) → (B*W, A, C, T)``,
  apply ``Conv2d`` with kernel ``(1, kernel_t)`` and stride
  ``(1, stride_t)`` (semantically the MATLAB ``[1, n]`` kernel — filters
  along ``T`` only, never crosses ``C``), reshape back to 5-D. Supports
  ``want_split_areas=True`` (grouped conv with ``groups=A``, per the
  user's note that within-area spatial info is meaningful while
  across-area is not) and the cross-area variant
  (``want_split_areas=False``). Multi-level stacks with optional
  ``ResNet`` residuals and ``RepetitionsPerBlock`` mirror the MATLAB
  pyramid; transposed convs handle the decoder upsample with optional
  cropping. The Default S&F bridge wraps the conv coder with a leading
  (decoder) or trailing (encoder) ``Linear`` projection to/from
  ``CrossAreaFusionSize``, matching
  ``cgg_constructStitchingAndFusionNetwork.m`` lines 84-129. Composite
  forward gained a 5-D-aware time-axis crop/pad helper for the
  transposed-conv stride math. The ``'Feedforward'`` bridge now also
  accepts 5-D input (flattens internally) so the composite always
  calls pre-encoder BEFORE the canonical flatten. 9 new tests cover
  ``PerWindowConvolutionalCoder`` (split-areas, cross-area, ResNet,
  the kernel-doesn't-mix-C invariant) and end-to-end Default S&F
  through the composite (5-D shape contract, gradient flow,
  Stage 1 autoencoder). Smoke run with
  ``stitching_and_fusion_layer: "Default"`` completes 3 synthetic
  epochs end-to-end. The 3 Gemini cascade variants (Phase 3) remain
  pending.
- **Data restructure to (W, T, A, C)** (correction landed alongside
  CC #3) — fixed a long-standing dimensional misunderstanding flagged
  by the user. The MATLAB data layout is ``(C, T, A, W, B)`` per
  trial: ``W`` is the GRU sequence axis (windows), ``T`` is within-
  window samples (MATLAB ``InputSize(2)``), ``A`` is areas (probes;
  ``InputSize(3)``), ``C`` is channels per area (``InputSize(1)``).
  The Python pipeline previously collapsed ``(C, T, A)`` into a
  single ``features`` axis on read, so the convolutional encoder
  variants (which need ``T``, ``A`` explicit) had no way to address
  the right axes. New ``models/layers/data_prep.py`` provides
  :class:`FlattenPerWindow` / :class:`UnflattenPerWindow`; the
  composite flattens ``(B, W, T, A, C) → (B, W, T*A*C)`` before the
  GRU and unflattens the decoder's output back to 5-D for the
  reconstruction loss. ``SyntheticTrialDataset`` gained
  ``samples_per_window`` and ``num_areas`` kwargs (defaults 1, 1 —
  emits collapsed 2-D ``(W, C)`` for backwards compat); both Synthetic
  Optimal and Two-Stage YAML configs now set ``T=2, A=2`` explicitly
  to exercise the multi-dim path on smoke runs. 15 new unit tests
  pin the multi-dim shape contract end-to-end. The convolutional
  encoder + Default S&F variant work (CC #3 Phase 2) was deferred —
  the first attempt was a 1-D conv over the W axis (wrong axis); the
  faithful per-window 2-D conv over ``(T, C)`` with ``A`` as the
  conv channel axis is the correct port and will follow this
  restructure. The Phase 1 architectural hooks (``pre_encoder`` /
  ``post_decoder`` slots; ``'Feedforward'`` S&F bridge) remain
  intact.
- **Stitching+Fusion Phase 1 — Feedforward variant** (Milestone CC #3
  Phase 1) — architectural hooks for the multi-area cross-fusion bridge
  described in `cgg_constructStitchingAndFusionNetwork.m`. `VariationalComposite`
  + `VariationalAutoencoder` gained optional `pre_encoder` / `post_decoder`
  slots; the forward methods route through them when set, and
  `copy_autoencoder_weights` propagates bridge weights on the Stage 1 →
  Stage 2 handoff. New `models/stitching_fusion/` package: `feedforward.py`
  hosts `FeedforwardStitchingFusion` (a per-timestep `nn.Linear` matching
  MATLAB's bare `fullyConnectedLayer(out_features)` block) plus the unified
  `build_stitching_fusion(network_type, *, in_features, cross_area_fusion_size,
  mode)` factory. The factory dispatches on MATLAB's option-set string
  (`'Feedforward'`, `'Default'`, three Gemini variants); Phase 1 implements
  `'Feedforward'` and raises `NotImplementedError` for the rest. Builder
  in `_build_ae_core` reads `cfg.stitching_and_fusion_layer`; when non-empty,
  derives `cross_area_fusion_size = hidden_sizes[0] * 2` (mirrors
  `cgg_constructNetworkArchitecture.m:125`), sizes the encoder around the
  fusion dim instead of raw `in_features`, sizes the decoder to reconstruct
  to the fusion dim, then wires the bridges. With S&F disabled (default
  empty string) the existing topology is unchanged. Smoke runs:
  `C_optimal_synthetic.yaml` and `C_two_stage_synthetic.yaml` both
  end-to-end with `stitching_and_fusion_layer: "Feedforward"` (two-stage
  exercises the bridge weight handoff).
- **MAE decoder loss kernel** (Milestone CC #2) — port of MATLAB's
  `cgg_lossELBO_MAE.m` and the `cgg_getDecoderOutputs.m` dispatch
  switch on `LossType_Decoder`. New `masked_mae_reconstruction_loss`
  in `training/losses/elbo.py` mirrors the MSE kernel's NaN-masking
  + batch-size normalization (Critical Note #38 applies to both),
  but uses `|diff|` and drops the `0.5` factor (MATLAB's `l1loss`
  is sum-of-absolutes; only `0.5 * l2loss` carries the half).
  New `compute_reconstruction_loss(..., loss_type=...)` dispatcher
  routes `"MSE"` / `"MAE"` (case-insensitive). Wired through
  `train_one_epoch` / `validate` / `train_unsupervised_epoch` /
  `validate_unsupervised` and the orchestrators
  `fit_supervised` / `fit_unsupervised` / `fit_two_stage`. CLI
  reads `cfg.loss_type_decoder` (default `"MSE"`). Smoke runs of
  both `C_optimal_synthetic.yaml` and `C_two_stage_synthetic.yaml`
  with `loss_type_decoder: "MAE"` complete end-to-end (Stage 1 recon
  drops 155→128 in 2 epochs with MAE scaling).
- **Hardware-aware gradient accumulation** (Milestone C #9) — port of
  MATLAB's `cgg_procGradientAggregation.m` +
  `cgg_getAccumulationSizeForCurrentSystem.m`. New
  `training/accumulation.py` module: `micro_batch_chunks(n_total,
  max_size)` yields `(start, end, weight)` triples partitioning the
  mini-batch; `get_accumulation_size_for_current_system(cfg.accumulation_information)`
  detects current device(s) via `torch.cuda.get_device_name` (or "CPU"),
  looks up the matching entry, returns the min across detected GPUs.
  `train_one_epoch` gains an `accumulation_max_size` parameter; when set,
  the inner loop runs forward+backward per micro-batch (loss scaled by
  `micro_size/mini_size`), accumulates gradients in `.grad`, then one
  `optimizer.step()` per mini-batch. When `None` or `>= mini_batch_size`,
  fast-path yields a single chunk (identical to non-accumulation). CLI
  resolves accumulation_max_size from `cfg.accumulation_information`
  (supports both dict form and OmegaConf list-of-dicts form). Gradient
  parity verified by test: full-batch vs 4-chunk accumulation produces
  gradients matching to ~1e-6. Step-count invariant pinned: 4 micro-
  batches → 1 optimizer step.
- **`confidence_history` cleanup** (resolved 2026-06-01) — Investigated
  the user-flagged concern. Traced the chain `fit_supervised` →
  `train_one_epoch` → `apply_confidence_routing` → `validate`. Verified
  via diagnostic: validate's classification_loss is **bit-identical**
  across three wildly different `confidence_history` values
  (initial all-1.0, mid-range with custom Beta, near-zero with clamped
  Beta). Cause: validate only consumes `cb_val.total_dropped`, which
  is computed from the model's confidence outputs + dropout mask alone
  (no EMA, no Beta, no history). The full `apply_confidence_routing`
  was doing per-stream-loss + Beta-update work that was then thrown
  away. Fix: extracted `compute_dropped_total_confidence` helper from
  the kernel; `validate` now calls it directly and dropped the dead
  `confidence_history` parameter (replaced with
  `use_interpolated_ce_for_confidence: bool = True`). Regression
  guard: tests pin that the helper produces the same `total_dropped`
  as `apply_confidence_routing` does, and that all branch availability
  combinations (trial-only / task-only / both / neither) work.
- **Aggregate prediction column in CM_Table** (Milestone C #8b) — port
  of the `Aggregation_Prediction` column that MATLAB writes regardless
  of MIL mode (`cgg_getPredictionFromClassifierProbabilities.m` lines
  163, 189; `cgg_getClassifierOutputsFromProbabilities.m` lines 188-193).
  New `aggregate_classifier_predictions(logits_per_dim, *, mil_mode)`
  helper in `losses/classification.py`. Same formula for both modes —
  sum across T then normalize each row to sum to 1: in non-MIL this
  averages per-timestep softmax (uniform prior over which timestep
  "wins"); in MIL it marginalizes the joint softmax over T (the
  normalization is a no-op there). `_write_cm_table_for_split` now
  threads `mil_mode` and writes the `aggregation_prediction` field to
  the .mat file alongside the window prediction (last-timestep argmax).
- **MIL softmax pooling in variational forward path** (Milestone C #8) —
  the kernel (`MILSoftmaxLayer`) was already 1e-6 parity-verified in
  C #4; this milestone wired it into the loss orchestrator. New
  `mil_multi_head_cross_entropy(logits_per_dim, targets, ...)` applies
  joint softmax over `(T, K_d)` per dim → sum over T → marginal probs
  → NLL on the target class's marginal. Mirrors MATLAB's pipeline
  (`cgg_softmaxLayer('SCT')` followed by `Confidence_Aggregation =
  sum(Y, [S, T])` from `cgg_getPredictionFromClassifierProbabilities.m`
  line 163) — same math, but kept as a loss-side transformation in
  Python (cleaner than baking the softmax into the classifier's last
  layer). The existing `interpolated_multi_head_cross_entropy` gained
  a `mil: bool = False` flag so MIL + confidence compose seamlessly:
  with `mil=True`, the joint-softmax-and-aggregate happens first, then
  the same closed-form interpolation `-log(c * p_target_marginal +
  (1-c))` applies on the marginal. `train_one_epoch` + `validate`
  thread a `mil_mode: bool` parameter; CLI routes from
  `cfg.multiple_instance_learning_type == "MIL"`.
  `C_optimal_synthetic.yaml` now enables both `confidence_type:
  ['Trial', 'Task']` AND `multiple_instance_learning_type: "MIL"`;
  smoke run completes 20 epochs end-to-end (final val_acc 0.41,
  comparable scale on train/val loss in the 0.4-0.6 range).
- **Confidence cleanup + validation interpolated CE** (Milestone C #7c) —
  (i) renamed `_branch_loss` → `_compute_confidence_stream_loss` and
  swept remaining "branch" references in docstrings/comments to "stream"
  to avoid collision with "classification branch"; (ii) added
  `symmetric_dropout: bool = False` ablation flag (default = MATLAB
  asymmetric: per-stream losses use undropped); (iii) `validate()` now
  threads an optional `ConfidenceHistory` and calls
  `apply_confidence_routing(confidence_dropout=1.0)` for the eval pass —
  uses interpolated CE without random masking, mirroring
  `torch.nn.Dropout`'s eval-mode behavior (dropout is a training-time
  regularizer). The validation updated_history is discarded (val must
  not mutate training state). Smoke run: val_loss drops from ~3.5 (raw
  CE) to ~0.26 (interpolated CE without dropout) — now on the same scale
  as training loss instead of an unrelated magnitude.
- **Eq. 2 interpolated cross-entropy** (Milestone C #7b) — port of the
  MATLAB `cgg_lossClassification` → `cgg_lossConfidence` chain's per-dim
  flow where the classifier cross-entropy is computed on the
  *confidence-weighted blend of prediction and target* (`Y' = c * Y +
  (1-c) * T`, line 75 of `cgg_lossConfidence.m`), then CE on `Y'`. For
  one-hot targets the math collapses to `-log(c * p_target + (1-c))`
  per-dim per-trial per-timestep, which the new
  `interpolated_multi_head_cross_entropy` helper computes via gather +
  closed-form (no full interpolated-prob tensor materialized). The
  orchestrator reads `cb.total_dropped` from
  `ConfidenceLossBreakdown` (new field — last-timestep × dropout, per-dim)
  to ensure dropout consistency with the branch-loss path. With C #7's
  Y_interpolated already at 1e-10 vs MATLAB (existing case A test) and
  the analytical math directly verified for c=0/c=1/intermediate values,
  the chain is transitively parity-verified. Smoke run with confidence
  enabled hits val_acc 0.458 at epoch 18 (vs 0.396 in C #7 and 0.427 in
  C #5) — the confidence-weighted regularization slows over-fitting.
- **Confidence routing in variational forward path** (Milestone C #7) —
  the kernel (`apply_confidence_routing`) was already 1e-10 parity-verified
  in C #5; this milestone wired it through the model + loss orchestrator.
  New `TrialConfidenceHead` (FC + sigmoid from Z), new `TaskConfidenceHead`
  (parallel FC + sigmoid per output dim, consuming the classifier's
  penultimate features via the new `DeepLSTMClassifier.forward_with_features`).
  `VariationalOutput` gained optional `trial_confidence` / `task_confidence`
  fields; `VariationalComposite` populates them when the heads are built.
  Builder reads `cfg.confidence_type` (case-insensitive, accepts list or
  bare string). The `Confidence_Beta` P-controller is now part of
  `ConfidenceHistory`: a separate MATLAB fixture probes
  `cgg_getConfidenceLossInformation` across three batches and pins Beta
  to ~1e-12. The fixture pre-reduces TrialConfidence/TaskConfidence
  via `cgg_getLastSequenceValue` + flatten/transpose before calling
  the inner function, mirroring the production data flow where
  `cgg_getClassifierOutputsFromProbabilities` strips the T axis (lines
  197/207) and stores last-timestep values in `CM_Table.TrialConfidence`
  / `CM_Table.TaskConfidence`, which `cgg_lossComponents` then reassigns
  back into the local variables (lines 441/447) before passing along
  the chain. So when `cgg_getConfidenceLossInformation.m` line 51
  reads `mean(TotalConfidence, "all")`, the T axis is already gone —
  the average is over `B*K` elements, matching the Python kernel's
  last-timestep `total_undropped.mean()`. `train_one_epoch` threads `ConfidenceHistory` across
  iterations (like `LossPriors`), advancing Beta + EMAs per batch and
  passing the live Beta to `aggregate_normalized_losses.confidence_beta`.
  `C_optimal_synthetic.yaml` now enables `confidence_type: ['Trial', 'Task']`
  and `weight_confidence: 1`; smoke run completes 20 epochs with the
  confidence loss visible in training (and the model converging modestly
  slower vs. no-confidence, as expected with the extra signal). Eq. 2
  prediction-to-truth interpolation was deferred from this commit and
  landed in Milestone C #7b (see entry below).
- **Full two-stage lifecycle** (Milestone C #6) — port of MATLAB's
  `cgg_trainAllAutoEncoder_v2`. Pythonic state machine, **not**
  file-existence branches: `fit_unsupervised` (Stage 1 — encoder +
  bottleneck + decoder, no classifier, "best" = min val loss),
  `fit_two_stage` orchestrator (Stage 1 → load Optimal Stage 1 weights
  back into the AE instance → `copy_autoencoder_weights` into Stage 2
  composite → `fit_supervised`). Separate `AutoencoderModel` +
  `AutoencoderOutput` type (no `logits` field) keeps the types honest
  so Stage 1 cannot accidentally compute classification math. Stage 1
  checkpoints land in `<result_dir>/stage1_autoencoder/`; Stage 2 root.
  Both stages resumable independently. Legacy `cgg_annealWeight` KL
  base anneal now config-wired via `KLBaseAnneal(weight_kl,
  weight_delay_epoch, weight_epoch_ramp)`, applied per-stage. New
  `C_two_stage_synthetic.yaml` config smoke-runs end-to-end (Stage 1
  reconstruction loss drops 85→36, KL ramps 0→1 in Stage 2 epochs 2-5,
  curriculum then takes over, final val_acc 0.438 beats the C #5
  single-stage 0.427).

## Next up — Milestone C complete; CC partial (3 of 8 done); pick CC remainder or D

Milestone C's *active production path* is end-to-end runnable
(variational core, curriculum schedules, two-stage lifecycle,
confidence routing with Beta P-controller and Eq. 2 interpolated CE,
MIL forward integration, aggregate prediction in CM_Table, hardware-
aware gradient accumulation). Milestone CC is partially done — 3 of
the 8 sub-milestones in `docs/PLAN.md` are complete.

### Option A — finish Milestone CC

The official numbering is from `docs/PLAN.md` lines 587-594 (CC.1 …
CC.8). Past commits used my-own "CC #1/#2/#3" sequential labels for
work order; the canonical mapping is:

* ~~**CC.4 — SGDM optimizer** alongside ADAM~~ ✅ done (commits
  31caca0, labeled "CC #1" in commit message)
* ~~**CC.3 — MAE / alternate decoder loss kernels**~~ ✅ done
  (commit 7345022, labeled "CC #2")
* ~~**CC.5 — Stitching + fusion layer**~~ ✅ all 5 variants done
  (commits ed10e0b → 78eaba9, labeled "CC #3 Phases 1-3")
  * Phase 1: Feedforward variant + composite hooks
  * Phase 2: Default convolutional variant (per-window 2-D conv, WantSplitAreas)
  * Phase 3: Three Gemini cascade variants
* ~~**CC.1 — Convolutional / ResNet architecture registry**~~ ✅
  done across two phases:
  * **Phase A** (commit f359d38) — architecture spec registry.
    ``models/architecture_registry.py`` provides
    :class:`ArchitectureSpec` (frozen dataclass mirroring
    ``cgg_constructNetworkArchitecture.m``'s flag bundle) and
    resolve/list/has-architecture helpers. Registered 8 architectures
    (SLURM sweep × 7 + production ``'GRU'``).
  * **Phase B** — encoder builders + CLI wiring. New
    ``models/conv_encoder.py`` provides
    :class:`ConvolutionalEncoder` and
    :class:`MultiFilterConvolutionalEncoder` Pythonic adapters that
    wrap :class:`PerWindowConvolutionalCoder` (CC.5 Phase 2) and
    :class:`GeminiStitchingFusionModule`'s ``'Parallel Single Level'``
    variant (CC.5 Phase 3) respectively. Both accept the composite's
    standard 3-D ``(B, W, T*A*C)`` input, internally reshape to
    explicit ``(B, W, T, A, C)`` for the per-window conv kernels,
    and re-flatten to 3-D for the bottleneck. The reshape is the
    Pythonic alternative to MATLAB's ``dlarray`` ``"CBTSS"`` format
    juggling — functionally identical (same kernel ``(1, n)``,
    stride, optional ``groups=A`` for split-areas) but with explicit
    boundary handling. Registered ``'Convolutional'``, ``'Resnet'``,
    ``'Multi-Filter Convolutional'`` via ``register_encoder`` so
    ``build_encoder(name, cfg)`` works through the unified registry.
    CLI's ``_build_model`` (B-path) gained ``samples_per_window`` /
    ``num_areas`` / ``stride`` plumbing and now computes
    ``flat_in_features = in_features * samples_per_window *
    num_areas`` for the encoder cfg (fixing a pre-existing bug where
    the post-data-restructure B path passed ``C`` instead of
    ``T*A*C``). 14 new tests cover the conv encoders + registry
    dispatch. Smoke runs: ``B_gru_classifier_synthetic.yaml`` with
    ``model_name`` set to each of ``Convolutional`` / ``Resnet`` /
    ``Multi-Filter Convolutional`` (T=4, A=2 multi-dim) trains
    end-to-end with train accuracy climbing (0.44→0.82 across 3
    epochs for Resnet, similar for the others). The other 36
    architectures in
    ``PARAMETERS_cgg_constructNetworkArchitecture.m`` not in the
    SLURM sweep are parameter combinations of the same builders;
    add to the registry as needed.
* ~~**CC.2 — PCA backbone**~~ ✅ done. New `models/layers/pca.py`
  provides `PCAEncodingLayer` / `PCADecodingLayer` (paired modules
  holding `components` and `mean` as buffers — no learnable params)
  plus a registry-facing `PCAEncoder` adapter and
  `fit_pca_encoder_decoder` helper. CLI's `_fit_pca_if_present` walks
  the model after construction and fits any PCA encoder on the
  training loader before the optimizer is built. Mirrors
  `cgg_PCAEncodingLayer.m` semantics: `z = (x - mean) @ components.T`
  with sklearn's `PCA(n_components=...)` providing the components.
  `ModelName='PCA'` registered via `register_encoder`. 19 unit tests;
  smoke run with `model_name: PCA` reaches val_acc 0.42 across 3
  synthetic epochs.
* ~~**CC.6 — Learnable offset/scale augmentation**~~ ✅ done
  (kernel + module + composite wiring). New
  `training/losses/offset_and_scale.py` ports
  `cgg_lossOffsetAndScale.m`: `offset_and_scale_targets(x)` computes
  the MATLAB targets (`T_Scale=range(x)-1, T_Offset=median(x)` for
  the default `'mX+b+X'` equation) by reducing over the per-area
  channel axis. `offset_and_scale_loss(x, y_scale, y_offset)`
  combines `0.5 * l2loss(Y_Scale, T_Scale, Mask=Mask_NaN) + 0.5 *
  l2loss(Y_Offset, T_Offset, Mask=Mask_NaN)` with the same batch-
  size normalization as the ELBO (Critical Note #38). Uses
  `torch.quantile(0.5)` instead of `torch.median` because PyTorch's
  median returns the lower-middle value for even lengths whereas
  MATLAB averages — `torch.quantile` matches MATLAB. New
  `models/layers/offset_scale.py` provides `LearnableOffsetScale`
  (two parallel FC heads producing `(Y_Scale, Y_Offset)` from latent
  `z`) and `find_learnable_offset_scale` for the auto-activation
  pattern from Critical Note #32. Composite integration: both
  `VariationalComposite` and `VariationalAutoencoder` gained a
  `learnable_offset_scale` slot, forward emits
  `output.offset_scale = (Y_Scale, Y_Offset)` when wired (else
  `None`), `copy_autoencoder_weights` propagates the head across
  the Stage 1 → Stage 2 handoff, and the builders read
  `cfg.want_learnable_offset` / `cfg.want_learnable_scale` to wire
  the head. The training loop computes `offset_and_scale_loss` when
  `output.offset_scale is not None` (loss aggregator already had a
  slot for this from earlier work). 21 unit tests cover targets,
  loss kernel (zero-when-equal, positive-otherwise, batch-norm,
  NaN mask, gradient flow), the decoder block, the auto-activation
  helper, composite + autoencoder forward, and the Stage 1 → 2
  handoff.
* ~~**CC.7 — `WeightedLoss=''` unweighted path**~~ ✅ done. The
  Python path was already functionally there:
  ``multi_head_cross_entropy`` accepts ``class_weights_per_dim=None``
  and CLI passes ``None`` whenever ``cfg.weighted_loss != 'inverse'``,
  mapping to MATLAB ``cgg_getWeightsForLoss.m`` lines 8-14's
  ``otherwise → Weights = cell(0)`` branch. CC.7 added explicit
  regression tests pinning the unweighted path + the inverse-weighted
  divergence on imbalanced data.
* ~~**CC.8 — Full SLURM sweep parameter coverage**~~ ✅ done. New
  `sweeps/parameter_coverage.py` documents the 47-variable support
  matrix (40 fully supported, 6 partial / never-exercised, 1 N/A).
  New `tests/integration/test_slurm_sweep_coverage.py` adds 24
  parametrized non-crash integration tests covering each
  registered ModelName (7), each ClassifierName (3), MSE/MAE loss,
  S&F variants (Feedforward / Default / 3 Gemini), ADAM/SGDM
  optimizers, and the WeightedLoss = Inverse / '' / 'None' branches.
  Tests live in `tests/integration/` and run in the default suite
  (~3s total).

Other untracked items mentioned in passing:
* **Data restructure to (W, T, A, C)** ✅ landed alongside CC.5 (the
  dimensional fix the user flagged when the conv encoder was being
  added).
* **Misc** — items like ConfidenceDropout config field, etc.

### Option B — Milestone D (cluster deployment)

Get the Python port runnable on real GPU hardware:

- **submitit / Ray Tune sweep launchers** for hyperparameter sweeps
- **GPU + ACCRE integration** — real-data path with multi-probe SSCTB
  inputs, `needReshape` plumbing, parallel `parfor`-equivalent
  micro-batch processing
- **Real-data dataset** — port `MatFileTrialDataset` (currently a
  TODO comment in `dataset.py`); validate against real ephys files

### What "done" looks like across all of these

Each option ends the same way: smoke-runnable target_milestone config,
parity test where applicable (~1e-10 vs MATLAB), pyright/interrogate
clean, HANDOFF.md + README.md updated, commit pushed.

## Quick command reference

```bash
# Smoke-test each runnable milestone config.
python -m neural_data_decoding train --config-name A_logistic_synthetic --fold 1
python -m neural_data_decoding train --config-name B_gru_classifier_synthetic --fold 1
python -m neural_data_decoding train --config-name C_optimal_synthetic --fold 1
python -m neural_data_decoding train --config-name C_two_stage_synthetic --fold 1

# Pre-flight check (refuses to clobber existing checkpoints).
python -m neural_data_decoding check-existing --config-name C_optimal_synthetic --fold 1
```

## Conventions

- **Output locations**: smoke-test artifacts and training outputs land in
  `<repo>/results/` (gitignored) by default, or pass `--output-root` to
  redirect (e.g. to `$ACCRE_DATA`). Never write to `/tmp/...` without
  saying so first.
- **Fixtures**: under `tests/fixtures/golden_weights/*.mat` (gitignored).
  Regenerate via the per-test `scripts/generate_t2_*.m` or
  `python scripts/prepare_golden_fixtures.py --milestone <N>`.
- **Disclosure**: any change to a file outside `neural_data_decoding/`
  should be announced before the tool call. This includes `~/.claude/`
  auto-memory directories.
- **Docstrings**: NumPy convention, class docstring carries the
  `Parameters` section (not `__init__`). `interrogate --fail-under=100`
  enforces it (config in `pyproject.toml`).
- **MATLAB is behavioral reference, not a syntactic blueprint.** Port
  for semantics; pick the most Pythonic expression of the same observable
  behavior. Example: MATLAB's `if IsOptimal { save_validation;
  save_test }` → Python `on_optimal_callback` hook fired only on a new
  best — same semantics, cleaner separation. Verify parity empirically
  against fixtures; **don't trust the plan's numeric example values**
  (Critical Note #38's `mask.sum()` denominator was wrong; the empirical
  MATLAB probe caught it). For high-risk ports, do ~5 passes of the
  MATLAB source including the call sites.

## Regenerating MATLAB fixtures

Parity fixtures live at `tests/fixtures/golden_weights/*.mat` (gitignored).
Regenerate any one with `matlab -batch "run('scripts/generate_t2_<name>_fixture.m')"`
or all of Milestone N's via `python scripts/prepare_golden_fixtures.py --milestone N`.
Current generators: stratification, T2 encoder (GRU+LSTM), T2 composite,
T2 ELBO, T2 MIL softmax, T2 confidence, T2 confidence Beta,
T2 dynamic-schedule interpolator, CM_Table conversion.

## See also

- `docs/PLAN.md` — frozen migration plan; the canonical spec for every
  Critical Note (especially #1 lifecycle, #29 confidence, #38 NaN mask).
- `CLAUDE.md` — concise repo-context bootstrap for AI coding assistants.
- `README.md` — public-facing status + quickstart.
