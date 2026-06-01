# Session handoff — neural_data_decoding

A self-contained snapshot of project state, conventions, and the next step.
Intended for a fresh contributor (human or AI) picking up the work — read
top-to-bottom, then start at "Next up". Last updated 2026-05-31.

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

Expected: **470 passed, 4 deselected** by default; **4 passed** under
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
| C — Full Optimal VAE | 🚧 **Core + curriculum + two-stage + confidence + Eq. 2 CE + MIL forward complete**; accumulation table still pending | VAE-core T2 ~1e-6; confidence kernel ~1e-10; Beta P-controller ~1e-12; curriculum interpolator ~1e-12; MIL+Eq. 2 CE analytical |
| CC — Extra-credit features | ⏳ Pending |  |
| D — Cluster deployment | ⏳ Pending |  |

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
  slower vs. no-confidence, as expected with the extra signal). NOTE:
  Eq. 2 prediction-to-truth interpolation (`y_interpolated`) is computed
  by the kernel but NOT yet wired through the classification loss
  (`apply_confidence_routing(compute_interpolation=False)` in the
  orchestrator); per-dim interpolated cross-entropy is deferred to a
  follow-up.
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

## Next up — Milestone C polish / cleanup, then CC or D

Milestone C's *active production path* is now end-to-end runnable
including confidence routing AND Eq. 2 interpolated CE. What remains
is integration of features whose kernels exist but aren't yet woven
into the variational forward path, plus hardware-aware tuning:

### Option A — Hardware-aware accumulation table (Critical Note #18)

The CLI ignores `cfg.accumulation_information`; current code always
uses `mini_batch_size` for the actual forward batch size. MATLAB's
`cgg_procGradientAggregation` accumulates multiple micro-batches per
optimizer step when the device's `MaxBatchSize` is smaller than
`MiniBatchSize`. Implement gradient accumulation that respects the
hardware table.

### Option B — start Milestone CC (extra-credit features) or D (cluster deployment)

If the deferred Milestone C items aren't blocking real-data runs, jump
ahead to:
- **CC**: SGDM optimizer, alternate loss types (MAE), stitching/fusion
  layer, etc. — niceties to bring the Python port to feature-parity
  with MATLAB's full configuration surface.
- **D**: submitit / Ray Tune sweep launchers, GPU training, ACCRE
  integration.

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
