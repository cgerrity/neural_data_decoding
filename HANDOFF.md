# Session handoff — neural_data_decoding

A self-contained snapshot of project state, conventions, and the next step.
Intended for a fresh contributor (human or AI) picking up the work — read
top-to-bottom, then start at "Next up". Last updated 2026-05-28.

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

Expected: **296 passed, 4 deselected** by default; **4 passed** under
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
| C — Full Optimal VAE | 🚧 **Core complete + variational smoke-runnable**; curriculum / two-stage / confidence-in-training-path pending | VAE-core T2 ~1e-6; confidence kernel ~1e-10 |
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

## Next up — Milestone C #5 (dynamic curriculum schedules)

Per Critical Notes #7, #8. Three sibling schedules driving augmentation,
loss weights, and freeze decisions; all sharing one piecewise-linear
interpolator (`cgg_calculateDynamicValue`).

### First step — 5-pass read of these MATLAB files

(See the `feedback-matlab-reference` memory for the read-pass discipline.
The user explicitly asked for ~5 passes on high-risk ports; these
qualify.)

1. `Processing_Functions_cgg/Parameters/PARAMETERS_cgg_selectDynamicParameters.m`
   — defines the `'Soft Three-Stage Curriculum - Shortened'`
   waypoint sets.
2. `Processing_Functions_cgg/ANN Functions/Training Functions/cgg_calculateDynamicValue.m`
   — the piecewise-linear interpolator every schedule uses.
3. `Processing_Functions_cgg/ANN Functions/Training Functions/cgg_generateLoadParameters_v2.m`
   — augmentation-magnitude schedule.
4. `Processing_Functions_cgg/ANN Functions/Training Functions/cgg_generateLossWeights_v2.m`
   — per-epoch loss weights (drives KL annealing).
5. `Processing_Functions_cgg/ANN Functions/Training Functions/cgg_generateFreezeParameters.m`
   — per-network freeze magnitudes.
6. The call sites in `cgg_trainNetwork.m` (look for
   `cgg_setFrozenNetwork_v2` and the augmentation-read pattern in the
   data pipeline — Critical Note #8 mandates live-read at
   `__getitem__`, NOT snapshot per epoch).

### Concrete first chunk

| File | Status |
|---|---|
| `src/neural_data_decoding/training/schedules/base.py` | New — abstract `Schedule` base + `piecewise_linear_value(epoch, waypoints)` helper |
| `src/neural_data_decoding/training/schedules/load.py` | New — `LoadSchedule` with `current_*` properties read by the Dataset |
| `src/neural_data_decoding/training/schedules/weights.py` | New — `WeightSchedule` (KL annealing built-in: `WeightDelayEpoch` + `WeightEpochRamp`) |
| `src/neural_data_decoding/training/schedules/freeze.py` | New — `FreezeSchedule` + `apply_freeze_schedule(net, schedule, epoch)` helper |
| `data/dataset.py::SyntheticTrialDataset` | Modify — accept optional `load_schedule` and read magnitudes live in `__getitem__` |
| `training/loop.py` | Modify — call `schedule.update(epoch)` at epoch start; thread freeze application |
| `configs/schedule/soft_three_stage_curriculum_shortened.yaml` | New — waypoint table per `PARAMETERS_cgg_selectDynamicParameters` |
| `tests/parity/test_t2_dynamic_schedules.py` | New — fixture-based parity (per-epoch values vs MATLAB; one fixture per schedule type) |
| `tests/unit/test_schedules.py` | New — piecewise-linear edge cases, live-read contract |

### Also fold in

- `RescaleLossEpoch` cadence in the training loop so the EMA-prior update
  respects `0`/`1`/`>1` cadence (currently always updates).
- Augmentation re-randomization per `__getitem__` (Critical Note #7) is
  already correct via the existing `rng` per-trial draw; the new piece
  is that the **magnitudes** themselves come from the live schedule.

### What "done" looks like

Smoke run with the curriculum config enabled, augmentation magnitudes
visibly tick across epochs (`STDChannelOffset` etc. logged via
`epoch_callback`), KL weight ramps from 0 → 1 between epochs
`WeightDelayEpoch` and `WeightDelayEpoch + WeightEpochRamp`, and at
least one per-network freeze waypoint takes effect. Parity tests pin
per-epoch schedule values to ~1e-10 against MATLAB.

### After #5

**#6 — full two-stage lifecycle**: Stage 1 unsupervised pre-training
(`NumEpochsAutoEncoder > 0`) handing off Optimal autoencoder weights to
Stage 2 supervised (which adds the classifier). Plus KL annealing
integration, hardware-aware accumulation table (Note #18), and the
`needReshape` / OutputBlock plumbing if real-data inputs (SSCTB) arrive
during this work.

## Quick command reference

```bash
# Smoke-test each runnable milestone config.
python -m neural_data_decoding train --config-name A_logistic_synthetic --fold 1
python -m neural_data_decoding train --config-name B_gru_classifier_synthetic --fold 1
python -m neural_data_decoding train --config-name C_optimal_synthetic --fold 1

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
T2 ELBO, T2 MIL softmax, T2 confidence, CM_Table conversion.

## See also

- `docs/PLAN.md` — frozen migration plan; the canonical spec for every
  Critical Note (especially #1 lifecycle, #29 confidence, #38 NaN mask).
- `CLAUDE.md` — concise repo-context bootstrap for AI coding assistants.
- `README.md` — public-facing status + quickstart.
