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

Wire the curriculum classes that drive the augmentation magnitudes, loss
weights, and freeze decisions per epoch (per Critical Notes #7, #8):

- **LoadParameters / load_schedule.py** — augmentation magnitudes the
  Dataset reads live each `__getitem__` (not snapshotted per epoch).
- **WeightParameters / weight_schedule.py** — per-epoch loss weights
  (drives KL annealing — Critical Note: `WeightDelayEpoch` / `WeightEpochRamp`).
- **FreezeParameters / freeze_schedule.py** — per-network freeze
  magnitudes (`cgg_setFrozenNetwork_v2`, Critical Note #4).
- All schedules port `cgg_calculateDynamicValue` (piecewise-linear
  interpolation between waypoints).
- Add `RescaleLossEpoch` cadence to the training loop so EMA prior
  updates respect the configured cadence (0=every iter, 1=per epoch,
  >1=every N epochs).

After #5: **#6 — full two-stage lifecycle** (Stage 1 unsupervised
pre-training with `NumEpochsAutoEncoder > 0`, then Stage 2 supervised
with the classifier added). Plus KL annealing and hardware-aware
accumulation table.

## Quick command reference

```bash
# Smoke-test each runnable milestone config.
python -m neural_data_decoding train --config-name A_logistic_synthetic --fold 1
python -m neural_data_decoding train --config-name B_gru_classifier_synthetic --fold 1
python -m neural_data_decoding train --config-name C_optimal_synthetic --fold 1

# Pre-flight check (refuses to clobber existing checkpoints).
python -m neural_data_decoding check-existing --config-name C_optimal_synthetic --fold 1
```

Remaining Milestone C items after #3, in order:
- **#4 — EMA prior normalization** + variational training integration so the
  CLI can run a Stochastic VAE end-to-end (currently the composite exists
  but isn't wired into `train_one_epoch`).
- **#5 — Dynamic curriculum schedules** (load / loss-weight / freeze).
- **#6 — Full two-stage lifecycle** (KL annealing, hardware-aware accumulation).

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
