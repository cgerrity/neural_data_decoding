# neural_data_decoding

Python port of the MATLAB neural decoding pipeline (`Processing_Functions_cgg/`),
implementing a variational autoencoder + multi-head classifier for multi-probe
ephys data. Reproduces the active production path in modern PyTorch while writing
`.mat`-compatible output where MATLAB-side analysis still consumes it.

> **Status: Milestone C core + curriculum + two-stage + confidence + Eq. 2 CE complete; A/B/C all smoke-runnable end-to-end.**
> Milestones 0 (foundation), A (logistic tracer), B (GRU + classifier), and
> Milestone C's variational core (VAE sampling + ELBO + confidence
> PD-controller + MIL pooling + EMA prior normalization + variational training
> integration + dynamic curriculum schedules + full two-stage lifecycle with
> KL annealing + confidence routing with Beta P-controller + **Eq. 2
> interpolated cross-entropy**) are done. Remaining for C: MIL forward
> integration, hardware-aware accumulation. T2 parity against MATLAB
> verified to ~1e-9 (composite forward), ~1e-10 (confidence kernel),
> 1e-6 (ELBO + MIL + sampling), ~1e-12 (curriculum interpolator + Beta
> P-controller). See [`docs/PLAN.md`](docs/PLAN.md) for the full migration plan.

## Quickstart

```bash
cd <path-to>/neural_data_decoding

# Set up a Python environment (3.10+).
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate              # Windows

# Install the package in editable mode with dev + docs extras.
pip install -e ".[dev,docs]"

# Verify the install.
python -c "import neural_data_decoding; print(neural_data_decoding.__version__)"
python -m neural_data_decoding --help
```

### Run a training (synthetic data, no real ephys required)

```bash
# Milestone A — Logistic Regression tracer bullet.
python -m neural_data_decoding train --config-name A_logistic_synthetic --fold 1

# Milestone B — GRU encoder + Deep LSTM classifier.
python -m neural_data_decoding train --config-name B_gru_classifier_synthetic --fold 1

# Milestone C — Stochastic VAE (GRU encoder → 2*latent bottleneck → sampling →
# decoder + Deep LSTM classifier). EMA prior normalization across recon+KL+
# classification; both validation and test CM_Tables written.
python -m neural_data_decoding train --config-name C_optimal_synthetic --fold 1

# Milestone C — full two-stage lifecycle: 5 epochs Stage 1 unsupervised
# pre-training (autoencoder only) → Optimal autoencoder weights handed off
# into a fresh composite → 15 epochs Stage 2 supervised fine-tuning. KL
# anneal applied per-stage.
python -m neural_data_decoding train --config-name C_two_stage_synthetic --fold 1

# Pre-flight check (aborts if a prior run's checkpoints would be clobbered).
python -m neural_data_decoding check-existing --config-name C_optimal_synthetic --fold 1
```

Output lands under `<repo>/results/<Epoch>/<Target>/<ModelName>/cfg-<hash>/fold-<N>/`
(gitignored):

- `CM_Table_Validation.mat` — written each epoch from the validation split
  during training; drives the Optimal-snapshot model selection.
- `CM_Table.mat` — written once at the end, after restoring the Optimal
  weights and running on the held-out **test** split. This is what
  downstream MATLAB analysis aggregates for the final reported results.
- `EncodingParameters.yaml`, `current_state.pt`, `optimal_state.pt` —
  resolved config + resume/best checkpoints.

Point `--output-root` at your `ACCRE_DATA` scratch directory for
cluster-equivalent paths.

## What works today

| Capability | Status |
|------------|--------|
| Stratified hierarchical K-fold | ✅ element-for-element MATLAB parity |
| Data pipeline + single-session batching | ✅ |
| Logistic + GRU/LSTM encoders, Deep LSTM classifier | ✅ |
| Two-stage lifecycle, checkpoint/resume (no optimizer state) | ✅ |
| `CM_Table.mat` + stable-schema `EncodingParameters.yaml` output | ✅ T4 round-trip parity |
| MATLAB → PyTorch weight conversion (GRU/LSTM/FC) | ✅ |
| VAE sampling, ELBO, confidence, MIL, curriculum schedules | ✅ Milestone C core |
| Full two-stage lifecycle + KL annealing | ✅ Milestone C #6 |
| Confidence routing in variational forward path | ✅ Milestone C #7 |
| Eq. 2 interpolated cross-entropy | ✅ Milestone C #7b |
| MIL pooling in variational forward path | ✅ Milestone C #8 |
| Hardware-aware gradient accumulation | ✅ Milestone C #9 |

### Parity precision achieved (T2 single-step forward pass)

Loading MATLAB-trained weights into the Python modules and forwarding the same
input yields:

- GRU / LSTM encoder stacks: **~1e-7** max abs diff
- Full composite (GRU encoder → FC bottleneck → Deep LSTM classifier): **~1e-9**

(Tolerance gate is 1e-5; observed agreement is far tighter.)

## Project layout

```
neural_data_decoding/
├── pyproject.toml                # PEP 621 project + dev/docs/cluster extras
├── src/neural_data_decoding/     # Top-level Python package
│   ├── cli.py                    # `python -m neural_data_decoding ...` entry point
│   ├── data/                     # Dataset, samplers, stratification, normalization
│   ├── models/                   # Encoder / bottleneck / decoder / classifier + composite
│   ├── training/                 # Loop, lifecycle, checkpoint, losses, schedules, monitoring
│   ├── interop/                  # MATLAB ↔ Python bridge (CM_Table, folder hierarchy,
│   │                             #   YAML, weight converter, matlab -batch runner)
│   ├── sweeps/                   # Submitit / Ray Tune launchers (Milestone D)
│   └── utils/                    # Paths, seeding, axis converters
├── configs/                      # Hydra-composable YAML configs
│   ├── target_milestone/         # A_logistic_synthetic, B_gru_classifier_synthetic,
│   │                             #   C_optimal_synthetic
│   └── schedule/                 # Curriculum-regime presets (Milestone C #5)
├── tests/                        # parity / unit / fixtures
├── notebooks/                    # Educational curriculum (~60 notebooks; Milestone E)
├── docs/                         # MkDocs narrative + Sphinx API reference (Milestone F)
└── scripts/                      # Fixture generators, doc builds
```

## Testing

```bash
# Default suite (fast; MATLAB-dependent round-trip tests deselected).
python -m pytest

# MATLAB round-trip / table-writer parity (needs a local MATLAB install;
# spawns `matlab -batch`, ~15-20s cold start). Manual milestone-boundary gate.
python -m pytest -m needs_matlab
```

Currently **758 tests pass** in the default suite (plus 4 MATLAB-gated parity
tests that run with `-m needs_matlab`).

Parity tests compare against MATLAB-generated reference fixtures. Those fixtures
are gitignored — regenerate them locally with the MATLAB-batch scripts in
`scripts/` (e.g. `scripts/generate_t2_composite_fixture.m`) or via
`python scripts/prepare_golden_fixtures.py`.

## Parity status

| Milestone | Status |
|-----------|--------|
| 0 — Foundation | ✅ Complete |
| A — Logistic tracer | ✅ Complete |
| B — GRU + Classifier | ✅ Complete (T2 single-step parity verified) |
| C — Full Optimal | ✅ Complete (VAE / ELBO / confidence / MIL / curriculum / two-stage / accumulation) |
| CC — Extra-credit features | ✅ Complete (CC.1 architecture registry + Conv/Resnet/Multi-Filter encoders, CC.2 PCA, CC.3 MAE, CC.4 SGDM, CC.5 all 5 S&F variants, CC.6 offset/scale augmentation, CC.7 unweighted loss, CC.8 SLURM sweep coverage + 24 integration tests) |
| D — Cluster deployment | ⏳ Pending |
| E — Educational curriculum | ⏳ Directory scaffold only (0 of ~55 notebooks authored) |
| F — Reference documentation | 🚧 ~6 concept pages + ~4 cookbook entries authored; Sphinx API stub exists; full curriculum coverage pending |

T3 convergence parity and T4 dashboard rendering are validated against real
multi-day MATLAB training runs and are tracked separately from the code-side
milestone completion above.

## Documentation

- **Reference documentation** (Milestone F): `docs/` — MkDocs narrative + Sphinx API.
  Build locally with `bash scripts/build_docs.sh`, output in `docs/build/`.
  Status: Sphinx API scaffold + ~6 concept pages + ~4 cookbook entries
  authored; the full curriculum coverage spec'd in `docs/PLAN.md` is still
  pending and will likely be filled in alongside or after Milestone D.
- **Educational notebooks** (Milestone E): `notebooks/` — target ~55 Jupyter notebooks
  across 10 modules taking a MATLAB programmer to expert Python/PyTorch
  fluency on this pipeline. **Status: directory scaffold only — 0 notebooks
  authored.** Originally planned to be written alongside each code milestone
  but in practice we've prioritized the code-side work first; the notebook
  curriculum is a follow-up to Milestone D.
- **Migration plan**: [`docs/PLAN.md`](docs/PLAN.md) — the canonical spec for
  this port, including the full list of MATLAB quirks that must be preserved.

## License

Proprietary. See parent repository.
