# neural_data_decoding

Python port of the MATLAB neural decoding pipeline (`Processing_Functions_cgg/`),
implementing a variational autoencoder + multi-head classifier for multi-probe
ephys data. Reproduces the active production path in modern PyTorch while writing
`.mat`-compatible output where MATLAB-side analysis still consumes it.

> **Status: Milestone B complete; Milestone C (Full Optimal) in progress.**
> Milestones 0 (foundation), A (logistic tracer), and B (GRU + classifier) are
> done and runnable end-to-end on synthetic data, with single-step forward-pass
> parity against MATLAB verified to ~1e-9. See
> `../Plans/neural_data_decoding_plan.md` for the full migration plan, milestone
> sequence, and the MATLAB quirks that must be preserved.

## Quickstart

```bash
cd "Neural Data Reading/neural_data_decoding"

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

# Pre-flight check (aborts if a prior run's checkpoints would be clobbered).
python -m neural_data_decoding check-existing --config-name B_gru_classifier_synthetic --fold 1
```

Output lands under `<repo>/results/<Epoch>/<Target>/<ModelName>/cfg-<hash>/fold-<N>/`
(gitignored): `CM_Table_Validation.mat`, `EncodingParameters.yaml`,
`current_state.pt`, `optimal_state.pt`. Point `--output-root` at your
`ACCRE_DATA` scratch directory for cluster-equivalent paths.

## What works today

| Capability | Status |
|------------|--------|
| Stratified hierarchical K-fold | ✅ element-for-element MATLAB parity |
| Data pipeline + single-session batching | ✅ |
| Logistic + GRU/LSTM encoders, Deep LSTM classifier | ✅ |
| Two-stage lifecycle, checkpoint/resume (no optimizer state) | ✅ |
| `CM_Table.mat` + stable-schema `EncodingParameters.yaml` output | ✅ T4 round-trip parity |
| MATLAB → PyTorch weight conversion (GRU/LSTM/FC) | ✅ |
| VAE sampling, ELBO, confidence, MIL, curriculum | 🚧 Milestone C |

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
│   └── target_milestone/         # A_logistic_synthetic, B_gru_classifier_synthetic
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

Currently **244 tests pass** in the default suite (plus 4 MATLAB-gated parity
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
| C — Full Optimal | 🚧 In progress (VAE / ELBO / confidence / MIL / curriculum) |
| CC — Extra-credit features | ⏳ Pending |
| D — Cluster deployment | ⏳ Pending |
| E — Educational curriculum | 🚧 Scaffolded (authored alongside code milestones) |
| F — Reference documentation | 🚧 Scaffolded (authored alongside code milestones) |

T3 convergence parity and T4 dashboard rendering are validated against real
multi-day MATLAB training runs and are tracked separately from the code-side
milestone completion above.

## Documentation

- **Reference documentation** (Milestone F): `docs/` — MkDocs narrative + Sphinx API.
  Build locally with `bash scripts/build_docs.sh`, output in `docs/build/`.
  Narrative concept/cookbook pages are authored alongside the code milestone that
  introduces them; several are still stubs pending Milestone C+.
- **Educational notebooks** (Milestone E): `notebooks/` — ~60 Jupyter notebooks
  taking a MATLAB programmer to expert Python/PyTorch fluency on this pipeline.
- **Migration plan**: `../Plans/neural_data_decoding_plan.md` — the canonical spec
  for this port, including the full list of MATLAB quirks that must be preserved.

## License

Proprietary. See parent repository.
