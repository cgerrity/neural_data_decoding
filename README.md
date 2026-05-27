# neural_data_decoding

Python port of the MATLAB neural decoding pipeline (`Processing_Functions_cgg/`),
implementing a variational autoencoder + multi-head classifier for multi-probe
ephys data. Reproduces the active production path in modern PyTorch while writing
`.mat`-compatible output where MATLAB-side analysis still consumes it.

> **Status: Milestone 0 (scaffolding) — not yet runnable.**
> See `../Plans/neural_data_decoding_plan.md` for the full migration plan,
> milestone sequence, and known MATLAB quirks that must be preserved.

## Quickstart (during Milestone 0)

```bash
# Clone the parent repo, then:
cd "Neural Data Reading/neural_data_decoding"

# Set up a Python environment.
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate              # Windows

# Install the package in editable mode with all dev + docs extras.
pip install -e ".[dev,docs]"

# Install pre-commit hooks (ruff, black, nbstripout, interrogate).
pre-commit install
nbstripout --install

# Verify the install.
python -c "import neural_data_decoding; print(neural_data_decoding.__version__)"
python -m neural_data_decoding --help
```

## Project layout

```
neural_data_decoding/
├── pyproject.toml                # PEP 621 project + dev/docs/cluster extras
├── src/neural_data_decoding/     # Top-level Python package
│   ├── config/                   # Typed configs (dataclasses)
│   ├── data/                     # Dataset, samplers, stratification, normalization
│   ├── models/                   # Encoder / decoder / classifier + custom layers
│   ├── training/                 # Loop, lifecycle, losses, schedules, monitoring
│   ├── interop/                  # MATLAB ↔ Python bridge (.mat I/O, folder hierarchy)
│   ├── sweeps/                   # Submitit / Ray Tune launchers
│   └── utils/                    # Paths, seeding, axis converters
├── configs/                      # Hydra-composable YAML configs
├── tests/                        # parity / unit / fixtures
├── notebooks/                    # Educational curriculum (~60 notebooks; Milestone E)
├── docs/                         # MkDocs narrative + Sphinx API reference (Milestone F)
└── scripts/                      # Standalone utilities (fixture prep, doc builds)
```

## Parity status

| Milestone | Status |
|-----------|--------|
| 0 — Foundation | 🚧 In progress (scaffolding) |
| A — Logistic tracer | ⏳ Pending |
| B — GRU + Classifier | ⏳ Pending |
| C — Full Optimal | ⏳ Pending |
| CC — Extra-credit features | ⏳ Pending |
| D — Cluster deployment | ⏳ Pending |
| E — Educational curriculum | ⏳ Pending |
| F — Reference documentation | ⏳ Pending |

## Documentation

- **Reference documentation** (Milestone F): `docs/` — MkDocs narrative + Sphinx API.
  Build locally with `bash scripts/build_docs.sh`, output in `docs/build/`.
- **Educational notebooks** (Milestone E): `notebooks/` — ~60 Jupyter notebooks
  taking a MATLAB programmer to expert Python/PyTorch fluency on this specific pipeline.
- **Migration plan**: `../Plans/neural_data_decoding_plan.md` — the canonical spec
  for this port, including the full list of MATLAB quirks that must be preserved.

## License

Proprietary. See parent repository.
