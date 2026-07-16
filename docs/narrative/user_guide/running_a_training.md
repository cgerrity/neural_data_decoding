# Running a training

The `train` subcommand runs one training session from a composed config.

## Basic invocation

```bash
python -m neural_data_decoding train --config-name <name> --fold <N>
```

`--config-name` selects a preset from `configs/target_milestone/` (without the
`.yaml`). The shipped presets:

| Config name | What it runs |
|---|---|
| `A_logistic_synthetic` | Logistic Regression tracer (no encoder) |
| `B_gru_classifier_synthetic` | GRU encoder + Deep LSTM classifier |
| `C_optimal_synthetic` | the full Optimal VAE |
| `C_two_stage_synthetic` | two-stage (unsupervised → supervised) |
| `real_data_base` | real `.mat` ephys data |

## Configuration layers

The effective config is composed and overridden in order (later wins):

1. `base.yaml` — defaults.
2. `target_milestone/<name>.yaml` — the experiment.
3. `--sweep-index N` — a sweep point's overrides.
4. `--session <name>` — which recording session.
5. `--override KEY=VALUE` — ad-hoc overrides (repeatable).
6. `--fold N` — the fold.

Composition is plain `OmegaConf.merge` (not Hydra, despite a `hydra-core`
dependency). Example with an override:

```bash
python -m neural_data_decoding train \
    --config-name C_optimal_synthetic --fold 1 \
    --override initial_learning_rate=5e-4 \
    --output-root $SCRATCH/my_runs
```

## Runtime flags

These control *how* a run executes rather than *what* config it composes:

| Flag | Default | Effect |
|---|---|---|
| `--seed N` | `0` | Seeds every RNG (torch, NumPy, `random`) before the model is built and trained, so the same `(config, fold, seed)` reproduces the **exact** same run. Vary it to draw a seed ensemble — e.g. `--seed 1`, `--seed 2`, … for a convergence study. Data splits carry their own fold-derived seeds and are unaffected. |
| `--device D` | `auto` | Compute device. `auto` prefers CUDA, else CPU (Apple MPS is opt-in via `--device mps`, not auto-selected). Also accepts explicit strings like `cuda`, `cuda:0`, `cpu`. |
| `--wandb` | off | Stream per-epoch metrics to a Weights & Biases run (composed onto the stdout logger, not replacing it). Pair with `--wandb-project NAME` and `--wandb-mode {online,offline,disabled}`. |
| `--force` | off | Allow training even when checkpoints already exist in the result directory (otherwise the pre-flight check aborts with exit code 2). |

!!! note "Reproducibility scope"
    Bit-exact reproducibility *across the MATLAB/Python boundary* is deliberately
    not a goal ([ADR 001](../adrs/001_tiered_parity_not_bit_exact.md)). `--seed`
    guarantees *intra-Python* determinism only — the same seed on the same
    machine and build. This is exactly what the seed-ensemble convergence check
    (`tests/parity/test_end_to_end_convergence.py`) relies on.

## Output

Each run writes to a deterministic, config-encoded directory (see
[Inspecting results](inspecting_results.md)): `CM_Table_Validation.mat`,
`CM_Table.mat`, `EncodingParameters.yaml`, and the `current`/`optimal`
checkpoints. `results/` is gitignored; use `--output-root` for scratch.

## Related

- [The quickstart](../quickstart.md) for a first run.
- [Resuming an interrupted run](resuming_an_interrupted_run.md).
- [The training lifecycle](../concepts/the_training_lifecycle.md).
