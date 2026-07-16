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

## Output

Each run writes to a deterministic, config-encoded directory (see
[Inspecting results](inspecting_results.md)): `CM_Table_Validation.mat`,
`CM_Table.mat`, `EncodingParameters.yaml`, and the `current`/`optimal`
checkpoints. `results/` is gitignored; use `--output-root` for scratch.

## Related

- [The quickstart](../quickstart.md) for a first run.
- [Resuming an interrupted run](resuming_an_interrupted_run.md).
- [The training lifecycle](../concepts/the_training_lifecycle.md).
