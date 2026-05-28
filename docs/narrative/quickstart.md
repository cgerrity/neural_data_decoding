# Quickstart

Get a training run going on synthetic data in a few minutes — no real ephys
data or MATLAB install required.

## Install

```bash
cd "Neural Data Reading/neural_data_decoding"

python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate              # Windows

pip install -e ".[dev,docs]"

python -c "import neural_data_decoding; print(neural_data_decoding.__version__)"
python -m neural_data_decoding --help
```

## Run a training

The pipeline ships with two runnable synthetic-data configurations:

```bash
# Milestone A — Logistic Regression tracer bullet (no encoder).
python -m neural_data_decoding train --config-name A_logistic_synthetic --fold 1

# Milestone B — GRU encoder + Deep LSTM classifier.
python -m neural_data_decoding train --config-name B_gru_classifier_synthetic --fold 1
```

Each run prints per-epoch train/validation metrics and writes its outputs to a
deterministic result directory:

```
<repo>/results/<Epoch>/<Target>/<ModelName>/cfg-<hash>/fold-<N>/
├── CM_Table_Validation.mat     # per-trial telemetry MATLAB analysis consumes
├── EncodingParameters.yaml     # resolved config (MATLAB-name field schema)
├── current_state.pt            # resume snapshot (no optimizer state — by design)
└── optimal_state.pt            # best-validation snapshot
```

`results/` is gitignored. To write into cluster-equivalent scratch, pass
`--output-root` (e.g. your `ACCRE_DATA` directory).

## Pre-flight check

Before training, confirm you won't clobber a previous run's checkpoints:

```bash
python -m neural_data_decoding check-existing --config-name B_gru_classifier_synthetic --fold 1
```

It prints the resolved result directory and whether a checkpoint already exists
(exit code 1 if so). Re-run `train` with `--force` to intentionally overwrite.

## Resume an interrupted run

Re-running the same `train` command resumes from `current_state.pt` — training
picks up at the next epoch. Note the optimizer state is intentionally **not**
saved (matching the MATLAB pipeline), so the first iteration after a resume
restarts the optimizer's moment estimates. See
[Resuming an interrupted run](user_guide/resuming_an_interrupted_run.md).

## What's next

- [The training lifecycle](concepts/the_training_lifecycle.md) — how the
  two-stage state machine works.
- [Single-session batching](concepts/single_session_batching.md) — why every
  minibatch comes from one session.
- The README in the repo root — current milestone status and verified parity
  precision.
