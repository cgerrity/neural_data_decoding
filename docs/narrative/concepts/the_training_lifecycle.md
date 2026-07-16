# The training lifecycle

The pipeline trains a model through a **two-stage state machine**, with
checkpointing designed for interruptible cluster jobs. This page explains how a
run progresses from start to finish.

## Two stages

When `NumEpochsAutoEncoder > 0`, training runs in two stages:

1. **Stage 1 — unsupervised pre-training** (`fit_unsupervised`). The encoder and
   decoder learn to reconstruct the neural signal with no classification
   pressure. This gives the encoder a useful latent representation before the
   classifier ever sees it.
2. **Handoff** (`copy_autoencoder_weights`). The best Stage 1 autoencoder
   weights are loaded into the composite model.
3. **Stage 2 — supervised training** (`fit_supervised`). The classifier is
   trained on the (now meaningful) latent, with the full multi-objective loss,
   the dynamic curriculum, and KL annealing.

With `NumEpochsAutoEncoder = 0`, Stage 1 is skipped and training is
single-stage supervised.

## Per-epoch flow (Stage 2)

Each epoch, `fit_supervised` does, in order:

1. `curriculum.update(epoch + 1)` — advance all schedule levers (the `+1`
   converts Python's 0-indexed epoch to MATLAB's 1-indexed).
2. `apply_freeze_to_optimizer(...)` — write the per-module learning-rate
   factors (freezing is LR-scaling, **not** `requires_grad` — see
   [Dynamic curriculum](dynamic_curriculum.md)).
3. `_resolve_epoch_loss_weights(...)` — snapshot the loss weights for the epoch.
4. `train_one_epoch(...)` then `validate(...)`.

Validation runs in `eval()` mode under `torch.no_grad()` — no gradients, and no
BatchNorm running-stat updates (Critical Note #34).

## Current vs Optimal checkpoints

Two checkpoints are maintained, with different jobs (Critical Notes #2, #3):

| Checkpoint | Written | Purpose |
|---|---|---|
| `current_state.pt` | **every epoch** | resume point |
| `optimal_state.pt` | on a **new best** validation accuracy | model selection |

- **Resume reads `current_state.pt`, never `optimal_state.pt`** — resuming
  continues where training left off, not from the best epoch.
- **Optimizer state is deliberately not saved** (matching MATLAB), so the first
  iteration after a resume restarts AdamW's moment estimates.

When a new best appears (`EpochHistory.is_best`), an `OnOptimalCallback` fires
and rewrites `CM_Table_Validation.mat` (and, at the end, `CM_Table.mat` from the
restored Optimal weights on the test set). See
[Multi-objective losses](multi_objective_losses.md) for what those files
contain, and [Resuming an interrupted run](../user_guide/resuming_an_interrupted_run.md)
for the operational side.

## Where to go deeper

The training-loop mechanics are taught step by step in the curriculum notebooks
under `notebooks/05_training_loop/` (custom loop, gradient accumulation,
clipping, checkpoint state machine, the two-stage lifecycle, BatchNorm state).
