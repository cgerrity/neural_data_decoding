# Resuming an interrupted run

Training is checkpointed every epoch so an interrupted run picks up where it
left off.

## How to resume

Re-run the **same** `train` command. The pipeline finds `current_state.pt` in
the resolved result directory and continues from the next epoch:

```bash
# first run (interrupted at, say, epoch 12)
python -m neural_data_decoding train --config-name C_optimal_synthetic --fold 1

# same command again -> resumes at epoch 13
python -m neural_data_decoding train --config-name C_optimal_synthetic --fold 1
```

## What resumes, and what doesn't

- **Model weights and epoch counter** — restored from `current_state.pt` (the
  every-epoch snapshot).
- **Optimizer state is intentionally NOT restored** (matching MATLAB, Critical
  Note #3). AdamW's moment estimates restart from zero on the first iteration
  after a resume. This is by design.
- Resume reads `current_state.pt`, **never `optimal_state.pt`** (Critical Note
  #2) — you continue training, you don't roll back to the best epoch.

## Starting fresh vs resuming

If a directory already has a checkpoint, `train` **resumes** it. To start over in
the same directory, pass `--force` (which overwrites) — but first confirm you
mean to, since it discards the previous run:

```bash
python -m neural_data_decoding check-existing --config-name C_optimal_synthetic --fold 1
# exit code 1 = a checkpoint exists
```

## On the cluster

A preempted or timed-out array task is recovered the same way — re-submit. Use
`check-existing` in a sweep wrapper to re-run only the unfinished tasks. See
[Recovering from failure](../deployment/recovering_from_failure.md).

## Related

- [The training lifecycle](../concepts/the_training_lifecycle.md).
- Notebook `notebooks/05_training_loop/05.5_checkpoint_resume_state_machine.ipynb`.
