# Recovering from failure

Cluster jobs get preempted, time out, or hit a transient error. The pipeline is
built to make a sweep **restartable and idempotent**, so recovery is usually a
matter of re-submitting.

## Resume an interrupted run

Every epoch writes `current_state.pt`. Re-running the *same* `train` command
resumes from it — training picks up at the next epoch. See
[Resuming an interrupted run](../user_guide/resuming_an_interrupted_run.md).

Note: the optimizer state is intentionally **not** saved (matching MATLAB), so
the first iteration after a resume restarts AdamW's moment estimates. This is by
design, not a bug ([the training lifecycle](../concepts/the_training_lifecycle.md)).

## Don't clobber completed runs

Before training, the pipeline checks whether the target directory already holds a
completed run and **aborts unless you pass `--force`** (Critical Note #22). For a
sweep, use `check-existing` as a pre-flight to *skip* already-done points:

```bash
python -m neural_data_decoding check-existing \
    --config-name optimal --sweep-index 5 --session-run-idx 3
# exit code 1 = a checkpoint exists (skip); 0 = safe to run
```

A sweep wrapper script can consult this exit code to re-submit only the tasks
that didn't finish, rather than redoing the whole array.

## Diagnosing the failure

If a task failed rather than just being interrupted:

1. Read its `Output_Files/...-SessionRunIDX-<task>.txt` for the traceback.
2. Match the symptom to a fix in [Debug a failing run](../cookbook/debug_a_failing_run.md)
   — NaN loss, OOM, divergence, or a parity mismatch.
3. If it's OOM, lower the accumulation micro-batch or request more `--mem`.

## Related

- [SLURM submission](slurm_submission.md)
- [Monitoring a running job](monitoring_a_running_job.md)
