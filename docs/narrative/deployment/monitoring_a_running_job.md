# Monitoring a running job

Once a sweep is submitted, you watch it with SLURM's own tools plus the
pipeline's per-run telemetry.

## SLURM-level

```bash
squeue -u $USER                     # your queued/running array tasks
squeue -u $USER -t RUNNING          # only the running ones
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # accounting (memory, time)
scancel <jobid>                     # cancel a job (or <jobid>_<task> for one task)
```

Each array task's stdout/stderr goes to the `--output` file named in the
`.slurm` script — by convention
`.../python_sweep-<idx>-SC<c>-IDX<i>-SessionRunIDX-<task>.txt`. Tail one to see
live progress:

```bash
tail -f Output_Files/python_sweep-5-SC1-IDX5-SessionRunIDX-3.txt
```

## Pipeline-level

Each run prints, at startup, a **banner** (git SHA, torch/GPU info, resolved
config, data shapes) and, per epoch, train/validation loss and accuracy plus the
current curriculum lever values. A "New optimal at epoch N" line marks each time
the best-validation model improved and the `CM_Table` files were rewritten.

The durable telemetry is the `CM_Table_Validation.mat` in each run's leaf
directory — refreshed on every new best, so it always reflects the current best
model (see [Inspecting results](../user_guide/inspecting_results.md)).

## Live experiment tracking (W&B)

Weights & Biases is the intended live-dashboard tool. It is currently a
**declared dependency but not yet wired** — telemetry today is the stdout print
above. The hook to add it (`EpochCallback`) exists; see notebook
`notebooks/08_output_and_analysis/08.5_weights_and_biases_integration.ipynb`.

## Related

- [Recovering from failure](recovering_from_failure.md)
- [The training lifecycle](../concepts/the_training_lifecycle.md)
