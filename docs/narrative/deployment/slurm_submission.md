# SLURM submission

A full sweep is hundreds of runs (config variants × sessions × folds). On a
cluster you submit them as a **SLURM array job** — one `sbatch` script the
scheduler expands into many parallel tasks.

## Emit a job script

The `sweep-emit-slurm` subcommand renders a ready-to-submit `.slurm` file for one
sweep point:

```bash
python -m neural_data_decoding sweep-emit-slurm \
    --sweep-index 5 \
    --config-name optimal \
    --output-path run_5.slurm \
    --num-sessions 25 --num-folds 10 \
    --time 48:00:00 --mem 64G --cpus-per-task 10 \
    --array-throttle 10 \
    --repo-dir /home/<you>/neural_data_decoding
```

The emitted script contains an array directive
`#SBATCH --array=1-<sessions*folds>%<throttle>` and a body that runs, per task:

```bash
python -m neural_data_decoding train \
    --config-name optimal \
    --sweep-index 5 \
    --session-run-idx $SLURM_ARRAY_TASK_ID
```

Submit it with `sbatch run_5.slurm`.

## The two axes

- **`--sweep-index`** selects the *config* (1..147, a curated table — see
  [Parameter sweeps](../user_guide/parameter_sweeps.md)).
- **`$SLURM_ARRAY_TASK_ID`** is `SessionRunIDX`, decomposed at runtime into
  `(session, fold)` in fold-across-sessions order (all sessions for fold 1, then
  fold 2). One task = one `(config, session, fold)`.

To run the whole config sweep, emit a `.slurm` per index (loop over
`1..total_sweep_count()`) and `sbatch` each.

## Etiquette

- Set a sane `--array-throttle` (the `%N`) so you use a fair share of the
  cluster, not all of it.
- `--repo-dir` and the venv-activate path must be the **cluster** paths (see
  [Running on ACCRE](../user_guide/running_on_accre.md) and
  [Environment detection](../concepts/single_session_batching.md) for the
  session model).

## Related

- [Cluster quickstart (ACCRE)](cluster_quickstart_accre.md)
- Notebook `notebooks/09_production_deployment/09.2_slurm_dispatch.ipynb`.
