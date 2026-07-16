# Running on ACCRE

The pipeline detects its host and resolves the right base directories, so the
same code runs on a laptop and on the ACCRE cluster. This page covers the
cluster specifics from a *user's* perspective; for first-time setup see
[Cluster quickstart (ACCRE)](../deployment/cluster_quickstart_accre.md).

## Environment detection

`get_base_paths()` returns the input / output / temporary roots for the detected
environment (Local / TEBA / ACCRE), porting `cgg_getBaseFolders`. On ACCRE:

- **input** (read-only data) and **output** (preserved results) under `/home`;
- **temporary** (scratch) on the fast lab filesystem.

Force detection if needed with the `NDD_FORCE_ENV=accre` env var or
`get_base_paths(environment=...)`.

## Storage tiers matter

Write hot, regenerable data (checkpoints, intermediates) to **temporary**
(fast, but periodically purged); write final results to **output** (preserved).
Conflating them risks losing results to a purge, or wasting backed-up quota on
scratch. Pass `--output-root` to route a run's output where you want it.

## Submitting work

Emit array jobs with `sweep-emit-slurm` and `sbatch` them — see
[SLURM submission](../deployment/slurm_submission.md). A run iterates sessions ×
folds because [every batch comes from one session](../concepts/single_session_batching.md);
the array task ID is `SessionRunIDX`.

## Analyzing on the cluster

Because the Python output is byte-compatible with MATLAB, the original MATLAB
aggregator (`DATA_cggAllNetworkEncoderResults.m`) runs on it unmodified. Python
can invoke MATLAB via the `matlab -batch` subprocess bridge (`matlab_runner`),
gated behind `matlab_available()`. See
[Inspecting results](inspecting_results.md).

## Related

- [Monitoring a running job](../deployment/monitoring_a_running_job.md).
- Notebook `notebooks/09_production_deployment/09.1_environment_detection.ipynb`.
