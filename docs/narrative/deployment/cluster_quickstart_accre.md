# Cluster quickstart (ACCRE)

Get the pipeline running on the ACCRE cluster. This assumes you already have an
ACCRE account and can `ssh` in.

## 1. Set up the environment

```bash
ssh <you>@login.accre.vanderbilt.edu
cd /home/<you>
git clone <repo-url> neural_data_decoding
cd neural_data_decoding

module load Python/3.11        # or the site's current Python module
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Confirm environment detection

The pipeline resolves its base directories from the host. On ACCRE it should
detect the cluster automatically:

```python
from neural_data_decoding.utils.paths import get_base_paths
p = get_base_paths()
print(p.environment.value)     # -> "accre"
print(p.input, p.output, p.temporary)
```

`input` (read-only data) and `output` (preserved results) live under `/home`;
`temporary` (scratch) is the fast lab filesystem. Write checkpoints to scratch,
final tables to output. Force an environment with
`get_base_paths(environment=...)` or the `NDD_FORCE_ENV` env var if detection
guesses wrong.

## 3. Smoke-test one run

Before a sweep, run a single synthetic training to confirm the install:

```bash
python -m neural_data_decoding train \
    --config-name B_gru_classifier_synthetic --fold 1 \
    --output-root $SCRATCH/ndd_smoke
```

## 4. Submit a sweep

Emit and submit array jobs as in [SLURM submission](slurm_submission.md). Watch
them with [Monitoring a running job](monitoring_a_running_job.md), and see
[Recovering from failure](recovering_from_failure.md) for restarts.

## Notes

- Point `--output-root` / `--repo-dir` at cluster paths, never a laptop path.
- A batch always comes from one session ([single-session batching](../concepts/single_session_batching.md)),
  so a run iterates sessions × folds via `SessionRunIDX`.
