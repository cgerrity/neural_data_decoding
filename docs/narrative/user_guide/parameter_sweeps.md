# Parameter sweeps

A sweep varies hyperparameters systematically to find the best config or measure
sensitivity. The pipeline uses a **curated** sweep, not an exhaustive grid.

## The sweep table

The MATLAB harness enumerates 47 sweep variables — far too many to grid-search
(`5^47` is astronomical). Instead, a hand-authored table of **147 points**
(`dispatcher.SWEEP_ENTRIES`) varies *one* aspect at a time from the Optimal
baseline. Each point is addressed by a single integer:

```python
from neural_data_decoding.sweeps import dispatcher

dispatcher.total_sweep_count()        # -> 147
e = dispatcher.lookup(5)
print(e.description, e.overrides)      # e.g. "Multi-Filter Network ...", {model_name: ...}
```

The 147 points fall into 15 groups (`iter_by_choice()`), one per MATLAB
`SLURMChoice` — architecture, L2 factor, data width, hidden sizes, classifier
sizes, mini-batch size, loss weights, optimizer, dropout, and so on. Most points
change only one or two config keys, so a result difference isolates that
variable's effect.

## Running a sweep

Pass `--sweep-index N` to `train` (locally) or emit array jobs with
`sweep-emit-slurm` (on a cluster — see
[SLURM submission](../deployment/slurm_submission.md)):

```bash
python -m neural_data_decoding train \
    --config-name optimal --sweep-index 5 --fold 1
```

The full run space is `sweep-index` (config) × `SessionRunIDX` (session × fold).

## Coverage

Not every sweep variable is fully ported. The docstring-only audit at
`sweeps/parameter_coverage.py` tracks each of the 47 variables' status
(supported / partial / N/A). Check it before running a point that touches a
partially-supported variable; `SweepEntry.notes` flags many of these.

## Related

- [Compare two sweep configs](../cookbook/compare_two_sweep_configs.md).
- Notebook `notebooks/09_production_deployment/09.4_parameter_sweeps.ipynb`.
