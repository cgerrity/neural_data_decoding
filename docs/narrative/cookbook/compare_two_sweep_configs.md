# Compare two sweep configs

Each sweep point is addressed by a single integer (`--sweep-index`), and each
writes its results to a deterministic, config-encoded directory. Comparing two
points is a matter of resolving both and reading their result tables.

## Identify the two points

The dispatcher maps a sweep index to a named config variation:

```python
from neural_data_decoding.sweeps import dispatcher

a = dispatcher.lookup(5)
b = dispatcher.lookup(50)
print(a.description, a.overrides)     # e.g. Multi-Filter Network ...
print(b.description, b.overrides)     # e.g. Classifier Hidden Sizes ...
```

Because most sweep points differ from the Optimal baseline in **one or two
config keys** (a one-variable-at-a-time design), the diff of `a.overrides` vs
`b.overrides` tells you exactly what changed — and therefore what any accuracy
difference can be attributed to.

## Find their results

Both runs write a `CM_Table.mat` into their leaf directory in the folder
hierarchy. Because the path encodes the config, sibling runs (same everything
but one variable) share a parent directory. See
[Inspecting results](../user_guide/inspecting_results.md) for reading the
tables, and [SLURM submission](../deployment/slurm_submission.md) for how the
runs are dispatched.

## Compute the comparison

For each config, load its `CM_Table.mat`, read `Aggregation_Prediction` vs
`TrueValue`, and compute accuracy per output dimension. The difference is the
variable's effect. The MATLAB aggregator
`DATA_cggAllNetworkEncoderResults.m` does exactly this across the whole tree and
groups results by config — run it on the Python output unmodified (see
[Running on ACCRE](../user_guide/running_on_accre.md)).

## Keep comparisons clean

Only attribute a result to a variable if the two points differ in *that one*
variable. The curated sweep keeps most points one-variable-from-Optimal for this
reason — if you author a multi-change `SweepEntry`, you lose the clean
attribution.

## Related

- [Parameter sweeps](../user_guide/parameter_sweeps.md).
- Notebook `notebooks/09_production_deployment/09.4_parameter_sweeps.ipynb`.
