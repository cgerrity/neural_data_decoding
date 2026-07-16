# Inspecting results

Every run writes a small set of files into a deterministic, config-encoded
directory. This page explains where they are and how to read them.

## Where results land

The result directory is a deep path where **each level encodes a piece of the
config** — model, learning rate, hidden sizes, augmentation, curriculum, and
finally the fold. This makes runs self-documenting and lets the MATLAB
aggregator discover them by walking the tree (Critical Note #15). The leaf
`Fold_{N}/` directory contains:

| File | Contents |
|---|---|
| `CM_Table.mat` | final per-trial results (test set, from the Optimal weights) |
| `CM_Table_Validation.mat` | validation results, rewritten on each new best |
| `EncodingParameters.yaml` | the full resolved config (stable MATLAB-name schema) |
| `current_state.pt` | resume snapshot |
| `optimal_state.pt` | best-validation snapshot |

## Reading a `CM_Table.mat` in Python

The table is a MATLAB struct-of-arrays, one row per trial:

```python
import scipy.io, numpy as np
m = scipy.io.loadmat("<...>/Fold_1/CM_Table.mat")["CM_Table"][0, 0]
pred   = m["Aggregation_Prediction"]   # (N, D) per-trial prediction
truth  = m["TrueValue"]                # (N, D) ground truth
acc = (pred.argmax(...) == truth).mean()   # accuracy per your convention
```

Fields: `DataNumber` (trial id, `(N,1)`), `TrueValue` `(N,D)`, `Window_k`
`(N,D)`, `Aggregation_Prediction` `(N,D)`, `TrialConfidence` `(N,1)`,
`TaskConfidence` `(N,D)`.

## Reading in MATLAB

The original analysis runs unmodified: `DATA_cggAllNetworkEncoderResults.m`
walks the `Encoding/` tree, loads every `CM_Table.mat`, computes accuracy, and
groups by config; `FIGURE_cggAllNetworkEncoderResults.m` plots. See
[Running on ACCRE](running_on_accre.md).

## Related

- [Multi-objective losses](../concepts/multi_objective_losses.md) — what the
  confidence columns mean.
- Notebooks `notebooks/08_output_and_analysis/08.1_*` and `08.2_*`.
