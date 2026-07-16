# Add a new target task

The **decoding target** — what the model predicts — is a plain config field, not
a registry. It's the lightest extension because the target isn't behavior to
swap; it's a column of labels to select.

## The shipped targets

`cfg.target` selects the label set. The shipped values are `Dimension` and
`Outcome`. The dataset reads `cfg.target` to pick which label column to produce,
and the folder-hierarchy namer includes it in the result path.

## Steps to add one

1. **Produce the label column.** Teach the dataset to emit your new target's
   labels. For the real-data loader (`MatFileTrialDataset`), that means mapping
   the target name to the right `.mat` field(s) and the per-dimension class
   counts; for synthetic data, extend the generator to produce that label.

2. **Set `cfg.target`.** Either in a target-milestone config or via
   `--override target=ReactionTime`.

That's it — no registry, no dispatcher edit. The multi-head classifier
([Multi-objective losses](../concepts/multi_objective_losses.md)) already
handles an arbitrary number of output dimensions per target.

## It's already a sweep dimension

The sweep table varies the target (`SLURMChoice` 14, "Target is Outcome"), so
your new target can be swept the same way once the dataset produces it — see
[Parameter sweeps](../user_guide/parameter_sweeps.md).

## Verify

Run a short synthetic training with `--override target=<yours>` and confirm:

- the result directory's path includes your target name;
- the `CM_Table.mat` has `TrueValue` / `Aggregation_Prediction` columns sized to
  your target's dimensions;
- accuracy is being computed against the right labels.

## Related

- Notebook `notebooks/09_production_deployment/09.6_extending_the_pipeline.ipynb`.
- [Inspecting results](../user_guide/inspecting_results.md).
