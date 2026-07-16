# Single-session batching

Every minibatch is drawn from a **single recording session**. This is a
deliberate constraint rooted in how multi-probe ephys data is organized, and it
shapes the data pipeline.

## Why one session per batch

Each session is recorded from its own set of probes, with its own channel
count, normalization statistics, and removed-channel pattern (stored as `NaN`).
Mixing trials from different sessions into one batch would mean:

- inconsistent channel dimensions across the batch;
- normalization statistics that don't apply uniformly;
- `NaN`-mask patterns that differ trial to trial in incompatible ways.

Keeping a batch within one session means every trial in it shares the same probe
layout, the same normalization, and a coherent NaN structure — so the tensor is
well-formed and the model sees a consistent input contract.

## How the data layout encodes it

Trials are shaped `(W, T, A, C)` — windows, time, areas (probes), channels —
matching MATLAB's `InputSize = [C, T, A]` plus a window axis. The removed
channels are `NaN` in the on-disk data; the pipeline zeroes them at the encoder
input (`NaNToZero`) and masks them in the reconstruction loss (see
[VAE sampling](vae_sampling.md) and the NaN-masked reconstruction notebook,
`notebooks/06_loss_orchestration/06.10_*`).

## Consequence for sweeps

Because a batch is one session, a full run iterates over sessions and folds. The
cluster sweep encodes this as `SessionRunIDX` — the SLURM array task ID —
decomposed into `(session, fold)` at runtime, ordered fold-across-sessions (all
sessions for fold 1, then fold 2). See
[SLURM submission](../deployment/slurm_submission.md) and
[Parameter sweeps](../user_guide/parameter_sweeps.md).

## Related

- The data-pipeline notebooks under `notebooks/03_data_pipeline/` cover the
  Dataset, collation, and the `(W, T, A, C)` layout.
