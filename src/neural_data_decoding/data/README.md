# `neural_data_decoding.data`

The data pipeline: turn raw trials — synthetic or on-disk MATLAB `.mat`
files — into batched `(B, W, T, A, C)` tensors the composite model can
consume, and own the session-aware batching, stratified K-fold
partitioning, normalization, augmentation, and `.mat` I/O that sit
around that. Every dataset emits trials shaped `(W, T, A, C)` (windows,
within-window samples, areas/probes, channels); batching adds the
leading `B`.

## Entry points

- **`SyntheticTrialDataset`** (`dataset.py`) — in-memory `Dataset` that
  generates classification-friendly trials with tunable signal/noise, so
  the full training loop can be exercised without real data on disk.
- **`MatFileTrialDataset`** (`mat_dataset.py`) — real-data `Dataset`: one
  `Decision_Data_*.mat` + paired `Target_*.mat` per trial, windowed and
  target-dispatched per `cgg_loadDataArray.m` / `cgg_loadTargetArray.m`.
  Builds the per-dim `raw-value → dense-class-index` mapping
  (`class_mapping_per_dim`, `num_classes_per_dim`) at construction.
- **`load_mat`** (`mat_files.py`) — the single `.mat` reader; auto-detects
  v7.3/HDF5 (via `mat73`) vs. pre-v7.3 (via `scipy.io`) from the header
  version word.

These three are what `__init__.py` re-exports. The rest are used by the
training/config layers that assemble the `DataLoader`:

- `collate_trials` (`dataset.py`) — the `collate_fn`; stacks trials into
  `{"x", "targets", "metadata"}`.
- `SingleSessionBatchSampler` (`samplers.py`) — pass as the DataLoader's
  `batch_sampler`; see ADR 004.
- `stratify` + `assign_folds` (`stratification.py`) — recursive
  hierarchical stratifier (ports `cgg_procAssignGroups*.m`) then round-robin
  fold mapping.
- `select_normalization` (`normalization.py`) — string-named recipe
  registry over `(channels, samples, areas)` tensors. **Only `'None'`
  (passthrough) and the one "Optimal" channel-z-score → global-min/max
  recipe are implemented today; the other 14 named recipes are registered
  as `NotImplementedError` stubs.** Standalone utility — not called inside
  either `Dataset.__getitem__`.
- `additive_augmentation_signal` + `generate_time_shift_samples`
  (`augmentation.py`) — the channel-offset / white-noise / random-walk /
  time-shift kernels; see ADR 010.

## Design decisions (ADRs)

- [ADR 004 — Single-session batching](../../../docs/narrative/adrs/004_single_session_batching.md).
  Every minibatch draws from exactly one session. **Honest note:** the
  shipped class is `SingleSessionBatchSampler`; despite the companion
  notebook's filename (`03.3_the_session_balanced_sampler`) and an
  `SessionBalancedBatchSampler` name in the plan, it does **no** cross-session
  balancing or quota — it partitions per session and shuffles batch order,
  nothing more.
- [ADR 010 — Augmentation per `__getitem__`](../../../docs/narrative/adrs/010_augmentation_per_getitem.md).
  Augmentation is re-drawn every `__getitem__` from a live schedule reference
  (never snapshotted), and only when `load_schedule is not None` (train sets;
  val/test pass `None` and stay clean). **Honest note:** unlike MATLAB's
  `fileDatastore`, these datasets hold their base features in memory and
  re-draw only the *additive* noise per read — the per-read re-randomization
  and live-schedule contract hold, but the disk re-read does not literally
  happen.
- [ADR 017 — NaN-masked reconstruction loss](../../../docs/narrative/adrs/017_nan_masked_reconstruction_loss.md).
  **Honest note on scope:** this subpackage does **not** mask or zero
  anything. Its only role in that plumbing is that `MatFileTrialDataset`
  emits the trial tensor with on-disk `NaN`s (removed channels) **preserved
  and untouched**, so the downstream loss can derive its mask from them. The
  `NaNToZero` encoder-input layer and the batch-size-normalized masked loss
  live in `models/` and `training/losses/`, not here. `SyntheticTrialDataset`
  injects no `NaN`s at all (the `dataset.py` module docstring's "returns the
  NaN-zeroed input" line is stale for the synthetic path).

## Learn more

- Concept page: [Single-session batching](../../../docs/narrative/concepts/single_session_batching.md).
- Cookbook: [Add a new target task](../../../docs/narrative/cookbook/add_a_new_target_task.md).
- Notebooks: [`03_data_pipeline/`](../../../notebooks/03_data_pipeline/) —
  Dataset vs. fileDatastore, DataLoader & collation, the sampler, K-fold
  stratification, normalization recipes, and the augmentation-per-call
  contract.
