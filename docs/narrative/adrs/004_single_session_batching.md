# ADR 004 — Single-session batching

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline trains on multi-probe ephys data where trials are grouped by
**recording session**. Each session is recorded from its own set of probes and
therefore carries its own channel count, its own normalization statistics, and
its own removed-channel pattern (stored on disk as `NaN`). The MATLAB routine
`cgg_procAllSessionMiniBatchTable.m` partitions trials by session and builds the
minibatch table so that **every minibatch contains trials from exactly one
session**; `cgg_procSplitSingleSessionDataStoreByMiniBatchSize.m` does the
per-session chopping into fixed-size chunks, with `WantFullBatch` controlling
whether a partial trailing chunk is dropped.

This is the *opposite* of the naive "shuffle all trials globally, then batch"
default a PyTorch port would reach for. A cross-session batch would mix
incompatible channel dimensions, apply normalization statistics that don't hold
uniformly across the batch, and combine `NaN`-mask patterns that differ trial to
trial — producing a malformed input contract. The single-session grouping is
also the precondition future **per-session stitching/fusion layers** depend on:
those layers apply a session-specific transform that only makes sense when every
trial in the batch shares one session.

Critical Note #9 flags this explicitly as an easy thing to get backwards, and
the risk register calls out "single-session sampler mis-implemented as
cross-session" as a high-likelihood, high-impact failure.

## Decision

Implement a custom batch sampler, `SingleSessionBatchSampler`, that is passed to
the PyTorch `DataLoader` as its `batch_sampler`. What the code actually does
today:

- The constructor takes a 1-D per-trial `session_ids` array (integers or
  strings — only equality is used) plus `batch_size`, `drop_last`, and `seed`.
  It groups trial indices by session id once, preserving within-session order.
- On each `__iter__`, it seeds an RNG from `seed + epoch`, shuffles the trial
  indices **within each session**, chops each session into `batch_size` chunks,
  keeps or drops the partial trailing chunk per `drop_last`, then shuffles the
  **order of the assembled batches** and yields them as plain `list[int]`
  index lists.
- The net effect: successive batches may come from different sessions, but no
  single batch ever mixes sessions. `drop_last=True` mirrors MATLAB's
  `WantFullBatch=true`. `set_epoch` follows the `DistributedSampler` pattern so
  each epoch gets a fresh, reproducible shuffle. `__len__` is computed from
  session sizes and is stable across epochs.

Honesty about scope — what the sampler does **not** do:

- Its only guarantee is that all indices in a yielded batch share one
  `session_ids` value. It does **not** inspect or validate that channel counts,
  normalization statistics, or `NaN` patterns are actually uniform within a
  session — it trusts the caller's per-trial labeling. The channel /
  normalization / NaN consistency benefits follow because trials within a
  session are homogeneous *by construction of the upstream data*, not because
  the sampler enforces them.
- Despite the companion notebook's filename (`..._the_session_balanced_sampler`)
  and an aspirational `SessionBalancedBatchSampler` name in the migration plan,
  the implemented class is `SingleSessionBatchSampler` and it performs **no
  cross-session balancing or quota**. It partitions per session and shuffles
  batch order; it does not equalize how many batches each session contributes,
  nor interleave them round-robin. A session with more trials simply yields more
  batches into the shuffled pool.

## Consequences

**Positive**

- Every batch is a well-formed tensor: one probe layout, one normalization
  regime, one coherent `NaN` structure — matching the `(W, T, A, C)` on-disk
  layout the encoder consumes.
- Establishes the invariant that per-session stitching/fusion layers require, so
  those layers can apply a uniform session-specific transform without runtime
  session-splitting inside the model.
- Reproducible and epoch-varying via `seed + epoch`; `__len__` is deterministic,
  so schedulers and progress bars are stable.
- Matches MATLAB training dynamics (per-session batching) rather than the PyTorch
  global-shuffle default, keeping T3 statistical parity credible.

**Negative**

- Batches are not i.i.d. draws over the whole dataset; gradient noise is
  correlated within a session, and one large session can dominate the shuffled
  batch pool since there is no balancing. If uniform per-session influence is
  ever wanted, that is a separate mechanism, not this sampler.
- Correctness hinges entirely on the caller supplying accurate per-trial
  `session_ids`. The sampler cannot detect a mislabeled trial that silently mixes
  sessions' data within a nominal "session."
- Partial trailing chunks (`drop_last=False`) produce variable-size batches,
  which downstream code and any batch-size-sensitive normalization must tolerate.
- Not a drop-in with stock `DataLoader` batching: callers must pass it as
  `batch_sampler` and must **not** also set `batch_size` / `shuffle`.

## Alternatives considered

1. **Global shuffle + stock `DataLoader` batching (the PyTorch default).**
   Rejected: mixes sessions within a batch, breaking the channel /
   normalization / `NaN` contract and the precondition for per-session stitching.
   This is exactly the inversion Critical Note #9 warns against.

2. **Sort by session and emit contiguous batches with no shuffling.** Rejected:
   it satisfies the one-session-per-batch rule but processes all of session 1,
   then all of session 2, etc., which biases optimization toward whatever session
   is seen last in an epoch and gives poor gradient mixing across sessions.

3. **A session-*balanced* sampler that equalizes each session's contribution
   (round-robin or quota).** Deferred, not chosen: the current requirement is
   purely the single-session invariant. Balancing adds policy (up/down-sampling,
   quota rules) that MATLAB does not apply here, so it was left out to preserve
   parity. The lingering `SessionBalancedBatchSampler` name in the plan and the
   notebook filename are artifacts of an earlier framing, not the shipped design.

4. **Split sessions inside the model / collate function instead of the sampler.**
   Rejected: pushes a data-organization concern into model code, complicates the
   forward pass, and still requires the sampler to know sessions to avoid
   splitting mid-batch. Cleaner to guarantee the invariant at batch-assembly time.

## References

- Implementing code: `src/neural_data_decoding/data/samplers.py`
  (`SingleSessionBatchSampler`).
- Unit test enforcing the invariant:
  `tests/unit/test_samplers.py::test_every_batch_is_single_session` (every
  emitted minibatch must contain trials from exactly one session).
- MATLAB reference: `cgg_procAllSessionMiniBatchTable.m` and
  `cgg_procSplitSingleSessionDataStoreByMiniBatchSize.m` (`WantFullBatch` ↔
  `drop_last`).
- Migration plan: Critical Note #9 (single-session sampler — every minibatch is
  from ONE session; the opposite of a naive reading).
- Companion notebook: `notebooks/03_data_pipeline/03.3_the_session_balanced_sampler.ipynb`.
- Concept page: [Single-session batching](../concepts/single_session_batching.md).
- Related: [ADR 001 — Tiered parity, not bit-exact](001_tiered_parity_not_bit_exact.md)
  (single-session batching is part of what keeps T3 statistical parity honest).
