# ADR 013 — Memory probe via `cuda.mem_get_info`

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline guarded against out-of-memory (OOM) failures with a helper,
`cgg_getMemoryInformation`, that inspected available device memory and tried to
**predict an imminent OOM before it happened**. It was called roughly six times
per sub-batch — once for each probe point inside the gradient-aggregation inner
loop — so a single mini-batch triggered many redundant memory queries. That
prediction fed the hardware-aware accumulation table (Critical Note #18), which
rewrites `AccumulationInformation` per device so the mini-batch is split into
micro-batches small enough to fit.

Two things had to be decided for the Python port:

1. **How to detect that a device cannot fit a full mini-batch**, so the training
   loop can shrink the micro-batch accordingly.
2. **How to survive an OOM that slips past the estimate** without corrupting the
   run or emitting an inscrutable CUDA stack trace.

MATLAB's per-sub-batch polling is wasteful and its estimator is coupled to
MATLAB's memory model. The port should not replicate that shape verbatim.

## Decision

**The accepted target design** is a dedicated probe at
`src/neural_data_decoding/training/monitoring/memory_probe.py` that calls
`torch.cuda.mem_get_info()` **once per mini-batch** (not six times per
sub-batch) to read free-versus-total VRAM, applies a headroom check to decide
whether the planned micro-batch fits, and wraps the forward/backward pass in a
`try/except torch.cuda.OutOfMemoryError` so any residual OOM is caught and
reported cleanly rather than crashing the process.

**Honest status of the implementation.** As of this ADR, that probe is **not yet
written**. There is no `torch.cuda.mem_get_info()` call anywhere in `src/`, and
`training/monitoring/memory_probe.py` does not exist (the `monitoring/` package
currently contains only its `__init__.py`). This ADR records the probe as the
*accepted-but-pending* approach.

**What the code actually does today** is a conservative, static substitute: a
device-name → max-micro-batch-size lookup table.
`get_accumulation_size_for_current_system` in
`src/neural_data_decoding/training/accumulation.py` takes the config-supplied
`accumulation_information` mapping (e.g. `{"CPU": 100, "NVIDIA RTX A6000": 20}`),
detects the running device(s) via `torch.cuda.get_device_name(i)` — or falls
back to the literal `"CPU"` when no CUDA device is present — looks each detected
name up in the table, and returns the **minimum** matching entry (or `None`,
meaning "single-pass, no accumulation", when the table is empty or no name
matches). That size then drives `micro_batch_chunks`, which partitions the
mini-batch into fitting slices. This avoids OOM by pre-declaring a safe
micro-batch size per known device, **without querying live free memory at all**.

In short: the *live* `mem_get_info` probe is the decided direction; the *static*
table is the shipped interim mechanism. New devices are handled today by adding a
row to the config table, not by measuring VRAM at runtime.

## Consequences

**Positive**

- The decided probe is far cheaper than MATLAB's polling: one query per
  mini-batch instead of six per sub-batch.
- Wrapping forward/backward in an OOM `try/except` converts a raw CUDA abort into
  an actionable, logged error — a strict improvement over silent process death.
- The static table shipped today is fully deterministic and requires no runtime
  memory introspection, so it behaves identically across machines given the same
  config — friendly to reproducibility and to CPU-only CI.
- Keeping the sizing logic behind `get_accumulation_size_for_current_system`
  means the future live probe can replace the table's *return value* without the
  training loop or `micro_batch_chunks` changing.

**Negative**

- The shipped static table cannot adapt to VRAM actually free at runtime: a
  device already partly occupied by another process, or an unusually large
  batch, can still OOM even though the table said the micro-batch "fits".
- Every new GPU model must be added to `accumulation_information` by hand;
  an unlisted device falls through to `None` (single-pass), which may itself OOM
  on large models.
- The ADR title advertises a `mem_get_info` probe that does not exist yet, so the
  document must be read together with this honesty note until the probe lands.

## Alternatives considered

1. **Port `cgg_getMemoryInformation` verbatim, polling ~6× per sub-batch.**
   Rejected: wasteful, and its estimator is tied to MATLAB's memory model rather
   than PyTorch's caching allocator.

2. **Ship the live `torch.cuda.mem_get_info()` probe immediately as the only
   sizing mechanism.** Deferred rather than rejected: the static table was needed
   first to unblock hardware-aware accumulation (Critical Note #18) end-to-end,
   and a purely reactive probe still needs a fallback micro-batch size when the
   headroom check says "too big". The table supplies exactly that floor, so it
   remains useful even after the probe lands.

3. **No proactive sizing at all — just `try/except` OOM and retry with a smaller
   batch.** Rejected as the sole strategy: reactive-only means the first pass of
   every run risks an OOM abort, and PyTorch OOM recovery mid-graph is fragile.
   The `try/except` is retained as the safety net *around* a proactive estimate,
   not as a replacement for one.

4. **Estimate memory analytically from tensor shapes and dtype.** Rejected:
   activation memory depends on the autograd graph, cuDNN workspace selection,
   and allocator fragmentation, none of which a static shape calculation
   captures reliably.

## References

- Current (interim) sizing mechanism:
  `src/neural_data_decoding/training/accumulation.py` —
  `get_accumulation_size_for_current_system` (static device→size table) and
  `micro_batch_chunks` (mini-batch partitioning).
- Planned probe location (not yet implemented):
  `src/neural_data_decoding/training/monitoring/memory_probe.py`.
- Migration spec: Critical Note #19 (memory probe replacing
  `cgg_getMemoryInformation`; "call it once per minibatch, not six times per
  sub-batch") and Critical Note #18 (hardware-aware accumulation table) in
  `docs/PLAN.md`.
- Curriculum coverage: `notebooks/05_training_loop/05.2_gradient_accumulation.ipynb`.
- Related decision on device handling:
  [ADR 014 — Single-GPU default, accelerate for multi](014_single_gpu_default_accelerate_for_multi.md).
- [`torch.cuda.mem_get_info` documentation](https://pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html).
