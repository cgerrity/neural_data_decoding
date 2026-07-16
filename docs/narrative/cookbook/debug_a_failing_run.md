# Debug a failing run

A quick-reference for the four most common failure modes. For the full
walkthrough with live examples, see notebook
`notebooks/09_production_deployment/09.5_debugging_a_failing_run.ipynb`.

## The loss is `NaN`

`NaN` is contagious — one poisons every downstream value. Sources and fixes:

- **`NaN` in the input** (a removed channel reached the encoder) → `NaNToZero`
  before the network.
- **`log(0)` / `0/0`** → the `eps` clamps in the loss kernels.
- **Exploding gradients** → gradient clipping (below).

**Diagnose by bisection:** add `assert not torch.isnan(t).any()` at the input,
after the encoder, and after each loss term. The earliest failing assert
localizes the source. `torch.autograd.set_detect_anomaly(True)` points at the
op that produced a `NaN` in the backward pass (slow — debugging only).

## Out of memory (OOM)

The batch is too big for the GPU. Don't shrink the effective batch — use
**gradient accumulation**: `micro_batch_chunks(n_total, max_size)` splits the
batch into micro-batches whose weights sum to 1, so the accumulated gradient
equals the full-batch gradient exactly.
`get_accumulation_size_for_current_system(table)` auto-sizes the micro-batch to
the detected GPU (Critical Note #18).

## Training diverges

Usually the updates are too big. Checklist:

- **Exploding gradients** → gradient clipping is wired to `cfg.gradient_threshold`
  (`clip_grad_norm_` caps the norm, rescaling so direction survives).
- **Learning rate too high** → lower `initial_learning_rate`.
- **Posterior collapse / KL too strong early** → KL annealing warmup
  ([The dynamic curriculum](../concepts/dynamic_curriculum.md)).
- **One loss dominating** → the EMA normalization
  ([Multi-objective losses](../concepts/multi_objective_losses.md)).

Change **one thing at a time** so you know what fixed it.

## A parity test fails

Read the reported max absolute difference:

- **~1e-7 to 1e-6** → floating-point noise (BLAS/summation order). Usually not a
  real bug; the tolerance may be slightly tight.
- **~1e-3 or larger** → a real divergence. **Probe the MATLAB function directly**
  rather than guessing — the empirical probe is the source of truth. The
  `needs_matlab` tier is skipped by default; run it explicitly to check actual
  MATLAB consumption.

## Prevention: the clobber check

Before training, `check-existing` (and the `--force` guard) detect a directory
that already holds a completed run (Critical Note #22), so a re-submitted sweep
can't silently overwrite results. See
[Recovering from failure](../deployment/recovering_from_failure.md).
