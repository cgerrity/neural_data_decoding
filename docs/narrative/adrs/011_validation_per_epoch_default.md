# ADR 011 — Validation per epoch by default

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline (`cgg_trainNetwork.m`) runs validation every
`ValidationFrequency` iterations (default 25), driven by a running global
iteration counter inside the double `for epoch / for batch` loop. That
per-iteration cadence was the path of least resistance in MATLAB, where the
monitor + checkpoint framework was already keyed off the iteration index.

Critical Note #21 in the migration plan flagged this as a place to become
Python-idiomatic rather than to mirror MATLAB literally: default to
**per-epoch** validation (the standard PyTorch rhythm — validate once after
each full pass over the data), while nominally reserving a
`validate_every_n_iterations` knob for anyone who wants the old MATLAB
cadence back. The teaching material takes the same line: notebook 05.1
frames "a real loop *validates* after each epoch" as the baseline rhythm and
never introduces sub-epoch validation.

## Decision

Validation runs **once per epoch, unconditionally**, in both training stages.

`fit_supervised` (the Stage 2 epoch orchestrator) contains a single
`for epoch in range(start_epoch, num_epochs)` loop. Inside it, after
`train_one_epoch` returns, it calls `validate(...)` once over the entire
`val_loader` (when a `val_loader` is provided). Optimal ("best") tracking is
therefore also epoch-granular: `is_best` is decided by comparing this epoch's
validation accuracy against `best_metric`, and the Optimal checkpoint + both
`CM_Table` writes fire on that per-epoch comparison. Stage 1 mirrors this:
`fit_unsupervised` calls `validate_unsupervised(...)` once per epoch. There is
no per-iteration validation branch anywhere in the loop bodies.

Honest note on the knob: `validate_every_n_iterations` **is present in
`configs/base.yaml` (defaulting to `null`) but is read by no code path.**
A repo-wide search finds it only in that one YAML line — no config dataclass
field, no CLI wiring, no consumer in `training/`. It exists to satisfy the
rule that every field must appear in every saved `EncodingParameters.yaml` so
`cgg_plotParameterSweep` can diff across runs (Critical Note #25), and to
document the intended default. The corollary is the part the ADR's title
does not say on its own: setting `validate_every_n_iterations` to an integer
today does **nothing** — the MATLAB per-iteration cadence that Critical
Note #21 envisioned as an opt-in is not implemented. Likewise MATLAB's
`validation_frequency: 25` is carried in the config for parity/completeness
but is not consumed by the Python loop. Per-epoch is not a configurable
default; it is the only behavior.

## Consequences

**Positive**

- The loop matches idiomatic PyTorch and the Module 05 curriculum, so a
  reader of `fit_supervised` sees exactly the rhythm 05.1 taught — no hidden
  iteration counter to reason about.
- Best-model selection, checkpointing, and the `CM_Table` writes all share a
  single, easy-to-audit granularity (the epoch), removing a class of
  off-by-one and "which iteration was best" ambiguities.
- Validation cost is bounded to one pass per epoch rather than a full
  validation sweep every 25 iterations, which for large iteration counts is
  meaningfully cheaper.

**Negative**

- Runs with very few, very large epochs get sparse validation signal; there
  is no supported way to validate mid-epoch.
- The config surface is misleading: `validate_every_n_iterations` looks like a
  live knob but is inert. A user who sets it expecting MATLAB cadence gets
  silent per-epoch behavior with no error. This is a documentation-only
  guardrail today, not an enforced one.
- Iteration-level convergence curves comparable to a MATLAB run at
  `ValidationFrequency = 25` cannot be produced without new code, so any T3
  statistical-parity comparison against MATLAB validation curves must align on
  epoch boundaries, not iterations.

## Alternatives considered

1. **Mirror MATLAB's per-iteration cadence exactly** (validate every
   `ValidationFrequency` iterations). Rejected: it imports MATLAB's global
   iteration-counter bookkeeping into an otherwise clean epoch loop, and it
   buys nothing scientifically — the model's validation accuracy at epoch
   boundaries is the quantity the pipeline actually selects Optimal weights on.

2. **Implement `validate_every_n_iterations` now** as a real, wired knob.
   Deferred, not rejected: it is genuinely useful for long-epoch debugging,
   but no current milestone needs it, and wiring it correctly means threading
   validation, best-tracking, and checkpoint writes to iteration granularity —
   nontrivial work for an unused feature. The field is left in place as the
   documented seam for that future work.

3. **Delete the unused field** to avoid the misleading surface. Rejected: it
   would violate the "every field appears in the saved parameters YAML" rule
   (Critical Note #25) and erase the intended-extension marker. The chosen
   compromise is to keep the field and document its inertness here.

## References

- Epoch-granular validation call site: `src/neural_data_decoding/training/lifecycle.py`
  (`fit_supervised` epoch loop; `fit_unsupervised` for Stage 1).
- Validation kernels: `src/neural_data_decoding/training/loop.py` (`validate`,
  `validate_unsupervised`).
- Config field (present, unconsumed): `configs/base.yaml`
  (`validate_every_n_iterations`, `validation_frequency`).
- Migration plan: Critical Note #21 (make validation timing Python-idiomatic);
  the field's presence-for-parity rationale is Critical Note #25.
- Curriculum: `notebooks/05_training_loop/05.1_the_custom_training_loop.ipynb`
  (the per-epoch train/validate rhythm).
- Related: [ADR 006 — resume reads current, not optimal](006_resume_reads_current_not_optimal.md)
  (the other half of the epoch-granular checkpoint story).
