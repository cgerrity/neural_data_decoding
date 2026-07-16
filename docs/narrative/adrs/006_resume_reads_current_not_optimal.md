# ADR 006 — Resume reads Current, not Optimal

**Status**: Accepted
**Date**: 2026-07-16

## Context

A production run trains for up to 500 epochs and can be killed at any moment —
a preempted cluster node, a crash, a `Ctrl+C`. When it restarts it must decide
which weights to reload. The MATLAB pipeline (`cgg_trainNetwork`) tracks **two**
snapshots with different purposes, and the Python port carries the same split:

* **Current** — written every epoch, unconditionally. In MATLAB
  `CurrentIteration.mat` + `Encoder-Current.mat`; here `current_state.pt`. It
  is the *latest* state, and it is what resume reads.
* **Optimal** — written only when validation strictly improves. In MATLAB
  `Encoder-Optimal.mat`; here `optimal_state.pt`. It is the *best-validation*
  state, used to produce the deliverable CM-tables, never for resume.

The tempting mistake is to "resume from the best." Late in training a model can
overfit, so its Current (latest) weights may score worse than its Optimal
(best-validation) weights. Reaching for Optimal on restart feels like recovering
the better model. It is a trap: Optimal can be many epochs behind the latest,
so resuming from it would silently rewind training on every restart and never
make forward progress past the overfitting point. Checkpointing exists to
*continue an interrupted run*, which is a distinct concern from *tracking the
best model* (Critical Note #2).

## Decision

Resume reads **Current, never Optimal** — and the code does exactly that today.

`save_current_checkpoint` runs at the end of every epoch and persists
`model_state_dict` + `epoch` + `iteration` + `best_metric`.
`save_optimal_checkpoint` runs only inside the `val_metric > best` branch and
persists `model_state_dict` + `epoch` + `metric`.

On startup, `fit_supervised` (and `fit_unsupervised`) calls
`load_current_checkpoint` and *only* that function. It restores the weights
in-place, then sets `start_epoch = resumed.epoch + 1`,
`best_metric = resumed.best_metric`, and `iteration = resumed.iteration`.
`load_optimal_checkpoint` is **never** called on the resume path.

Two honest nuances the title does not spell out, recorded here plainly:

1. **Optimal is still read — just not for resume.** The sole in-code caller of
   `load_optimal_checkpoint` today is the Stage 1 → Stage 2 handoff in
   `fit_two_stage` (`lifecycle.py:547`), which loads the best *pre-training*
   autoencoder weights into the composite before supervised training begins —
   a deliberate transfer of the best snapshot, not a resume. (The function is
   also the intended entry point for downstream best-weights evaluation, though
   the end-of-run test-set `CM_Table` is currently written from the in-memory
   `on_optimal_callback` during training rather than by reloading
   `optimal_state.pt`.)

2. **Resume continues; it is not a perfectly faithful continuation.** The
   Current snapshot carries `best_metric` forward, so the Optimal-gating bar
   survives a restart and Optimal is not spuriously re-written after resume.
   But optimizer state is deliberately not saved (Critical Note #3, see
   [ADR 005](005_no_optimizer_state_in_checkpoints.md)), so on resume `AdamW`'s
   moment estimates restart from zero. Resume restores weights and progress
   counters, not the full optimizer trajectory — the first few post-resume
   iterations feel like fresh training before the moments rebuild.

The bottom line matches the title: resume rolls training *forward* from the
latest state, never *back* to the best.

## Consequences

**Positive**

- Restarts always make forward progress. Resume picks up the most recent state,
  so an interrupted run cannot rewind past its overfitting point on every
  restart.
- The two snapshots keep their two jobs unambiguous: Current for continuation,
  Optimal for the deliverable. Named filename constants
  (`CURRENT_CHECKPOINT_FILENAME`, `OPTIMAL_CHECKPOINT_FILENAME`) prevent
  stringly-typed path confusion.
- Because `best_metric` rides along in the Current snapshot, the Optimal high-
  water mark is preserved across restarts — resume does not lower the bar or
  re-emit an Optimal write it should not.
- Behavior matches MATLAB, so cross-checks against the reference pipeline stay
  meaningful.

**Negative**

- Resume is not bit-faithful to an uninterrupted run: the optimizer restarts
  from scratch, so an interrupt-then-resume trajectory diverges slightly from
  an uninterrupted one. Parity tests account for this (ADR 001 tiering).
- If a run overfits and is then interrupted, resume reloads the (worse) latest
  weights, not the best ones. That is correct for *continuing training* but can
  surprise anyone who expects resume to recover the best model — hence this ADR.
- The distinction relies on callers never wiring `load_optimal_checkpoint` into
  a resume path. The function's docstring warns against it, but nothing at the
  type level forbids the mistake.

## Alternatives considered

1. **Resume from Optimal (best-validation).** Rejected: Optimal can lag the
   latest state by many epochs, so every restart would rewind training and a
   run that overfits could never advance past that point. Conflates "continue"
   with "recover the best model."

2. **Keep a single checkpoint that is both latest and best.** Rejected: the two
   roles genuinely conflict once a model overfits — the latest weights (needed
   to continue) and the best weights (needed for the deliverable) diverge. One
   file cannot serve both.

3. **Save optimizer state in Current for a fully faithful resume.** Rejected
   here as out of scope; it is its own decision recorded in
   [ADR 005](005_no_optimizer_state_in_checkpoints.md). The MATLAB pipeline
   intentionally drops optimizer state to keep checkpoints small, and this port
   matches that.

## References

- Save/load state machine: `src/neural_data_decoding/training/checkpoint.py`
  (`save_current_checkpoint`, `save_optimal_checkpoint`,
  `load_current_checkpoint`, `load_optimal_checkpoint`).
- Resume wiring: `src/neural_data_decoding/training/lifecycle.py` —
  `fit_supervised` / `fit_unsupervised` resume block
  (`load_current_checkpoint` → `start_epoch = resumed.epoch + 1`), and the
  Optimal-only Stage 1 → Stage 2 handoff in `fit_two_stage`.
- Migration spec: Critical Note #2 (resume always uses Current, never Optimal;
  Optimal is a separate high-water-mark snapshot) in `docs/PLAN.md`.
- Walkthrough: `notebooks/05_training_loop/05.5_checkpoint_resume_state_machine.ipynb`.
- Related decision: [ADR 005 — No optimizer state in checkpoints](005_no_optimizer_state_in_checkpoints.md)
  (Critical Note #3).
