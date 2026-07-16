# ADR 020 — Confidence loss: five subtleties

**Status**: Accepted
**Date**: 2026-07-16

## Context

The Optimal model self-reports **confidence** and uses it to modulate its own
classification loss. In MATLAB this lives in `cgg_lossConfidence.m` and
`cgg_getConfidenceLossInformation.m`, and the migration plan flags it as the
**single highest-risk port in the whole codebase** (Critical Note #29). The risk
is not size — the controller is three lines — it is that five distinct, easy-to-
miss behaviors are packed into one kernel, and getting any one wrong does **not**
crash. Training simply diverges: the budget regularizer stops balancing the loss
attenuation and confidence collapses to all-0 or all-1. Only an empirical
parity probe against the MATLAB trajectory catches a mistake.

Two things about the MATLAB source actively mislead a literal porter:

- The controller function is named `"Autonomous Equilibrium Controller"` /
  `"PD-controller"`, but there is **no derivative and no integral term** — it is
  pure proportional feedback.
- `cgg_getConfidenceLossInformation.m` reads `mean(TotalConfidence, "all")`,
  which looks like a full `(B, T, K)` mean; in reality the confidence tensors are
  already last-timestep-reduced to `(B, K)` by the time they arrive (Critical
  Note #36), so the mean averages `B·K` elements, not `B·T·K`.

We had to decide exactly which behaviors to reproduce, and which to expose as
opt-in flags versus hardcode as defaults.

## Decision

Reproduce all five subtleties in `apply_confidence_routing` and its helpers, with
comments mapping each code block to its subtlety. What the code does **today**:

1. **Multiplicative Task × Trial conjunction (Eq. 1).** When both heads are
   active, `TotalConfidence = TaskConfidence * TrialConfidence`, element-wise, on
   the **last-timestep** slice of each stream. It is a logical-AND: the total is
   high only when both a clean trial *and* a decodable dimension agree. With only
   one head active the conjunction is skipped and that stream *is* the total.

2. **ConfidenceDropout (default 0.5), separate from network dropout.** With
   probability `1 - confidence_dropout` each entry is **reset to 1** (not zeroed).
   A parallel "dropped" confidence path is built. The default is **asymmetric**
   (`symmetric_dropout=False`, MATLAB parity): the **dropped** path feeds only the
   Eq. 2 interpolation, while the **undropped** path feeds the per-stream budget
   regularizer *and* the Beta controller. Resetting to 1 biases the interpolation
   toward "keep predicting" (confidence 1 → no interpolation).

3. **Prediction-toward-truth interpolation (Eq. 2).** The classifier prediction
   is blended with its target by the dropped total confidence:
   `Y' = c·Y + (1 - c)·T`. A low-confidence trial is scored against a target it
   is partly handed, so its classification loss is pulled toward zero. This is
   **the mechanism**, not postprocessing; the budget regularizer (each stream's
   loss pushes confidence toward 1) is what stops the degenerate "always unsure →
   no loss" solution.

4. **Stop-gradient on the historical EMA.** The running dataset-level confidence
   history is `.detach()`-ed before it enters the blend, and each freshly computed
   EMA value is detached before it becomes the next batch's history. Only the
   current batch's contribution carries gradient; without this, gradient would
   leak backward through the entire training history. The Beta controller state is
   likewise detached — the network must not be able to game the controller.

5. **BatchFraction-governed cadence (Eq. 7).** The dataset blend is
   `updated = history·(1 - γ) + batchMean·γ`, with `γ = batch_fraction` — the
   fraction of the dataset in this minibatch. It is a **variable** coefficient, not
   a fixed EMA rate: a full-dataset batch (`γ = 1`) replaces history outright; a
   tiny batch barely moves it.

**The 1/γ correction (Eq. 10) is opt-in, off by default.** The blend leaks `γ`
into every gradient reaching the confidence head (because only `batchMean·γ`
carries gradient, `∂updated/∂head = γ`), so shrinking the batch silently weakens
the confidence objective. Setting `want_batch_correction=True` divides each
stream loss by `γ`, and `(1/γ)·γ = 1` cancels it exactly — but **only because**
of the subtlety-4 stop-gradient. It is gated behind `want_dataset_confidence`
(no blend → no `γ` to cancel) and defaults to `False`; `want_dataset_confidence`
itself defaults to `True`. So the title's "with the 1/γ correction" describes an
available knob, not the default path.

The controller (`_update_confidence_beta`) is recorded plainly as a **pure
P-controller** despite the "PD" name: `diff = 0.5 - mean(undropped total)`,
`beta ← beta·(1 + diff·1.0)`, clamped to `[0.1, 10.0]`, detached. Fixed point is
mean confidence 0.5 (calibration). See
[the confidence controller](../concepts/the_confidence_pd_controller.md).

## Consequences

**Positive**

- Behavioral parity with MATLAB at T2: each subtlety is independently
  golden-vector tested, plus an end-to-end test, so a regression localizes to one
  subtlety rather than surfacing as a mysterious training divergence.
- The heavily-annotated kernel makes the "read the code, not the label" lesson
  (pure-P despite "PD", last-timestep despite `"all"`) explicit for future
  readers.
- Exposing `want_batch_correction` and `symmetric_dropout` as flags lets us ablate
  the two most debatable behaviors without forking the kernel.

**Negative**

- The kernel is intricate: five interacting behaviors, an asymmetric dropped/
  undropped split, and a control loop that is easy to sign-flip. A flip would not
  crash — it would silently destabilize confidence — so the parity probe is
  load-bearing, not optional.
- The asymmetric-dropout default (dropped for interpolation, undropped for the
  budget and the controller) is genuinely surprising and must be kept in sync with
  MATLAB by hand; `symmetric_dropout=True` exists only as an unvalidated ablation.
- `want_batch_correction` defaulting off means batch-size-invariant confidence
  gradients are *not* the out-of-the-box behavior; callers who change batch size
  and expect identical confidence dynamics must opt in.

## Alternatives considered

1. **Fold everything into "an uncertainty estimate" and simplify.** Rejected —
   Critical Note #29 is explicit that this is not a generic uncertainty head; the
   exact interpolation + budget + controller interplay is what keeps confidence
   calibrated. Dropping any piece changes training behavior silently.

2. **Trust the "PD-controller" name and implement a real PD (or PID) loop.**
   Rejected — the MATLAB has no D or I term. Adding one would diverge from parity
   and chase a derivative that does not exist. Recorded as pure-P instead.

3. **Always apply the 1/γ correction (make it the default).** Rejected — MATLAB
   guards it behind `WantBatchCorrection`, and forcing it on would break single-
   step parity for the common `γ = 1` full-batch case and double-correct if the
   confidence weight is scaled elsewhere. Kept as an opt-in flag.

4. **Use the dropped confidence everywhere (symmetric).** Rejected as the default
   — MATLAB passes the *undropped* streams to the budget regularizer and the
   controller, using dropped confidence only for the Eq. 2 interpolation. Symmetric
   behavior is retained solely as an ablation switch.

5. **Let the EMA history carry gradient (skip the stop-grad).** Rejected — gradient
   would leak through the entire training history, and the clean `∂updated/∂head =
   γ` that makes the Eq. 10 correction exact would no longer hold.

## References

- Implementing kernel: `src/neural_data_decoding/training/losses/confidence.py`
  (`apply_confidence_routing`, `_compute_confidence_stream_loss`,
  `_update_confidence_beta`, `ConfidenceHistory`).
- MATLAB source ported: `cgg_lossConfidence.m` (blend + streams) and
  `cgg_getConfidenceLossInformation.m` (Beta controller).
- Migration plan: Critical Note #29 (the five subtleties) and Critical Note #36
  (last-timestep sequence convention).
- Notebooks: `notebooks/06_loss_orchestration/06.6_confidence_routing.ipynb`
  (conjunction + routing), `notebooks/06_loss_orchestration/06.7_the_confidence_pd_controller.ipynb`
  (the pure-P controller), `notebooks/06_loss_orchestration/06.9_per_batch_prior_correction.ipynb`
  (the γ blend + 1/γ correction).
- Concept pages: [the confidence controller](../concepts/the_confidence_pd_controller.md)
  and [multi-objective losses](../concepts/multi_objective_losses.md) (where the
  confidence loss joins the single total loss).
