# ADR 021 — EMA prior normalized to classification

**Status**: Accepted
**Date**: 2026-07-16

## Context

The multi-objective loss combines up to five components — Reconstruction,
KL, Classification, Confidence, and OffsetAndScale — that live on wildly
different natural scales (reconstruction can be ~1000× the classification
cross-entropy). A raw weighted sum is therefore dominated by whichever
component happens to be largest, and the configured per-component weights
stop meaning anything.

MATLAB's fix (`cgg_getLossInformation.m` + `cgg_processLossComponent`)
is EMA prior normalization: each component is divided by a running average
of its *own* magnitude (its "prior"), flattening every component to ~1.0.
But flattening everything to unit scale throws away the absolute magnitude
of the objective, so MATLAB then re-inflates every component by a single
shared multiplier — `Rescale_Value` — before applying weights. The design
question this ADR records is: **which component's prior should be that
shared multiplier?**

MATLAB's answer (`cgg_getLossInformation.m:117-134`): `Rescale_Value`
defaults to `Prior_Loss_Classification`, falling back to
`Prior_Loss_Reconstruction` when classification is inactive. Every
component gets rescaled onto classification's magnitude. This is the
behavior the port must reproduce, and it is captured in Critical Note #30.

## Decision

The port reproduces the MATLAB reference-selection rule in
`_determine_rescale_value`. To be precise about what the code actually
does — the title is shorthand, not the mechanism:

- The classification EMA prior is **not itself renormalized**. Rather, each
  component is divided by its *own* EMA prior (in `_process_component`,
  step "normalize"), then multiplied by a single detached `rescale_value`
  (step "rescale"), then weighted. Classification's prior is that shared
  `rescale_value`, so every component — including classification itself —
  ends up expressed on classification's magnitude.

- `_determine_rescale_value` picks the reference by a fallback chain:
  1. If classification is active this batch **and** an effective
     classification prior exists → return classification's prior.
  2. Else if reconstruction is active with an effective prior → return
     reconstruction's prior.
  3. Else → return `1.0` (no rescaling).

- "Effective prior" means the stored EMA prior, or — when none has been
  accumulated yet (first iteration) — the current batch's *detached* loss.
  This is the first-iteration bootstrap: on step 1 every `loss / prior`
  is ≈ 1.0, so weight × reference is what remains.

- The reference is a **constant scaling factor in the gradient** — the
  returned tensor is always `.detach()`-ed. The model cannot game
  normalization by shrinking the reference instead of the loss.

- `rescale_value` is computed **before** any prior is EMA-updated this
  batch (Critical Note #30: it uses the *pre-update* priors). The call
  site orders this deliberately — `_determine_rescale_value` runs before
  the per-component `_process_component` calls that mutate the priors.

The rationale for classification being the primary reference: classification
is the deliverable. The pipeline's output is decoding accuracy;
reconstruction and KL are means to a better classifier. Anchoring the shared
scale to classification makes each weight read as "how much does this
auxiliary loss matter *relative to the task*." The reconstruction fallback
exists precisely for the case where there is no task yet — a pure-autoencoder
Stage 1 (two-stage lifecycle) where `classification_loss is None`.

## Consequences

**Positive**

- Weights express balance *relative to the deliverable*, not accidental raw
  scales — the intended knob behavior.
- Graceful degradation: Stage 1 autoencoder pretraining (no classifier)
  still normalizes correctly via the reconstruction fallback; a
  no-components-active edge case returns a `1.0` scalar rather than erroring.
- Line-for-line parity with MATLAB's `Rescale_Value` selection, verified
  empirically rather than from the plan text.

**Negative**

- The reference is selected per batch from which components are *active*
  (loss is not `None`) — not from a static config flag. A caller that
  forgets to thread priors, or passes components inconsistently across
  steps, can silently flip the reference and change the loss balance.
- Because the divide uses each component's freshly EMA-updated prior while
  the rescale uses the *pre-update* classification prior, classification's
  own `loss / prior × rescale` is only exactly `loss` at steady state; there
  is a small transient asymmetry during rapid scale drift. This is faithful
  to MATLAB, not a bug, but it surprises readers who expect the anchor
  component to pass through untouched.
- The choice bakes in an assumption that classification is always the
  scientific deliverable. A future objective where reconstruction is the
  product would need this reference rule revisited.

## Alternatives considered

1. **Reconstruction as the primary reference.** Rejected: reconstruction is
   auxiliary. Scaling the classifier's contribution to reconstruction's
   magnitude would make weights read relative to a means-to-an-end, and
   reconstruction's magnitude drifts most over training as the decoder
   improves. (It remains the *fallback* only when classification is absent.)

2. **A fixed constant reference (e.g. always `1.0`).** Rejected: leaving
   every component at ~1.0 discards the objective's absolute magnitude and
   couples the effective learning rate to the component count. It also
   diverges from MATLAB, breaking parity.

3. **The mean (or max) prior across active components as the reference.**
   Rejected: it has no principled tie to the deliverable, is not what MATLAB
   does, and would make the shared scale wander as auxiliary components come
   and go under the curriculum.

4. **A config-selectable reference component.** Rejected as premature: it
   adds a knob nothing currently needs, and the classification→reconstruction
   fallback already covers every configuration the pipeline runs today.

## References

- Implementing code: `src/neural_data_decoding/training/losses/multi_objective.py`
  — `_determine_rescale_value` (reference selection), `_process_component`
  (normalize → rescale → weight pipeline), and `aggregate_normalized_losses`
  (call site that computes `rescale_value` before updating priors).
- MATLAB source ported: `cgg_getLossInformation.m:117-134` (`Rescale_Value`
  default + fallback) and `cgg_processLossComponent`.
- Migration Critical Note **#30** — EMA prior normalization uses
  Classification's prior as the reference, with pre-update priors.
- Notebook: `notebooks/06_loss_orchestration/06.4_the_ema_prior_normalization_deep_dive.ipynb`
  — §2.3 "Why classification is the reference."
- Related decision: [ADR 009](009_ema_prior_cadence_via_rescale_loss_epoch.md)
  — *when* the EMA priors update (`RescaleLossEpoch` cadence).
- Concept background: [multi-objective losses](../concepts/multi_objective_losses.md).
