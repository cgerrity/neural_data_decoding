# ADR 019 — Single total loss, three subnetworks

**Status**: Accepted
**Date**: 2026-07-16

## Context

The Optimal model is not one network but three wired into a shared trunk: an
**encoder** that produces a latent `z`, a **decoder** branch that reconstructs
the input from `z`, and a **classifier** branch that decodes the target from
`z`. The encoder is shared — its latent feeds both branches — so it sits
downstream of every loss component.

MATLAB assembles the objective from per-branch sub-totals and then takes a
single gradient (`cgg_lossComponents.m:491–504`):

```matlab
Loss_Decoder    = Loss_Reconstruction + Loss_KL + Loss_OffsetAndScale;
Loss_Classifier = Loss_Classification + Loss_Confidence;
Loss_Encoder    = Loss_Decoder + Loss_Classifier;
Gradients.{Encoder,Decoder,Classifier} = dlgradient(Loss_Encoder, .Learnables);
```

The names `Loss_Decoder` and `Loss_Classifier` are misleading: they are
**telemetry intermediate sums**, not subnetwork-specific gradients. All three
networks gradient-flow from the *single* root `Loss_Encoder`. The naive port —
call backward once per branch sub-total, or give each subnetwork its own
optimizer — either double-counts the shared encoder's gradient or forces
`retain_graph` recomputation, and it fights the shared trunk with inconsistent
updates. Critical Note #28 flags this as a Medium-likelihood / High-impact
mistake.

## Decision

Assemble one scalar and call `.backward()` on it exactly once; let autograd do
the routing.

What the code actually does today:

- **`aggregate_normalized_losses`** (the Milestone C+ path) materializes the
  three-way split faithfully: `loss_decoder = recon + KL + offset/scale`,
  `loss_classifier = classification + confidence`, and
  `loss_encoder = loss_decoder + loss_classifier`. It returns `loss_encoder`
  as `total`; the `decoder` and `classifier` sub-totals ride along on the
  `NormalizedLossBreakdown` **for logging only**. Nothing calls `.backward()`
  on them.
- **`aggregate_total_loss`** (the Milestone A/B path) does *not* build the
  decoder/classifier grouping at all — it flat weight-sums whatever components
  are passed straight into a single `total`. The `= Loss_Decoder +
  Loss_Classifier` decomposition is therefore only ever *materialized* in the
  normalized aggregator. Both paths produce one 0-D scalar.
- **`training/loop.py`** backprops that single scalar and nothing else. The
  active-model branches (Milestone A `aggregate_total_loss`, C+
  `aggregate_normalized_losses`) each set one `total_loss`, and the loop calls
  `.backward()` on it. Autograd walks the shared graph once, depositing each
  component's gradient only in the parameters that produced it — the decoder
  gets reconstruction gradient, the classifier gets classification gradient,
  and the **shared encoder accumulates the element-wise sum of both branches**
  (`∇(a + b) = ∇a + ∇b`), because it lies on a live path to every component.

One honest nuance about "backward once": under hardware-aware gradient
accumulation the loop splits a mini-batch into chunks and calls
`(total_loss * chunk_weight).backward()` **once per chunk**, accumulating into
`.grad` before a single `optimizer.step()`. So "one backward" means *one
backward per total scalar* — there may be several accumulating backwards per
optimizer step, but there is never a separate backward on the decoder vs. the
classifier sub-total. The gradient root is always the single encoder scalar.

## Consequences

**Positive**

- The autograd graph *is* the routing table: a parameter receives a
  component's gradient iff a path connects them. No manual "this loss trains
  that layer" bookkeeping, no three optimizers.
- The shared encoder gets the **combined** signal — "make the latent both
  reconstruct and classify well" — which is exactly what a single backward on
  the sum delivers.
- One graph traversal per chunk: no `retain_graph`, no recomputing the
  encoder's backward per branch, no risk of a stray `zero_grad` between branch
  backwards erasing the accumulation.
- The `decoder`/`classifier` sub-totals are preserved for telemetry, matching
  MATLAB's per-subnetwork logging without paying for separate gradients.

**Negative**

- The MATLAB field names (`Loss_Decoder`, `Loss_Classifier`, and even
  `Loss_Encoder`) invite the reader to think they are subnetwork-scoped
  gradients. They are not, and the code's naming inherits that trap; the
  docstrings and notebook 06.11 have to actively correct it.
- Correctness now hinges on there being exactly one `.backward()` target. A
  reviewer must verify no code path backprops a sub-total — the PLAN risk
  table calls this out as a required review check.
- The two aggregators diverge in structure (flat sum vs. explicit three-way
  split), so "what is `total`?" has a slightly different answer per milestone,
  even though both collapse to a single backpropped scalar.

## Alternatives considered

1. **Call `.backward()` separately on `Loss_Decoder` and `Loss_Classifier`.**
   Rejected: without `retain_graph=True` the second call errors (the shared
   graph is freed); with it, the encoder's backward is recomputed and its
   gradient is effectively double-counted through the shared trunk. This is the
   exact failure Critical Note #28 warns against.
2. **Give the encoder, decoder, and classifier separate optimizers and
   per-subnetwork gradients.** Rejected: the shared encoder would receive
   inconsistent updates from independent backwards instead of the single summed
   signal it needs. One optimizer over all parameters plus one
   `total.backward()` is both simpler and correct.
3. **Literally mirror MATLAB's three `Gradients.{Encoder,Decoder,Classifier}`
   structs.** Rejected as a non-Pythonic mirror (see
   [ADR 002](002_pythonic_structure_over_matlab_mirror.md)): PyTorch autograd
   already produces per-parameter gradients from one root over
   `model.parameters()`, so reconstructing three explicit gradient containers
   would add machinery that buys nothing.

## References

- Assembly of the three sub-totals into the single gradient root:
  `src/neural_data_decoding/training/losses/multi_objective.py`
  (`aggregate_normalized_losses` → `loss_decoder` / `loss_classifier` /
  `loss_encoder`; the Milestone A/B `aggregate_total_loss` flat sum).
- The single backward on `total_loss`: `src/neural_data_decoding/training/loop.py`.
- Migration Critical Note **#28** ("All three networks backprop from the same
  total loss"): `docs/PLAN.md`, and the corresponding risk row in the PLAN
  risks table.
- MATLAB source: `cgg_lossComponents.m:491–504`.
- Teaching notebook:
  `notebooks/06_loss_orchestration/06.11_single_total_loss_three_subnetworks.ipynb`.
- Concept page: [Multi-objective losses](../concepts/multi_objective_losses.md).
