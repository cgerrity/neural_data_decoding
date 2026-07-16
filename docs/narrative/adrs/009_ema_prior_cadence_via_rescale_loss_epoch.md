# ADR 009 — EMA prior cadence via RescaleLossEpoch

**Status**: Accepted
**Date**: 2026-07-16

## Context

The loss orchestrator normalizes each loss component by a running EMA
of that component's magnitude (see [ADR 001](001_tiered_parity_not_bit_exact.md)
for the parity model this is verified under). Those EMA priors have to be
*updated* as training proceeds, and MATLAB does not update them on a fixed
schedule — the update frequency is a tunable knob, `cfg_Encoder.RescaleLossEpoch`.

In MATLAB (`cgg_getLossInformation`) the semantics of that single integer are:

- `RescaleLossEpoch == 0` (the Optimal/production default) — the EMA updates
  on **every iteration**.
- `RescaleLossEpoch == 1` — the EMA updates **once per epoch**.
- `RescaleLossEpoch > 1` — the EMA updates **once, every `N` epochs**.

The subtlety is that this is a *two-granularity* rule. `RescaleLossEpoch`
is compared against the epoch index to decide whether *this epoch* is an
update epoch, but the actual EMA write happens (or doesn't) inside the
per-iteration loop. A naive port that only gates at epoch granularity, or
only at iteration granularity, gets the `> 1` case wrong. Getting the cadence
wrong is a known training-instability risk (`docs/PLAN.md` risk table:
"EMA prior normalization implemented wrong (esp. `RescaleLossEpoch` cadence)").

## Decision

Split the cadence into two functions across the two granularities, keyed by
a three-value strategy string — `"every_iter"`, `"first_iter_only"`, `"never"`.

**Epoch granularity.** Once per epoch, `fit_supervised` calls
`_update_priors_strategy_for(epoch, rescale_loss_epoch)` to pick this
epoch's strategy. What the code actually computes:

- `rescale_loss_epoch <= 0` → `"every_iter"`.
- `rescale_loss_epoch == 1` → `"first_iter_only"` (returned on *every* epoch).
- `rescale_loss_epoch > 1` → `"first_iter_only"` on update epochs and
  `"never"` on all other epochs. The update-epoch test is
  `(epoch + 2) % rescale_loss_epoch == 1`, which reproduces MATLAB's
  1-indexed `mod(Epoch + 1, N) == 1` given Python's 0-indexed `epoch`
  (MATLAB `Epoch = epoch + 1`).

**Iteration granularity.** Inside `train_one_epoch`, each batch calls
`_should_update_priors(strategy, batch_idx)`, which returns `True` for
`"every_iter"`, `batch_idx == 0` for `"first_iter_only"`, and `False`
for `"never"`.

Composing the two: `RescaleLossEpoch == 1` fires on batch 0 of every epoch
(= once per epoch); `RescaleLossEpoch > 1` fires on batch 0 of every `N`-th
epoch (= once every `N` epochs) and is inert on the intervening epochs; and
`RescaleLossEpoch == 0` fires on every batch of every epoch.

Honest note on the title: the three-mode strings are the *implementation's
internal per-epoch outcomes*, not a 1:1 relabeling of the three
`RescaleLossEpoch` cases. The `> 1` case produces two different strategy
strings across the training run — `"first_iter_only"` on its update epochs
and `"never"` elsewhere — which is exactly how "every `N` epochs" is
realized. There is no dedicated "every N epochs" mode; that behavior is an
emergent property of selecting the strategy fresh each epoch.

A legacy `update_priors: bool` parameter is retained for older call sites
that predate the strategy string: when `update_priors_strategy` is `None`,
`train_one_epoch` maps `True → "every_iter"` and `False → "never"`. Call
sites that pass an explicit strategy (all current ones, via `fit_supervised`)
ignore the bool.

## Consequences

**Positive**

- The `> 1` cadence is correct because the two granularities are handled by
  two functions with a single shared vocabulary, rather than one over-loaded
  conditional. The epoch-level function never has to reason about batches,
  and the batch-level function never has to reason about epochs.
- `RescaleLossEpoch == 0`, the production default, is the fast path
  (`"every_iter"` → unconditional `True`) with no per-batch modular arithmetic.
- The strategy string is testable in isolation: all three modes, and the
  off-by-one in the `(epoch + 2)` translation, can be unit-tested without
  spinning up a training loop.

**Negative**

- The cadence lives in two files (`lifecycle.py` selects, `loop.py` applies).
  A reader tracing "when do priors update?" must follow the strategy string
  across the call boundary rather than reading one function.
- The `(epoch + 2) % N == 1` expression is opaque without the index-base
  comment; it is a translation of MATLAB's 1-indexed convention and is easy
  to misread as an arbitrary constant.
- The retained `update_priors` bool is a second, redundant way to express
  the two extreme modes, kept only for backward compatibility.

## Alternatives considered

1. **A single per-iteration predicate `should_update(epoch, iteration)`**
   reading `RescaleLossEpoch` directly (as the migration note literally
   suggested). Rejected: it re-derives the epoch-level decision on every
   batch and forces epoch arithmetic into the hot loop, and it makes the
   three cadence outcomes harder to test in isolation.

2. **A boolean-only interface (`update_priors: True/False`)**. Rejected: it
   cannot express `RescaleLossEpoch > 1` ("every `N` epochs") at all — only
   the two extremes. It survives solely as a backward-compat shim.

3. **Precomputing an explicit per-epoch update schedule** (e.g. a boolean
   list of length `num_epochs`) up front. Rejected as unnecessary state:
   the strategy is a pure function of `(epoch, rescale_loss_epoch)` and is
   cheap to recompute each epoch, and materializing it would duplicate the
   curriculum's own per-epoch recompute pattern.

## References

- Epoch-level strategy selection: `_update_priors_strategy_for` in
  `src/neural_data_decoding/training/lifecycle.py` (defined ~L598; called
  from `fit_supervised` ~L252).
- Iteration-level gate: `_should_update_priors` in
  `src/neural_data_decoding/training/loop.py` (defined ~L466; called from
  `train_one_epoch` ~L212). The `update_priors_strategy` docstring on
  `train_one_epoch` documents the three modes and their MATLAB mapping.
- Migration Critical Note #6 (`docs/PLAN.md`) — "EMA prior normalization is
  controlled by `RescaleLossEpoch`"; risk-table entry mandates testing all
  three cadence modes (`0`, `1`, `> 1`).
- MATLAB source: `cgg_getLossInformation` (EMA state + `RescaleLossEpoch`
  cadence).
- Companion notebooks on the EMA mechanism these updates drive:
  `notebooks/06_loss_orchestration/06.4_the_ema_prior_normalization_deep_dive.ipynb`
  and
  `notebooks/06_loss_orchestration/06.12_ema_prior_normalization_deep_dive.ipynb`
  (order-of-operations, first-iteration degeneracy, and the shared
  classification reference the cadence governs the update frequency of).
