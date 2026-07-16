# ADR 017 — NaN-masked reconstruction loss

**Status**: Accepted
**Date**: 2026-07-16

## Context

Recording channels that are dropped during preprocessing are not deleted from
the on-disk `.mat` arrays — deleting them would ragged the tensor shapes across
sessions. Instead they are stored as `NaN`: a channel-shaped hole in an
otherwise dense array. That single storage choice creates two conflicting
requirements for the variational autoencoder:

- The **encoder must never see a `NaN`.** `NaN` is contagious under IEEE
  arithmetic — the first `matmul` would spread it across every output feature
  and the whole forward pass would collapse to `NaN`.
- The **reconstruction loss must know exactly where the `NaN`s are**, so it does
  not penalize the decoder for failing to reconstruct channels that were never
  recorded.

MATLAB satisfies both by touching two different tensors. The
`sequenceInputLayer` runs `cgg_setNaNToValue(x, 0)` as its normalization
function (`cgg_constructNetworkArchitecture.m:127-129`), zeroing the `NaN`s
*before* the encoder; the loss separately computes
`0.5 * l2loss(Y, T, Mask=~isnan(T))` against the **original, still-NaN** target.

The highest-risk ambiguity in the whole port was the **normalization
denominator** of that `l2loss`. The migration plan's note suggested dividing by
`mask.sum()` — the number of unmasked elements — which is what "masked mean loss"
intuitively implies. An empirical probe of MATLAB's `l2loss` showed that is
wrong: it divides by the **batch size** regardless of how many elements the mask
keeps. Using `mask.sum()` instead would silently rescale the reconstruction
gradient by a data-dependent factor and drift the port off parity with no error
or crash to flag it.

## Decision

Two tensors flow through the pipeline, and the reconstruction loss is
normalized by batch size — matching what the code does today:

1. **`NaNToZero` at the encoder head (the input tensor).** `NaNToZero` is a
   stateless `nn.Module` whose `forward` is
   `torch.where(torch.isnan(x), full_like(x, value), x)` with `value=0.0` by
   default (`0` is the population mean of Z-scored data, a neutral substitute).
   It replaces **only** `NaN` — `±inf` pass through untouched, matching
   `cgg_setNaNToValue`. This is layer (a): the encoder receives a fully finite
   input.

2. **NaN-preserving target in the loss (a different tensor).** The
   reconstruction loss is passed the **original** target with its `NaN`s intact.
   Note plainly: the loss function does **not** zero any `NaN` itself — it
   *derives* the mask `~torch.isnan(y_target)` from those `NaN`s. If a caller
   pre-scrubbed the target, the mask would be all-ones and the masking would be
   a no-op. The two-tensor contract is load-bearing.

3. **Masked MSE via `torch.where`, normalized by batch size.**
   `masked_mse_reconstruction_loss` computes:

   ```python
   mask = ~torch.isnan(y_target)
   diff = torch.where(mask, y_pred - y_target, torch.zeros_like(y_pred))
   return 0.5 * (diff**2).sum() / y_pred.shape[batch_dim]
   ```

   Masked positions are zeroed with `torch.where`, **not** `mask * diff` —
   because `NaN · 0 = NaN` in IEEE, a multiply would leave the `NaN` in place
   and poison the sum. The denominator is `y_pred.shape[batch_dim]` (the batch
   axis size), **not** `mask.sum()`. This reproduces
   `0.5 * l2loss(Y, T, Mask=~isnan(T))`.

The MAE variant (`masked_mae_reconstruction_loss`) is identical except it sums
absolute differences and omits the `0.5` factor (MATLAB's `l1loss` is a plain
sum-of-absolutes, whereas `0.5 * l2loss` is half-sum-of-squares by convention);
it uses the same batch-size normalization. `compute_reconstruction_loss`
dispatches between them on a case-insensitive `"MSE"`/`"MAE"` string.
`per_channel_reconstruction_loss` runs the same masked MSE per channel slice but
returns **detached** scalars — telemetry only, never backpropagated (Critical
Note #33).

## Consequences

**Positive**

- The encoder is guaranteed a finite input and the decoder is never penalized
  for absent channels — the two failure modes the `NaN` storage scheme creates
  are both closed.
- The batch-size denominator keeps the reconstruction gradient's scale on parity
  with MATLAB, so the reconstruction/KL/classification loss balance ports over
  unchanged.
- `torch.where` masking is `NaN`-safe by construction; there is no silent `NaN`
  leak into the summed loss.
- The MSE and MAE paths share one normalization rule, so a future loss-type
  switch inherits the correct behavior.

**Negative**

- The batch-size (not `mask.sum()`) denominator is **counterintuitive** for a
  loss called "masked" — a maintainer's reasonable first instinct is to divide
  by the unmasked-element count, which would be a silent parity regression. The
  module docstring and Critical Note #38 exist specifically to guard this.
- Correctness depends on an **implicit contract between two call sites**: the
  encoder input must be NaN-zeroed *and* the loss target must stay
  NaN-preserving. Neither function can enforce the other's half; wiring them to
  the same tensor would break masking with no error.
- Deriving the mask from `NaN` means any *unintended* `NaN` in the target (e.g.
  a genuine data bug) is silently absorbed as a "removed channel" rather than
  raising.

## Alternatives considered

1. **Normalize by `mask.sum()` (unmasked-element count).** Rejected: the
   empirical `l2loss` probe (`Y=[1 2;3 4;5 6]`, `T=0` except `T(2,2)=NaN` →
   masked sum-of-squares `75`, `l2loss == 37.5 == 75/2`, not `75/5`) proves
   MATLAB divides by batch size. This was the plan's suggested value and it was
   wrong.

2. **Zero the `NaN`s once, up front, and feed one tensor to both the encoder
   and the loss.** Rejected: it destroys the mask. The loss can only find the
   removed channels because the target still carries `NaN`; a pre-zeroed target
   yields an all-true mask and penalizes the decoder on channels that were never
   recorded.

3. **Mask with `mask * diff` instead of `torch.where`.** Rejected: `NaN · 0 =
   NaN` in IEEE, so the masked positions survive the multiply and poison the
   `.sum()`. `torch.where` selects a real `0` for masked positions and never
   evaluates the `NaN` difference into the reduction.

4. **`torch.nan_to_num` for the input transform.** Rejected: it also rewrites
   `±inf`, whereas `cgg_setNaNToValue` replaces only `NaN`. `NaNToZero` uses
   `torch.where(isnan, …)` to match MATLAB exactly and leave `±inf` visible.

## References

- Reconstruction + KL kernels: `src/neural_data_decoding/training/losses/elbo.py`
  (`masked_mse_reconstruction_loss`, `masked_mae_reconstruction_loss`,
  `compute_reconstruction_loss`, `per_channel_reconstruction_loss`).
- Input transform (layer (a)): `src/neural_data_decoding/models/layers/nan_to_zero.py`
  (`NaNToZero`).
- Migration Critical Note #38 (two-layered NaN handling; batch-size
  normalization, verified empirically) and Critical Note #33 (per-channel
  reconstruction is detached telemetry).
- Walkthrough notebook: `notebooks/06_loss_orchestration/06.10_nan_masked_reconstruction.ipynb`.
- Concept page: [VAE sampling](../concepts/vae_sampling.md) — the ELBO section
  covers the same batch-size / `torch.where` subtleties.
