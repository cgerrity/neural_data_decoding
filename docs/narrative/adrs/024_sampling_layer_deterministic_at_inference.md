# ADR 024 — Sampling layer deterministic at inference

**Status**: Accepted
**Date**: 2026-07-16

## Context

The encoder is **variational**: instead of a single latent point it emits the
latent statistics `[mu | logSigmaSq]` concatenated along the channel axis. The
`SamplingLayer` splits them and produces the latent code `Z` that the decoder
(and, depending on `EncoderOutputType`, the classifier) consume.

The MATLAB reference `cgg_samplingLayer.m` deliberately behaves **differently in
training vs inference** by splitting the work across two methods:

- `forward` (line 94, used by `trainnet` during training) draws
  `epsilon = randn(...)` and returns the proper reparameterized sample
  `Z = epsilon .* sigma + mu`.
- `predict` (line 44, used by `predict` / `minibatchpredict` during
  validation and test) draws `epsilon = randn(...)`, then **zeroes it**
  (`epsilon = epsilon * 0`), so `Z = epsilon .* sigma + mu` collapses to
  `Z = mu` — deterministic.

This split is easy to lose in a port. The textbook PyTorch VAE samples in
*both* modes, and PyTorch collapses MATLAB's two methods into a single
`forward`, so a naive translation would sample at inference too. The failure is
silent and expensive: the same trial would produce different classification
probabilities on each call, so reported test-set accuracy — and the record of
"which trials the model got right" — would jitter run to run, defeating
reproducibility (Critical Note #35).

## Decision

`SamplingLayer.forward` branches on the module's `self.training` flag and
reproduces the two MATLAB behaviors:

- **Train mode** (`self.training` is `True`): draw `eps = torch.randn_like(mu)`
  and return `z = mu + eps * torch.exp(0.5 * logvar)` — the reparameterization
  trick, noise scaled by the standard deviation.
- **Eval mode** (`self.training` is `False`): return `z = mu` directly — the
  mode of the latent Gaussian, deterministic and reproducible.

Two honesty points about what the code actually does today:

1. The Python eval path does **not** draw `eps` at all — it assigns `z = mu`
   directly, rather than mirroring MATLAB's draw-then-multiply-by-zero. The
   value of `z` is identical either way (both equal `mu`), so the behavioral
   (T2) parity on `z` holds exactly; the only divergence is RNG-state
   consumption — MATLAB burns one `randn` draw at inference, Python burns none.
   Since `z` is the only observable output, this has no effect on predictions.
2. Both branches return the full tuple `(z, mu, logvar)` **regardless of mode**.
   `mu` and `logvar` are always emitted because the ELBO's KL term needs them,
   even though only training uses the sampled `z` distinctly.

The layer never sets `self.training` itself; it reads the flag that PyTorch
maintains and that `model.train()` / `model.eval()` flip recursively across all
submodules. A direct consequence: at eval time the `EncoderOutputType`
Stochastic-vs-Deterministic distinction collapses, because both feed the
classifier `mu` once `z == mu`.

## Consequences

**Positive**

- Inference is reproducible — identical inputs give identical predictions, so
  test-set accuracy and per-trial correctness are stable across runs.
- Predictions represent the model's best estimate (the mode of the latent
  Gaussian), not a random draw from its uncertainty.
- Training retains the stochastic regularization that keeps the latent space
  smooth and gives the KL term meaning.
- The behavior matches MATLAB's `predict` path, keeping the eval-time forward
  pass inside the T2 single-step parity envelope (see
  [ADR 001 — Tiered parity](001_tiered_parity_not_bit_exact.md)).

**Negative**

- Correctness now depends on the caller remembering `model.eval()` before any
  validation or prediction pass. Forgetting it silently re-enables sampling
  (alongside Dropout and BatchNorm's train behavior) and reintroduces jitter.
- A symmetric footgun exists: after a mid-epoch validation pass the loop must
  call `model.train()` again, or the remainder of training runs deterministically
  with no sampling.
- Anyone who genuinely wants stochastic inference (e.g. Monte-Carlo uncertainty)
  cannot get it in eval mode without extra plumbing — `eval()` disables exactly
  that draw by design.

## Alternatives considered

1. **Always sample (textbook VAE, sample in both modes).** Rejected —
   nondeterministic inference breaks reproducibility of test-set evaluation and
   diverges from MATLAB's `predict` behavior.

2. **Mirror MATLAB literally: draw `randn` then multiply by zero at inference.**
   Rejected — the multiply-by-zero is a numeric no-op that only wastes an RNG
   draw and adds a line of dead arithmetic. Assigning `z = mu` directly yields
   the same `z` and is clearer.

3. **Expose a flag to force stochastic predictions at inference.** Rejected as
   the default — deterministic-mean inference is the standard VAE behavior and
   what parity requires. Such a mode could be added later behind an explicit
   opt-in without changing the default.

4. **Return only `z` in eval mode (drop `mu` / `logvar`).** Rejected — the loss
   orchestration (KL term) and the confidence heads read `mu` / `logvar`
   regardless of mode, so both must always be returned.

## References

- Implementing code: `src/neural_data_decoding/models/layers/sampling.py` —
  `SamplingLayer.forward`, the `if self.training:` branch (lines 106–112).
- MATLAB reference: `cgg_samplingLayer.m` — `forward` (line 94, training),
  `predict` (line 44, inference).
- Migration Critical Note #35 in `docs/PLAN.md`.
- Notebook `notebooks/06_loss_orchestration/06.13_sampling_layer_deterministic_at_inference.ipynb`.
- Concept: [VAE sampling](../concepts/vae_sampling.md).
