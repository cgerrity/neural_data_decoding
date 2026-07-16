# VAE sampling

The encoder is **variational**: it outputs a distribution over the latent, not a
single point. This page explains the sampling layer and its train/eval
behavior.

## The reparameterization trick

The encoder produces a mean `mu` and log-variance `logvar` per latent dimension.
The `SamplingLayer` draws a latent `z`:

- **Training:** `z = mu + eps * exp(0.5 * logvar)`, with `eps ~ N(0, 1)` fresh
  each forward. Sampling makes `z` stochastic, which regularizes the latent
  space and gives the KL term meaning. The `eps` is *outside* the network, so
  gradients still flow through `mu`/`logvar` (the reparameterization trick).
- **Evaluation:** `z = mu`, deterministic (Critical Note #35). Predictions are
  reproducible and represent the model's best estimate (the mode of the
  Gaussian).

The switch is the module's `self.training` flag, flipped by `model.train()` /
`model.eval()` recursively. Forgetting `model.eval()` at inference is a common
bug — predictions jitter run to run.

## The ELBO

The reconstruction and KL terms together form the evidence lower bound:

- **Reconstruction:** NaN-masked MSE between the decoder output and the
  NaN-preserving target. Normalized by **batch size** (not the unmasked-element
  count — an empirically-verified parity subtlety, Critical Note #38). Removed
  channels (`NaN` in the target) contribute zero via `torch.where` (not
  `mask * diff`, which would leave `NaN * 0 = NaN`).
- **KL:** `-0.5 * sum(1 + logvar - mu² - exp(logvar))`, pulling the posterior
  toward a unit Gaussian. Its weight is annealed in gradually to avoid
  posterior collapse — see [Dynamic curriculum](dynamic_curriculum.md).

## Stochastic vs deterministic placement

`EncoderOutputType` decides what the *classifier* sees: the sample `z`
(Stochastic, the Optimal choice — regularizes the classifier) or the mean `mu`
(Deterministic). The decoder always gets `z`. At inference both collapse to
`mu`, so predictions are deterministic regardless.

## Related

- [Multi-objective losses](multi_objective_losses.md) — how reconstruction + KL
  join the classification and confidence terms.
- Notebooks `notebooks/06_loss_orchestration/06.2_*`, `06.3_*`, `06.13_*`.
