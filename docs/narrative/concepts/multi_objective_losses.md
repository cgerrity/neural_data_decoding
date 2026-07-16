# Multi-objective losses

The Optimal model optimizes **four loss components at once** — reconstruction,
KL divergence, classification, and confidence (plus an optional offset/scale
term). This page explains how they are combined into a single scalar the
optimizer can minimize.

## The components

| Component | What it drives | Kernel |
|---|---|---|
| Reconstruction | the decoder reconstructs the input (NaN-masked MSE) | `training/losses/elbo.py` |
| KL divergence | the latent posterior stays near a unit Gaussian | `training/losses/elbo.py` |
| Classification | the multi-head classifier decodes the target (MIL-pooled CE) | `training/losses/classification.py` |
| Confidence | the model's self-estimated certainty stays calibrated | `training/losses/confidence.py` |

Each lives on a wildly different numeric scale, so a raw sum would be dominated
by whichever component is largest at the moment.

## EMA prior normalization

The fix (Critical Notes #6, #30) is **normalize before summing**. In
`aggregate_normalized_losses`, each component is:

1. divided by its own **running EMA magnitude** (a "prior"), bringing it to
   ≈ 1.0;
2. rescaled to a shared reference (classification's prior — the deliverable);
3. multiplied by its per-component **weight**.

The result: the *weights* — not the accidental scales — control the balance. The
priors are detached running state, threaded from batch to batch, updated on a
cadence set by `RescaleLossEpoch`.

## One total loss, three subnetworks

The components assemble into sub-totals, then one root (Critical Note #28):

```
Loss_Decoder    = reconstruction + KL + offset/scale
Loss_Classifier = classification + confidence
total           = Loss_Decoder + Loss_Classifier      # the gradient root
```

`total.backward()` is called **once**. Autograd routes each component's gradient
only to the subnetwork that produced it, while the **shared encoder accumulates
all of them** (the sum of the branch gradients, tensor-for-tensor). The
`decoder`/`classifier` sub-totals are returned for logging, not for separate
backward passes.

## Related concepts

- [VAE sampling](vae_sampling.md) — where the reconstruction + KL terms come from.
- [The confidence P-controller](the_confidence_pd_controller.md) — how the
  confidence loss is kept calibrated.
- Notebooks under `notebooks/06_loss_orchestration/` teach every component,
  the normalization, and the NaN-masked reconstruction in depth.
