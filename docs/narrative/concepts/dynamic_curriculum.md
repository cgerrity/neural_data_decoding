# The dynamic curriculum

Rather than training with fixed hyperparameters, the Optimal model uses a
**curriculum**: several hyperparameters vary *per epoch* according to a
schedule. This enacts "reconstruct first, classify later" and prevents the
latent from collapsing.

## The three levers

All three are per-epoch schedules built on one interpolation primitive
(`piecewise_anneal_value`, a port of `cgg_calculateDynamicValue`):

| Lever | Controls | Consumed |
|---|---|---|
| **LoadParameters** | data-augmentation strengths (channel offset, white noise, random walk, time shift) | **live**, inside the Dataset's `__getitem__` (Critical Note #8) |
| **LossWeights** | per-component loss weights, incl. KL annealing | snapshotted **once per epoch** |
| **FrozenNetwork** | per-subnetwork learning-rate factors | applied at **epoch start** |

A single `CurriculumBundle.update(epoch)` advances all three (and the KL
base-anneal). The `Schedule` object holds the per-parameter waypoints; presets
live as YAML in `configs/schedule/`.

## The "Soft Three-Stage Curriculum"

The Optimal preset choreographs a smooth handoff:

1. **Reconstruct (epochs ~1–10):** clean data, full reconstruction weight,
   classifier near-frozen and near-zero-weighted. The autoencoder learns.
2. **Handoff (~10–20):** noise ramps up, classification weight and the
   classifier's learning rate ramp up together.
3. **Classify (~20+):** reconstruction weight and the encoder's learning rate
   decay; the classifier owns the objective on a now-stable latent.

## Freezing is LR-scaling, not `requires_grad`

A common misconception: freezing does **not** toggle `requires_grad`. It scales
each subnetwork's learning rate (`group["lr"] = base_lr * factor`), because
MATLAB's `setLearnRateFactor` uses a `1e-2` "slow-learn" factor that a boolean
`requires_grad` cannot express. See `notebooks/07_dynamic_curriculum/07.5_*`.

## The interpolator's off-by-one

`piecewise_anneal_value` preserves a MATLAB `Epoch - 1` quirk: at each internal
waypoint the ramp reaches only `(span-1)/span` of the way, snapping to the
target one epoch later. This is intentional (parity-pinned), not a bug — see
`notebooks/07_dynamic_curriculum/07.2_*`.

## Related

- [The training lifecycle](the_training_lifecycle.md) — where the curriculum is applied each epoch.
- [Add a new curriculum schedule](../cookbook/add_a_new_curriculum_schedule.md) — how to author a regime.
