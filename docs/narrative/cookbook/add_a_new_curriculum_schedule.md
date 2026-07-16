# Add a new curriculum schedule

A curriculum regime is **data, not code** — you author a YAML file and reference
it by name. No Python changes.

## Steps

1. **Pick a regime name** and find its slug. The name maps to a filename via
   `slugify_regime`:

   ```python
   from neural_data_decoding.training.schedules.library import slugify_regime
   slugify_regime("My Custom Warmup")   # -> "my_custom_warmup"
   ```

2. **Create the YAML** at `configs/schedule/<slug>.yaml` with up to three
   blocks — `weights`, `freeze`, `augmentation` — each a set of per-parameter
   waypoints:

   ```yaml
   matlab_name: "My Custom Warmup"
   weights:
     classification:
       epoch_points: [0, 10, 15]
       magnitude_points: [1.0e-2, 1.0e-2, 1.0]
     kl:
       epoch_points: [10, 100]
       magnitude_points: [1.0e-4, 1.0]
   freeze:
     classifier:
       epoch_points: [0, 10, 15]
       magnitude_points: [1.0e-2, 1.0e-2, 1.0]
   augmentation:           # shared across all std_* augmentations
     epoch_points: [5, 10, 25, 45]
     magnitude_points: [1.0e-2, 1.0, 1.0, 1.0e-2]
   ```

   Each waypoint list is `(epoch, magnitude)` pairs; the magnitude multiplies the
   base value. Values ramp linearly between waypoints and clamp outside. Note the
   one-epoch off-by-one at internal waypoints (intentional, parity-pinned —
   [The dynamic curriculum](../concepts/dynamic_curriculum.md)).

3. **Point the config at it** — set `dynamic_parameter_set: "My Custom Warmup"`
   in your target config or via `--override`.

## Semantics to know

- **Freeze magnitudes are learning-rate factors:** `1.0` = trainable, `0.0` =
  frozen, `1e-2` = slow-learn. Not booleans.
- **Load augmentation is read live** per batch; **loss weights** are snapshotted
  once per epoch.
- Keep the freeze and weight schedules **consistent** — don't down-weight a loss
  while its subnetwork stays fully trainable.

## Verify

`load_curriculum_by_name("My Custom Warmup", ...)` should build a
`CurriculumBundle` without a `FileNotFoundError` (which means the slug didn't
match a file). Trace it across epochs as in notebook
`notebooks/07_dynamic_curriculum/07.6_walkthrough_soft_three_stage_curriculum_shortened.ipynb`.
