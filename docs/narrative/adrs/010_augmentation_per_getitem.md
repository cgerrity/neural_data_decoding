# ADR 010 — Augmentation per `__getitem__`

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline augments each trial with synthetic noise (channel
offset, white noise, random walk, time shift) whose *strength* is a
scheduled, per-epoch quantity — the `LoadParameters` produced by
`cgg_generateLoadParameters_v2` and applied by
`cgg_generateDataAugmentationSignal.m`. Two behaviors are implicit in that
design and easy to lose in a port:

- **Re-randomization.** MATLAB reads each trial through a `fileDatastore`,
  which re-invokes the read function on *every* access. The augmentation
  noise is therefore freshly drawn each time a trial is read — it is never
  computed once and reused.
- **Live schedule read.** That same read function reads the *current
  epoch's* `LoadParameters` state. The magnitudes are not baked in when the
  datastore is created; they reflect whatever the curriculum has set for the
  epoch in progress.

A naive PyTorch port breaks one or both silently. The two classic traps:
(1) cache the augmented tensor per trial ("compute once, reuse") — which
kills the augmentation benefit and freezes the noise; and (2) snapshot the
schedule's magnitudes into the `Dataset` at construction or at epoch start —
which decouples the curriculum from the data so `schedule.update(epoch)`
never reaches a batch. Neither trap raises an error; both quietly degrade
training. These are recorded as migration Critical Note #7 (re-randomization)
and Critical Note #8 (live schedule read).

## Decision

The `Dataset` owns both the base signal and the augmentation magnitudes, and
re-derives the augmentation on **every** `__getitem__`, reading the schedule
live. Concretely, in `data/dataset.py`:

- `SyntheticTrialDataset.__getitem__` calls `_apply_augmentation` on every
  access, but **only when `self.load_schedule is not None`**. The training
  `Dataset` is built with a schedule; the validation and test datasets pass
  `load_schedule=None`, so their augmentation branch is skipped entirely and
  they stay clean by construction.
- **Re-randomization (Critical Note #7).** `_apply_augmentation` calls
  `additive_augmentation_signal(..., rng=self._aug_rng)`, where `self._aug_rng`
  is a single long-lived `numpy.random.Generator` seeded once at construction
  (`augmentation_seed`, default `0`). Fresh noise is drawn on every call. The
  method returns a *new* array (`features_np + noise`) and never writes back
  to `self._features`, so nothing is cached — the same trial read twice gets
  two different augmentations. Because the generator is seeded once and drawn
  in call order, the sequence is deterministic and reproducible (this is what
  the ~1e-6 seeded augmentation-parity test asserts) while still being fresh
  per read.
- **Live schedule read (Critical Note #8).** Inside `_apply_augmentation`,
  the per-component magnitude is read at call time via
  `_current_or_none(name)`, which returns `sched.current(name) if name in
  sched else None`. `self.load_schedule` is held as a *reference* to the live
  schedule object, not a snapshot. So when the training loop calls
  `schedule.update(epoch)` at the top of an epoch, the very next
  `__getitem__` — and therefore the next batch — sees the new strength, with
  **no `DataLoader` rebuild and no `Dataset` re-instantiation**. A key absent
  from the schedule yields `None`, which the augmentation kernel treats as
  disabled (contributes 0), matching MATLAB's NaN-disable semantics.

**Honest note on the title vs. the mechanism.** "Augmentation per
`__getitem__`" is the *contract*, not the literal MATLAB mechanism. This
`Dataset` holds its base features in memory (`self._features`) rather than
re-reading each trial from a `.mat` file per access the way MATLAB's
`fileDatastore` does. The per-read contract is preserved regardless: the base
signal is identical on each read, and only the *additive* augmentation is
re-drawn and re-scaled from the live schedule every call. The disk-read
mechanism is an implementation detail; the re-randomization and live-read
guarantees are what parity depends on, and those hold here.

## Consequences

**Positive**

- The curriculum is coupled to the data *live*: a per-epoch
  `schedule.update` is reflected on the next batch, so the clean → noisy →
  clean regime works without rebuilding the pipeline.
- No silent parity loss from caching: every read sees fresh noise, giving the
  real augmentation benefit that few-trial neural data needs.
- Reproducible for tests: one seeded generator yields a deterministic draw
  sequence, so seeded augmentation-parity checks are stable.
- Validation and test data stay clean by construction (`load_schedule=None`)
  — no accidental augmentation of the metrics.

**Negative**

- Augmentation cost is paid on every read; there is no memoization. This is
  acceptable because the additive-noise kernel is cheap next to the
  forward/backward pass.
- Correctness hinges on the `Dataset` holding a *reference* to the live
  schedule, not a snapshotted magnitude. A refactor that snapshots the
  `std_*` values at construction would silently break the curriculum — this
  is called out as a common error in notebook 07.3 §5.
- The long-lived generator lives on the `Dataset` instance. With a
  multi-worker `DataLoader` (`num_workers > 0`), the generator is
  fork-copied into each worker; per-worker reseeding must be arranged to
  avoid correlated draws across workers. Single-process loading (the default
  path here) is unaffected.

## Alternatives considered

1. **Cache the augmented tensor per trial (compute once, reuse).** Rejected.
   It freezes the noise so every epoch sees identical corruption — no
   augmentation benefit — and it cannot track schedule changes. This is
   exactly the silent-parity-loss trap of Critical Note #7.

2. **Snapshot the schedule magnitudes into the `Dataset` once per epoch
   (set-once).** Rejected. It forces the training loop to poke the `Dataset`
   each epoch, adds coupling, and diverges from MATLAB, where the read
   function reads current `LoadParameters` live. The live-read is simpler and
   reflects the change on the very next batch (Critical Note #8).

3. **Apply augmentation as a transform outside the `Dataset` (in
   `collate_fn` or a `DataLoader` transform reading a global).** Rejected. It
   scatters ownership of the base signal and the augmentation magnitudes, and
   obscures the per-trial re-randomization. The `Dataset` is the natural
   owner of both.

4. **Re-seed the generator on every `__getitem__` with a per-call seed.**
   Rejected. Seeding identically each call produces identical noise for every
   trial (a documented common error), and a bespoke per-index seeding scheme
   adds complexity for no gain — a single long-lived generator draws fresh
   noise naturally.

## References

- Implementing code: `src/neural_data_decoding/data/dataset.py`
  (`SyntheticTrialDataset.__getitem__`, `_apply_augmentation`,
  `_current_or_none`) and `src/neural_data_decoding/data/augmentation.py`
  (`additive_augmentation_signal`, `generate_time_shift_samples`).
- Schedule factory read live at call time:
  `src/neural_data_decoding/training/schedules/factory.py`
  (`make_load_schedule`).
- Migration Critical Note #8 (LoadSchedule live read by `Dataset`) and its
  sibling Critical Note #7 (augmentation re-randomization per `__getitem__`),
  in `docs/PLAN.md`.
- Walkthrough and common-error list:
  `notebooks/07_dynamic_curriculum/07.3_load_parameters.ipynb` (§2.4 live-read
  contract, §5 common errors).
- MATLAB references: `cgg_generateDataAugmentationSignal.m`,
  `cgg_generateLoadParameters_v2`, and `fileDatastore` per-access read
  semantics.
- Related concept page: [dynamic curriculum](../concepts/dynamic_curriculum.md).
