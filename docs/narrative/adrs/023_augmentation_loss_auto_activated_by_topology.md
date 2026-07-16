# ADR 023 — Augmentation loss auto-activated by topology

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline supports a learnable augmentation pathway: a decoder
branch (built by `cgg_generateAugmentBlock.m` when the config requests
learnable offset/scale) emits per-window estimates of a scale factor and an
offset, and `cgg_lossOffsetAndScale.m` compares them to targets derived from
the input itself (`range(X) - 1` and `median(X)` for the default
`AugmentEquation='mX+b+X'`).

The decisive detail is *how the loss orchestrator decides to run this term*.
MATLAB's `cgg_lossComponents.m` (lines ~368-375) does not read a dedicated
"compute the augmentation loss" boolean. Instead it **inspects the decoder's
layer graph** for the augment layers `reshape_offset_Augment` /
`reshape_scale_Augment`, and only invokes the loss when those layers are
present (and `WeightOffsetAndScale` is not NaN). The config that requested the
augment block already stamped those layers into the graph; the orchestrator
rediscovers them by topology rather than by a second, independent flag.

The naive Python port would introduce exactly that redundant flag — e.g. a
`compute_offset_and_scale_loss` bool consulted at loss time. That duplicates
state: the head could exist without the flag, or the flag could be set with no
head, and the two would silently drift out of sync.

## Decision

Gate the augmentation loss on the **presence of the augmentation head in the
composite's module tree**, not on a separate loss-layer flag. Concretely, what
the code does today:

- **Construction (build time).** `build_variational_composite` builds a
  `LearnableOffsetScale` head only when the config sets
  `want_learnable_offset` **or** `want_learnable_scale`. This is the direct
  analog of MATLAB's config stamping the augment block into the decoder graph.
- **Activation (forward time).** `VariationalComposite.forward` emits
  `offset_scale = self.learnable_offset_scale(z)` **iff**
  `self.learnable_offset_scale is not None` — gated purely by the module's
  presence, never by re-reading a config value.
- **Aggregation.** `aggregate_total_loss` / `aggregate_normalized_losses` add
  `w("offset_and_scale") * offset_and_scale_loss` **iff** the passed
  `offset_and_scale_loss is not None`.
- **Topology probe.** `find_learnable_offset_scale(module)` walks the module
  tree with `isinstance` and returns the head or `None` — the 1:1 mirror of
  MATLAB inspecting the layer graph.

So at the loss-orchestration layer there is no dedicated boolean; the head's
existence *is* the switch.

**Honest state of the wiring (this is not fully end-to-end yet).** The loss
kernel `offset_and_scale_loss` and the finder `find_learnable_offset_scale`
are currently exercised **only** by unit tests
(`tests/unit/test_offset_and_scale.py`). The runtime training loop
`train_one_epoch` wires reconstruction / KL / classification / confidence
only — it does not read `out.offset_scale`, never calls the kernel, and never
passes `offset_and_scale_loss=` into the aggregators. The auto-activation
*pattern* is implemented across the plumbing (config → head → forward output →
aggregator parameter, all presence-gated), but the final hop (the epoch loop
consuming `out.offset_scale`) is not connected, so the term does not yet
contribute to any gradient in end-to-end training. Separately, the CLI weight
`offset_and_scale` defaults to `0.0`, so even once wired the term contributes
nothing unless the config sets `weight_offset_and_scale`.

## Consequences

**Positive**

- Single source of truth is the decoder topology. No flag can drift out of
  sync with whether the augment head actually exists.
- Parity reasoning is 1:1 with MATLAB: both discover the term by inspecting the
  graph the config built, rather than by a second independent switch.
- Presence-based gating is uniform across the forward output and both
  aggregators, so an augment head placed anywhere in the tree "just works"
  without new config keys at the loss layer.

**Negative**

- Two-layer gating can mislead: construction is still config-driven
  (`want_learnable_*`), so "no config flag" is true only at the
  loss-orchestration layer, not for the whole pipeline.
- The kernel and finder are presently test-only; a reader who sees the kernel,
  the finder, and the aggregator support may wrongly assume the term flows in
  end-to-end training. It does not yet — closing the `train_one_epoch` hop is
  required before the augmentation loss can affect training.
- Silent no-op risk: with the weight defaulting to `0.0` (and even after
  wiring), a present head still contributes nothing without an explicit
  `weight_offset_and_scale`, which can read like a bug.

## Alternatives considered

1. **Dedicated loss-time config flag** (`compute_offset_and_scale_loss` read in
   the loss orchestrator). Rejected: duplicates the head's existence as
   separate boolean state that can drift, and diverges from MATLAB's
   graph-inspection pattern.
2. **Always compute the loss, weight it to zero when disabled.** Rejected:
   wastes forward compute building targets and head outputs that need not
   exist, forces the head to be present unconditionally, and invites spurious
   NaN handling when there is no augment head at all.
3. **Register the loss via an explicit callback/registry at composite-build
   time.** Rejected as heavier than needed: module presence already carries the
   signal, and MATLAB has no such registry to mirror.

## References

- Loss kernel + targets: `src/neural_data_decoding/training/losses/offset_and_scale.py`
- Decoder head + topology probe: `src/neural_data_decoding/models/layers/offset_scale.py` (`LearnableOffsetScale`, `find_learnable_offset_scale`)
- Composite build-time gating + forward presence-gate: `src/neural_data_decoding/models/composite.py` (`build_variational_composite`, `VariationalComposite.forward`)
- Aggregator presence-gating: `src/neural_data_decoding/training/losses/multi_objective.py` (`aggregate_total_loss`, `aggregate_normalized_losses`)
- Current test-only exercise: `tests/unit/test_offset_and_scale.py`
- CLI weight default: `src/neural_data_decoding/cli.py` (`weight_offset_and_scale` → `0.0`)
- MATLAB behavioral reference: `cgg_lossComponents.m` (orchestrator graph inspection), `cgg_lossOffsetAndScale.m` (loss), `cgg_generateAugmentBlock.m` (augment block build)
- Migration Critical Note #32 (topology auto-activation); batch-size normalization per Critical Note #38
- Milestone CC.6
- Notebook: `notebooks/06_loss_orchestration/06.11_single_total_loss_three_subnetworks.ipynb`
- Related: [ADR 002 — Pythonic structure over MATLAB mirror](002_pythonic_structure_over_matlab_mirror.md)
