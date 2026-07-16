# ADR 005 — No optimizer state in checkpoints

**Status**: Accepted
**Date**: 2026-07-16

## Context

The neural_data_decoding pipeline is a Python port of the MATLAB pipeline in
`Processing_Functions_cgg/`. Checkpointing exists for one purpose: so that an
interrupted training run — a preempted cluster node, a crash, a `Ctrl+C` — can
resume from the most recent save rather than restarting from scratch.

The MATLAB pipeline made a deliberate choice here. `cgg_saveIterationInformation.m`
**commented out** the line that would have persisted `OptimizerVariables.mat`.
The stated motivation was file size: the optimizer's per-parameter state roughly
doubles what a checkpoint has to hold, and the user chose to omit it. The
consequence baked into MATLAB is that on resume the optimizer is reinitialized —
an interrupted-then-resumed run follows a slightly different trajectory than an
uninterrupted one.

A faithful port has to decide between two options: match MATLAB's intentional
omission, or "fix" it by persisting optimizer state the way a textbook PyTorch
checkpoint would. Those two choices produce measurably different resume behavior,
so the decision is not cosmetic.

## Decision

The Python checkpoint persists **model weights and progress bookkeeping only —
never optimizer state.** Concretely, the payload written by
`save_current_checkpoint` contains exactly four keys: `model_state_dict`,
`epoch`, `iteration`, and `best_metric`. There is no `optimizer_state_dict` key.
The Optimal snapshot written by `save_optimal_checkpoint` is even leaner
(`model_state_dict`, `epoch`, `metric`).

On resume, `load_current_checkpoint` loads the weights **in place** into the
caller's model and returns a `CheckpointState` carrying the bookkeeping. It does
not touch an optimizer — the caller must instantiate a fresh one. Because the
default optimizer is `AdamW` (Critical Note #5), a fresh optimizer means AdamW's
first- and second-moment (momentum / variance) estimates restart from zero. The
first few iterations after a resume therefore behave like the very beginning of
training — no accumulated momentum — and then stabilize as the moments rebuild.

This matches MATLAB exactly, and it is documented at three levels: the
`checkpoint.py` module docstring, the notebook, and this ADR.

## Consequences

**Positive**

- Smaller checkpoints. For AdamW the omitted state is two moment tensors per
  parameter, so persisting it would roughly double the file size — the exact
  cost the MATLAB pipeline set out to avoid.
- Behavioral parity with MATLAB. An interrupt-then-resume in Python drifts the
  same way it does in MATLAB, so parity comparisons across a resume boundary are
  apples-to-apples rather than confounded by a Python-only "full checkpoint."
- A trivially simple save/load surface: one payload shape, no optimizer coupling,
  and a resume contract that is easy to reason about and test.

**Negative**

- Interrupt-then-resume is **not** bit-identical to uninterrupted training. The
  moment estimates must rebuild, so the post-resume trajectory diverges slightly.
  Parity tests for the "interrupt + resume" path must account for this expected
  drift rather than asserting exact continuity.
- A brief warmup after every resume: early post-resume steps are noisier until
  AdamW's moments repopulate. For a 500-epoch run resumed rarely this is
  negligible; a pathologically frequently-resumed run would pay the cost more
  often.
- The behavior surprises anyone expecting standard PyTorch "full checkpoint"
  semantics (weights **and** optimizer). Mitigated by the module docstring, the
  notebook, and this record making the omission explicit.

## Alternatives considered

1. **Persist optimizer state (standard PyTorch full checkpoint).** Rejected. It
   breaks parity with MATLAB — the resume drift would differ from the reference
   pipeline — doubles the checkpoint size the user deliberately kept small, and
   contradicts an explicit design choice. The small resume-drift is absorbed
   under statistical (T3) parity, not bit-exact parity (see
   [ADR 001](001_tiered_parity_not_bit_exact.md)), so persisting the optimizer
   buys no parity benefit.

2. **Persist optimizer state behind an opt-in config flag.** Rejected for now.
   It adds a second code path and a new parity variable with no current consumer,
   while the default would still have to match MATLAB. Worth revisiting only if a
   real workload resumes often enough that the moment-rebuild warmup becomes a
   measured cost.

3. **Warm-restart the optimizer with a short learning-rate ramp after resume** to
   mask the cold-start of the moments. Rejected. It introduces a schedule wrinkle
   that MATLAB does not have, muddying parity, to smooth over a warmup that is
   negligible at the expected resume cadence.

## References

- Implementing code: `src/neural_data_decoding/training/checkpoint.py` —
  `save_current_checkpoint` writes the payload keys `model_state_dict`, `epoch`,
  `iteration`, `best_metric` (no `optimizer_state_dict`); `load_current_checkpoint`
  restores weights in place and leaves the caller to build a fresh optimizer.
- Migration plan: Critical Note #3 (`docs/PLAN.md`) — MATLAB's
  `cgg_saveIterationInformation.m` commented out the `OptimizerVariables.mat`
  save; optimizer omitted deliberately to reduce file size.
- Default optimizer rationale: Critical Note #5 (`docs/PLAN.md`) — `AdamW` is the
  default ADAM-equivalent path, which is why resume resets AdamW's moment estimates.
- Notebook: `notebooks/05_training_loop/05.5_checkpoint_resume_state_machine.ipynb`
  — §2.4 (why optimizer state is not saved) and §5 (post-resume warmup is
  expected, not a bug).
- Related: [ADR 001](001_tiered_parity_not_bit_exact.md) — the resume-drift is
  handled at the statistical-parity tier (T3), not asserted bit-exact (T2).
