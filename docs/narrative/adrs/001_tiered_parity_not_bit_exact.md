# ADR 001 — Tiered parity, not bit-exact

**Status**: Accepted (Milestone 0)
**Date**: 2026-05-26

## Context

The neural_data_decoding pipeline is a Python port of the existing MATLAB
pipeline in `Processing_Functions_cgg/`. A natural first instinct is to aim
for "1-to-1 mathematical and behavioral parity" between the two — i.e., the
Python pipeline produces bit-identical outputs to the MATLAB pipeline for
identical inputs.

Bit-exact parity is **not achievable** in practice, and pretending it is
produces an endless chain of false-positive bug hunts. The reasons:

- ADAM, BatchNorm momentum conventions, and default initializers differ between
  MATLAB Deep Learning Toolbox and PyTorch in ways that cannot be made
  identical without rewriting the native kernels.
- Random number generators differ between the two languages even given the
  "same" seed; MATLAB uses Mersenne Twister with a different state layout from
  CPython, and PyTorch has its own ATen-side generator.
- Floating-point reduction order is non-deterministic on the GPU; even within
  a single language, two consecutive runs with the same seed can differ
  bitwise.

If we tried to enforce bit-exact parity, every observed mismatch would be a
crisis that the implementer would have to chase. The vast majority of those
mismatches are irrelevant noise — the model converges to the same statistical
distribution regardless.

## Decision

Define a **tiered parity model** with four levels. The Python pipeline must
satisfy each tier appropriate to the component being tested.

| Tier | Goal | Verification |
|------|------|--------------|
| **T1 — Design parity** | Same architecture topology, same loss components, same hyperparameter surface, same data preprocessing pipeline. | Code review + architecture-graph diffing. |
| **T2 — Single-step numerical parity** | Forward pass on identical input + weights matches within tolerance (1e-5 fp32, 1e-3 after BatchNorm). | `pytest` golden-vector tests; weights ported from MATLAB `.mat` checkpoints. |
| **T3 — Statistical parity** | Same convergence behavior; accuracy distributions match across seeds (KS test or paired-bootstrap CI). | 5–10 seed runs each side; compare per-epoch validation accuracy curves. |
| **T4 — Output-format parity** | Python output `.mat` files load cleanly into MATLAB analysis scripts and produce equivalent downstream figures. | Run MATLAB `DATA_cggAllNetworkEncoderResults` and `FIGURE_cggAllNetworkEncoderResults` on Python output; visual diff. |

**Bit-exact parity is explicitly NOT a goal.** Tests should never assert
`y_python == y_matlab` element-wise.

## Consequences

**Positive**

- The parity-debugging effort focuses on behaviors that actually matter for
  downstream science: same model topology, same convergence, same downstream
  plots.
- Implementers have a clear standard to apply per component: choose the tier
  appropriate for the question being asked.
- Tests run faster and are less flaky because they don't have to chase
  numerical noise.

**Negative**

- It is theoretically possible for a subtle bug to hide behind the tolerance
  thresholds. We mitigate this with golden-vector T2 tests on every loss kernel
  and a T3 multi-seed convergence check at each milestone boundary.
- Reviewers must understand which tier applies to which test — the same
  question ("does this match MATLAB?") has different right answers depending
  on the context.

## Alternatives considered

1. **Aim for bit-exact parity.** Rejected for the reasons above.

2. **Aim for statistical-only parity (T3 only).** Rejected because it would let
   single-component bugs hide behind noise from other components. The
   per-component T2 golden-vector tests are critical for localizing failures.

3. **Aim for T4-only (downstream-plot equivalence).** Rejected for the same
   reason as (2), and additionally because downstream plots are noisy
   summaries; a bug could change individual trial-level predictions without
   moving the aggregate plot meaningfully.

## References

- Migration plan: `Plans/neural_data_decoding_plan.md` — "Parity Goals (Tiered)" section.
- Pytest parity gates: `tests/parity/` (currently scaffolded; populated per milestone).
- Discussion of related tradeoffs:
  [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html).
