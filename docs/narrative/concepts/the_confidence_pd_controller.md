# The confidence controller

The model estimates its own **confidence**, and a small controller keeps that
confidence *calibrated* over training. Despite the MATLAB name ("Autonomous
Equilibrium Controller" / "PD-controller"), the implementation is a **pure
proportional (P) controller** — three lines of arithmetic, no derivative or
integral term.

## Two confidence heads

- **Trial confidence** — reads the latent; "is this trial's signal clear?"
  (per-trial, shape `(B, 1)`).
- **Task confidence** — reads the classifier's penultimate features; "is this
  dimension decodable?" (per-dimension, shape `(B, K)`).

They combine multiplicatively: `TotalConfidence = Task × Trial` (a logical-AND —
high only when *both* are confident).

## The P-controller

Each batch, a running scalar `beta` is nudged toward keeping mean confidence at
a target of `0.5`:

```
diff     = 0.5 - mean(TotalConfidence)         # the gap
new_beta = beta_prev * (1 + diff * 1.0)        # proportional, multiplicative
new_beta = clamp(new_beta, 0.1, 10.0)          # safety rail
```

`beta` then scales the confidence loss (`loss * weight * beta`). The loop is
**negative feedback**: over-confident → `beta` shrinks → less pressure →
confidence eases back toward `0.5`; under-confident → `beta` grows. The fixed
point is exactly mean confidence `0.5` (calibration — right about half the
time).

## Why it's the highest-risk port

It's tiny, mislabeled ("PD" with no D), and its direction (a *multiplicative*,
*signed*-gap update) is easy to flip — and a sign error would not crash, it
would silently *destabilize* confidence. Only an empirical parity probe against
the MATLAB trajectory catches that. The lesson: **read the code, not the label.**

## Eq. 2 interpolation

The confidence also modulates the classification loss via an interpolated
cross-entropy: `Y' = c·Y + (1-c)·T`, so a low-confidence trial is scored against
a target it's partly *given*, and a confident mistake is penalized more.

## Related

- [Multi-objective losses](multi_objective_losses.md) — where the confidence
  loss joins the others.
- Notebooks `notebooks/06_loss_orchestration/06.6_*` (routing) and `06.7_*`
  (the controller).
