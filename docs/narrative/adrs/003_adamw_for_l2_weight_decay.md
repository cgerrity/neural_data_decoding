# ADR 003 — AdamW for L2 weight decay

**Status**: Accepted
**Date**: 2026-07-16

## Context

MATLAB regularizes weights through `trainingOptions(..., 'L2Regularization',
L2Factor)`. Internally this adds the L2 penalty's gradient, `L2Factor · w`, to
the **data gradient** *before* the optimizer's update step — the regularization
is **coupled** into the gradient the optimizer consumes:

```
grad_total = grad_data + L2Factor · w      % L2 folded into the gradient
w          = adamUpdate(w, grad_total)     % Adam then rescales the SUM
```

For plain SGD this coupled form is *exactly* "decay every weight by `lr · λ`
after the data step" — coupled and decoupled L2 coincide. But the MATLAB
configs the port must reproduce default to `Optimizer = ADAM`
(`configs/base.yaml`), and on Adam the two forms **diverge**. Adam divides
every gradient by `√v` (a per-parameter running estimate of gradient
magnitude). When `L2Factor · w` is folded into the gradient, that penalty term
is divided by `√v` too, so each weight's effective decay becomes entangled with
its own gradient history rather than being a clean, uniform `λ`. The
regularization strength turns into an accident of gradient scale.

ADR 001 established that MATLAB is the behavioral reference and parity is the
default goal. This decision records one of the very few places where the port
**deliberately does not match MATLAB**, because the MATLAB recipe (Adam +
coupled L2) is a known-suboptimal form the field has since improved on.

## Decision

The port uses **`torch.optim.AdamW`** (decoupled weight decay) for the
ADAM-equivalent path and forwards MATLAB's `L2Factor` straight into the
optimizer's `weight_decay`. There is **no L2 term anywhere in the loss kernel**
— no `training/losses/weight_decay.py` was ever created, and nothing adds
`λ‖w‖²` to the total loss sum. The regularization lives entirely in the
optimizer.

Concretely, what the code does today:

- `resolve_optimizer_factory` maps the config string `"ADAM"` →
  `torch.optim.AdamW` and `"SGDM"` → `torch.optim.SGD(momentum=0.9)`
  (`training/freezing.py`).
- `_build_optimizer` reads `wd = float(cfg.l2_factor)` and passes it as
  `weight_decay=wd` to whichever factory is selected — for the single-group
  path, the per-module freeze-group path
  (`build_optimizer_with_module_groups`), and the Stage-1 two-stage optimizer
  alike (`cli.py`).
- `AdamW` applies `lr · λ · w` directly to each weight, untouched by `√v`, so
  every weight decays at the honest rate `λ`.

On `SGDM`, coupled and decoupled decay coincide (the SGD case above), so
`SGD(weight_decay=…)` is **exact parity** with MATLAB's grad-side L2 there — the
divergence is confined to the Adam path, where AdamW is the mathematically
cleaner form.

Note the naming trap: notebook `06.8`'s title, "L2 inside the loss kernel," is
how MATLAB *thinks* about the penalty (grad-side), not where it lives in this
codebase. The port moves it out of the loss and into the optimizer, where
decoupled weight decay belongs.

## Consequences

**Positive**

- Weight decay is uniform `λ` across all parameters, independent of gradient
  history — the regularization means what it says.
- Follows the modern standard (Loshchilov & Hutter 2019); AdamW generalizes
  better than Adam + coupled L2 in the literature.
- One knob (`l2_factor`) threads through every optimizer build — single-group,
  per-module freeze groups, and the Stage-1 handoff — so decay is consistent
  across frozen and live parameter groups.
- The `SGDM` path is exact parity, so choosing that optimizer reproduces MATLAB
  bit-for-bit on the regularization term.

**Negative**

- A user porting MATLAB hyperparameters may expect Adam + grad-side L2 and be
  surprised the training dynamics differ. Mitigated by notebook `06.8`, which
  walks through the divergence explicitly, and by this ADR.
- `torch.optim.Adam`'s own `weight_decay` argument is the *coupled* form (it
  adds L2 to the gradient internally). Only `AdamW` is decoupled; the class
  name is the entire difference, and picking the wrong class silently
  reintroduces the MATLAB behavior. This is a latent footgun for future edits.
- Bit-exact parity with MATLAB on the Adam path is intentionally unattainable
  — consistent with ADR 001's tiered-parity stance, but it means the L2 term
  is not a valid target for T2 numerical parity tests under `ADAM`.

## Alternatives considered

1. **Reproduce MATLAB exactly: `torch.optim.Adam` + grad-side `λ·w`.** Rejected.
   This faithfully mirrors MATLAB but bakes in the known-suboptimal Adam +
   coupled-L2 recipe — the `√v` rescale entangles each weight's decay with its
   gradient history. Blindly matching the reference here would import a defect.

2. **`torch.optim.Adam(weight_decay=…)`.** Rejected. PyTorch's `Adam`
   `weight_decay` is *also* the coupled form (L2 added to the gradient
   internally), so this is functionally alternative (1) with a shorter call. It
   is specifically the thing the notebook warns against.

3. **Add an explicit `λ‖w‖²` term to the loss sum** (a
   `training/losses/weight_decay.py` kernel, as the plan's early file tree
   sketched). Rejected. On Adam an in-loss L2 term still flows through `√v` (it
   becomes part of the gradient), so it reproduces the coupled behavior, not
   decoupled decay; and if combined with an optimizer `weight_decay` it would
   regularize twice. The file was never created.

4. **A gradient-hook escape hatch** that injects `λ·w` under any optimizer, for
   users who explicitly want MATLAB grad-side semantics. Considered as an
   optional path in the plan; not implemented, because `SGDM` already gives
   exact parity where it matters and `ADAM` deliberately should not.

## References

- Optimizer factory + weight-decay threading: `resolve_optimizer_factory` and
  `build_optimizer_with_module_groups` in
  `src/neural_data_decoding/training/freezing.py`.
- `l2_factor → weight_decay` wiring: `_build_optimizer` and the Stage-1
  optimizer build in `src/neural_data_decoding/cli.py`
  (`weight_decay=float(cfg.l2_factor)`); default `l2_factor: 1.0e-4` and
  `optimizer: ADAM` in `configs/base.yaml`.
- Migration Critical Note #5 (L2 weight decay — use the modern Python
  approach) in `docs/PLAN.md`.
- Notebook: `notebooks/06_loss_orchestration/06.8_l2_inside_the_loss_kernel.ipynb`.
- Prior decision on parity philosophy: [ADR 001 — Tiered parity, not
  bit-exact](001_tiered_parity_not_bit_exact.md).
- [Decoupled Weight Decay Regularization (Loshchilov & Hutter,
  2019)](https://arxiv.org/abs/1711.05101) — introduces AdamW and names this
  exact distinction.
