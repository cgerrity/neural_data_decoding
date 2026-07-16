# ADR 022 — He initialization, explicit

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline builds every fully-connected layer with an explicit
initializer argument: `fullyConnectedLayer(..., "WeightsInitializer", "he")`
(`cgg_generateSimpleBlock.m:63`, `cgg_selectBottleNeck.m:62`). MATLAB's
`'he'` draws weights from `N(0, 2/fan_in)` — Kaiming-*normal*, scaled for
the fan-in, with the factor of 2 accounting for ReLU zeroing half its
inputs. It also zeros the biases.

PyTorch's `nn.Linear` does **not** default to this. Its default is
`kaiming_uniform_` with `a=sqrt(5)` — a *uniform* distribution at a
different effective scale, chosen for historical reasons — and it samples
biases from `uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))` rather than zeroing
them. So a port that leaves initialization implicit starts every learnable
FC layer from a different distribution than MATLAB, on both weights and
biases.

Critical Note #31 flagged this: the two initializers are mathematically
different distributions, the discrepancy is a T1 design-parity concern
(same architecture *includes* the same initializer), and the fix is one
explicit line per FC layer. The same note also observes that MATLAB's
recurrent layers use Glorot init, where PyTorch's default differs but is
closer, and recommends leaving those alone.

## Decision

Every `nn.Linear` layer constructed in the models package is followed
immediately by two explicit initialization calls:

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
nn.init.zeros_(layer.bias)
```

The `nonlinearity="relu"` argument is what yields the `2/fan_in` scale —
i.e. the same `N(0, 2/fan_in)` distribution MATLAB's `'he'` produces — and
`nn.init.zeros_` reproduces MATLAB's zeroed biases. This is applied
inline in each module's `__init__`, at each FC site in the variational /
Deep-LSTM pipeline:
the `Feedforward` transform in `_EncoderBlock` (`encoder.py:183`), the
`LinearBottleneck` (`bottleneck.py:114`), the decoder output layer
(`decoder.py:147`), the classifier head (`classifier.py:363`), and both
confidence heads — `TrialConfidenceHead` (`confidence_heads.py:71`) and
each per-dimension branch of `TaskConfidenceHead` (`confidence_heads.py:124`).

What the code deliberately does **not** do — and this qualifies the ADR's
title:

- **Only FC (`nn.Linear`) layers get explicit He init.** The `GRU` and
  `LSTM` transforms in `_EncoderBlock`, and the LSTM stack in the
  classifier, keep PyTorch's default recurrent initialization. This is
  intentional (Critical Note #31): MATLAB uses Glorot there, PyTorch's
  default is close enough, and it is not "He". So the port is *not*
  uniformly He-initialized — it is He on the feedforward layers only.
- **`MultiHeadClassifier` (the Milestone-A logistic head) keeps PyTorch's
  default `nn.Linear` init.** Its per-dimension output heads
  (`classifier.py:101`) are *not* re-initialized. This is an active FC
  site left on defaults, which is why the covered set above is the
  variational / Deep-LSTM pipeline — not literally every `nn.Linear` in
  the tree.
- **Explicit He matches the starting *distribution*, not the sampled
  values.** Python and MATLAB RNGs differ (see
  [ADR 001](001_tiered_parity_not_bit_exact.md)), so two freshly-built
  models are never bit-identical even with a "matched" seed. The value of
  setting He explicitly is that both sides sample from the same
  distribution, which supports T3 convergence comparison and keeps the
  architecture a faithful T1 copy — not that init makes forward passes
  bit-exact. The T2 weight-load parity test (load matched MATLAB weights,
  compare forward output) is unaffected by init because the loaded weights
  overwrite it; that test is how we *confirm* the init discrepancy carries
  no residual risk, not something init produces.

There is no shared `_init_weights` helper; the two-line idiom is repeated
at each construction site, each carrying a comment or docstring citing
Critical Note #31.

## Consequences

**Positive**

- Freshly-built Python FC layers sample from the same `N(0, 2/fan_in)`
  distribution as MATLAB, so early-epoch convergence behavior is
  comparable (T3) and the architecture stays a faithful T1 copy including
  its initializer.
- Biases start at zero on both sides, matching MATLAB and removing a
  second silent source of divergence.
- The intent is documented at every site (Critical Note #31 citations),
  so a future reader does not mistake the override for redundant noise
  over PyTorch's defaults.

**Negative**

- The two-line He + zero-bias idiom is duplicated across five modules with
  no shared helper; adding a new FC layer means remembering to repeat it,
  and omission is silent (the layer still trains, just from PyTorch's
  distribution). A `reset_parameters`-style helper would centralize it but
  was not introduced.
- Initialization parity is partial by design: recurrent layers are left on
  PyTorch defaults, so "explicit He everywhere" is not literally true and
  the codebase is not a single uniform initialization scheme.
- Explicit init does nothing for bit-exact parity; the benefit is
  distributional only, which can mislead anyone expecting matched seeds to
  reproduce MATLAB weights.

## Alternatives considered

1. **Rely on PyTorch's `nn.Linear` defaults.** Rejected: `kaiming_uniform_(a=sqrt(5))`
   is a different distribution (uniform, different scale) with non-zero
   biases, so it silently breaks T1 design parity with MATLAB's `'he'`
   from step 0. Both may train, but the port would no longer be a faithful
   copy of the MATLAB architecture.

2. **Apply He init uniformly to recurrent layers too.** Rejected: MATLAB's
   `lstmLayer`/`gruLayer` use Glorot, not He, so forcing He onto them would
   *introduce* a mismatch rather than remove one. PyTorch's recurrent
   defaults are the closer choice (Critical Note #31).

3. **Centralize init in a shared `_init_weights(module)` helper applied via
   `module.apply(...)`.** Reasonable and would cut duplication, but rejected
   for now: a blanket `apply` would need per-type branching to avoid
   touching recurrent layers, and the inline two-line form keeps the intent
   visible at each FC site next to its Critical Note #31 citation. Left as a
   possible future refactor.

4. **Chase bit-exact initial weights via a matched RNG bridge.** Rejected on
   the same grounds as [ADR 001](001_tiered_parity_not_bit_exact.md):
   MATLAB and PyTorch RNGs are not reconcilable without rewriting kernels,
   and bit-exact parity is explicitly a non-goal. Matching the distribution
   is sufficient.

## References

- Implementing code: `src/neural_data_decoding/models/encoder.py:183`
  (`_EncoderBlock` Feedforward transform),
  `src/neural_data_decoding/models/bottleneck.py:114` (`LinearBottleneck`),
  `src/neural_data_decoding/models/decoder.py:147` (decoder output),
  `src/neural_data_decoding/models/classifier.py:363` (classifier head),
  `src/neural_data_decoding/models/confidence_heads.py:71` and `:124`
  (`TrialConfidenceHead` and `TaskConfidenceHead`).
- MATLAB source of the design: `cgg_generateSimpleBlock.m:63`,
  `cgg_selectBottleNeck.m:62` (`"WeightsInitializer","he"`).
- Migration rationale: Critical Note #31 in `docs/PLAN.md`.
- Concept walkthrough: `notebooks/04_architecture/04.8_weight_initialization_he_vs_pytorch_defaults.ipynb`.
- Related decision on why distributional (not bit-exact) parity is the bar:
  [ADR 001 — Tiered parity, not bit-exact](001_tiered_parity_not_bit_exact.md).
- [He et al. 2015 — Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
  — origin of He init and the fan-in reasoning.
