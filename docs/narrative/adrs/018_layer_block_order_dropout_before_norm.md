# ADR 018 — Layer block order: dropout before norm

**Status**: Accepted
**Date**: 2026-07-16

## Context

Each level of the Simple-branch encoder is a small block: a transform
(GRU / LSTM / Feedforward) followed by up to three optional stages —
dropout, normalization, and an activation. The MATLAB source that this
port mirrors, `cgg_generateSimpleBlock.m:116–121`, appends those stages
to the layer array in a fixed sequence:

```matlab
if Dropout > 0;          layers = [layers; dropoutLayer(Dropout)];       end
if WantNormalization;    layers = [layers; layerNormalizationLayer];     end
if ~isempty(Activation); layers = [layers; activationLayer(Activation)]; end
```

That produces the per-block order **Transform → Dropout → Norm →
Activation**. This is *not* the arrangement most deep-learning references
recommend. The conventional order is **Transform → Norm → Activation →
Dropout**, in which normalization acts on the raw transform output and
dropout is applied last, after the nonlinearity. MATLAB's order differs
in one conspicuous way: **dropout comes before norm** (and before the
activation), rather than last.

The four stages do not commute. Normalizing before vs. after an
activation, or dropping activations before vs. after normalization,
yields a genuinely different function of the same weights and inputs —
not merely a cosmetic reshuffle. So the natural instinct to "clean up"
the order to the textbook arrangement would be a silent change to the
model's training dynamics, and a MATLAB-trained checkpoint would not
reproduce under the reordered block. Critical Note #27 flagged this as a
parity landmine and asked for it to be pinned in an ADR.

## Decision

Preserve MATLAB's exact block order. The Python port assembles and runs
each block as **Transform → Dropout → Norm → Activation**, matching the
MATLAB source layer-for-layer, and does not adopt the conventional order.

What the code actually does today (`models/encoder.py`, class
`_EncoderBlock`):

- `__init__` pre-builds all four slots so the order cannot drift and the
  forward path stays branch-free: the transform layer; then
  `nn.Dropout(dropout)` (or `nn.Identity` when `dropout == 0`); then
  `nn.LayerNorm(hidden_size)` (or `nn.Identity` when
  `want_normalization` is `False`); then the activation module (or
  `nn.Identity` for `''`).
- `forward` applies them in exactly that sequence: transform, then
  `self.dropout`, then `self.norm`, then `self.activation`. A docstring
  and inline comment name Critical Note #27 at the call site so a future
  reader does not "correct" it.

An honest caveat about what the title implies vs. what runs in
production: the ordering is only *observable* when two or more of the
optional stages are active. The production "Optimal" GRU config sets
`dropout=0.5`, `want_normalization=False`, `activation=''` — so norm and
activation are both `nn.Identity` and dropout is the only live stage.
For that config the block is effectively Transform → Dropout, and the
order distinction is a no-op. The order becomes load-bearing for configs
that turn on norm and/or activation alongside dropout (for example a
Feedforward block with `want_normalization=True` and `activation='ReLU'`),
which is exactly where a "fixed" order would silently diverge from
MATLAB. The order is preserved unconditionally so that every config,
not just the production one, transfers faithfully.

## Consequences

**Positive**

- MATLAB-trained checkpoints load and reproduce, because the composed
  function is identical block-for-block.
- The decision is un-editable by construction: slots are fixed in
  `__init__` and the `forward` has no `if dropout > 0`-style branching to
  tempt a reordering.
- Parity tests can verify a single block's activations against MATLAB one
  block at a time, since the sub-modules are kept separate rather than
  fused into a single `nn.GRU(num_layers=N)`.

**Negative**

- The order looks "wrong" to anyone applying textbook intuition, so it
  needs the docstring, the inline Critical Note #27 comment, this ADR,
  and the notebook to defend it against well-meaning cleanup.
- Dropout-before-norm means dropout's zeroing is partly re-centered by a
  downstream LayerNorm (when norm is on), which is not how the
  convention behaves — a subtlety a reviewer must keep in mind rather
  than assume standard semantics.
- The distinction is invisible in the production GRU config (only dropout
  is live there), so a regression that reorders the block could pass the
  Optimal-config smoke run and only surface on norm+activation configs;
  coverage must include such a config.

## Alternatives considered

1. **Adopt the conventional Transform → Norm → Activation → Dropout
   order.** Rejected: it is a different function of the same weights, so
   MATLAB checkpoints would not reproduce and training dynamics would
   shift silently — the exact failure Critical Note #27 warned against.

2. **Make the order configurable (a flag choosing MATLAB vs.
   conventional).** Rejected as unneeded surface area: nothing in the
   pipeline wants the conventional order, and a toggle invites the wrong
   setting to slip into a config and break parity quietly.

3. **Only preserve the order for configs where it matters, and simplify
   the production GRU path.** Rejected: the production path already
   collapses to Transform → Dropout via `nn.Identity` slots, so there is
   nothing to gain, and a special-cased build would reintroduce the very
   ordering ambiguity this ADR removes.

## References

- Implementing code: `models/encoder.py` — `_EncoderBlock.__init__`
  (slot construction) and `_EncoderBlock.forward` (the
  transform → dropout → norm → activation sequence).
- MATLAB source of the order: `cgg_generateSimpleBlock.m:116–121`.
- Migration Critical Note #27 (`docs/PLAN.md`) — pins the block order and
  requests this ADR.
- Walkthrough and non-commutativity demonstration:
  `notebooks/04_architecture/04.2_building_a_simple_encoder.ipynb`
  (§2.2 shows norm→activation vs. activation→norm produce different
  output distributions from identical inputs).
- Related decision on reproducing a MATLAB quirk deliberately:
  the `'SoftSign'`→`nn.Softplus` naming-bug parity in the same module
  (Critical Note #37).
- Overarching parity philosophy: [ADR 001](001_tiered_parity_not_bit_exact.md).
