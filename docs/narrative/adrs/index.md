# Architecture Decision Records

ADRs are short markdown documents recording **why** significant design decisions
were made. They are immutable — once accepted, an ADR is never deleted; if a
later decision supersedes it, the new ADR links back to the old one.

Each ADR follows the same template:

1. **Context** — the situation that prompted the decision.
2. **Decision** — what was chosen.
3. **Consequences** — what this commits us to (positive and negative).
4. **Alternatives considered** — options that were weighed and rejected.
5. **References** — links to relevant code, plan sections, external sources.

## Index

| # | Title | Status |
|---|-------|--------|
| [001](001_tiered_parity_not_bit_exact.md) | Tiered parity, not bit-exact | Accepted |
| 002 | Pythonic structure over MATLAB mirror | _Planned_ |
| 003 | AdamW for L2 weight decay | _Planned_ |
| 004 | Single-session batching | _Planned_ |
| 005 | No optimizer state in checkpoints | _Planned_ |
| 006 | Resume reads Current, not Optimal | _Planned_ |
| 007 | MAT interop surface | _Planned_ |
| 008 | Hydra config composition | _Planned_ |
| 009 | EMA prior cadence via RescaleLossEpoch | _Planned_ |
| 010 | Augmentation per `__getitem__` | _Planned_ |
| 011 | Validation per epoch by default | _Planned_ |
| 012 | Pre-flight check, no overwrite | _Planned_ |
| 013 | Memory probe via `cuda.mem_get_info` | _Planned_ |
| 014 | Single-GPU default, accelerate for multi | _Planned_ |
| 015 | Two doc toolchains (MkDocs + Sphinx) | _Planned_ |
| 016 | Minimal MATLAB cross-referencing in API docs | _Planned_ |
| 017 | NaN-masked reconstruction loss | _Planned_ |
| 018 | Layer block order: dropout before norm | _Planned_ |
| 019 | Single total loss, three subnetworks | _Planned_ |
| 020 | Confidence loss: five subtleties | _Planned_ |
| 021 | EMA prior normalized to classification | _Planned_ |
| 022 | He initialization explicit | _Planned_ |
| 023 | Augmentation loss auto-activated by topology | _Planned_ |
| 024 | Sampling layer deterministic at inference | _Planned_ |
