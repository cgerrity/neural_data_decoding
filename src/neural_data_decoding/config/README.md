# `neural_data_decoding.config`

**Status: stub / placeholder.** This subpackage is currently empty apart
from a module docstring. It exports **no classes and no functions today**.
Do not import anything from here expecting it to work — there is nothing to
import yet.

## What it is for

This package is reserved for **typed configuration dataclasses** — the
intended Python replacement for MATLAB's `CheckVararginPairs` / `varargin`
parameter pattern. The idea is a validated, statically-typed layer that would
sit *on top of* the raw config dict once one is needed. That layer has not
been built; it was explicitly deferred as "possible but not needed to ship"
(see ADR 008, Alternative #3).

## Where config actually lives today

Config is **not** assembled here. It is composed at runtime from YAML by the
CLI, using plain `OmegaConf.merge` (not Hydra — see the honesty note below).
The real entry points are in [`../cli.py`](../cli.py):

- **`_load_config(name)`** — loads `configs/base.yaml` and
  `configs/target_milestone/<name>.yaml` and deep-merges them
  (`OmegaConf.merge(base, target)`, target keys win). Returns a `DictConfig`.
- **`_apply_cfg_flags(cfg, args)`** — layers CLI overrides on after the merge:
  `--sweep-index`, `--session-run-idx`, `--session`, `--override KEY=VALUE`,
  `--fold` (precedence, later wins, in that order).

The composed config is a bare `DictConfig` of ~70 fields, threaded through the
pipeline as-is. The config files themselves live in
[`../../../configs/`](../../../configs/) (`base.yaml` plus the
`target_milestone/*.yaml` overlays).

## Honesty note (read before trusting names)

Despite the "hydra" naming on the ADR and notebook, **there is no Hydra in the
composition path** — no `import hydra` anywhere under `src/`. Composition is
one `OmegaConf.merge` call. `hydra-core` is a declared-but-unused dependency,
and the `defaults:` blocks plus `configs/architecture/` and `configs/sweep/`
directories are inert vestiges. The full truth is recorded in ADR 008.

## References

- **ADR 008 — Composable YAML config (OmegaConf):**
  [`../../../docs/narrative/adrs/008_hydra_config_composition.md`](../../../docs/narrative/adrs/008_hydra_config_composition.md)
  (the decision, and why it is OmegaConf and not Hydra).
- **Cookbook — Compare two sweep configs:**
  [`../../../docs/narrative/cookbook/compare_two_sweep_configs.md`](../../../docs/narrative/cookbook/compare_two_sweep_configs.md)
- **Notebook — Module 09 (Production Deployment), 09.3:**
  [`../../../notebooks/09_production_deployment/09.3_hydra_config_composition.ipynb`](../../../notebooks/09_production_deployment/09.3_hydra_config_composition.ipynb)
  (see Section 2.3, "The honest state: OmegaConf, not Hydra").
