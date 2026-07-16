# ADR 008 — Composable YAML config (OmegaConf)

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline configures a run through a *cascade* of parameter
functions: `PARAMETERS_cgg_runAutoEncoder.m` returns a struct of base
defaults, and experiment-specific functions
(`PARAMETERS_OPTIMAL_cgg_runAutoEncoder_v3.m` and friends) take that struct
and overwrite a handful of fields. It is a layered-override pattern expressed
as a chain of function calls — base defaults first, experiment overrides
second.

Porting that to Python, the base config has ~70 fields. If every experiment
were a standalone file, changing one shared default (say the gradient-clip
threshold) would mean editing every file, and they would drift out of sync.
We wanted the same "base plus overrides" idea, but declarative: a single
`base.yaml` holding the shared defaults once, and a small per-experiment
overlay that lists *only what differs*.

The migration plan (`docs/PLAN.md`) called for this to be built with
**Hydra** — "the flat-table parameter struct (use Hydra config composition
properly)", "load from YAML via Hydra/OmegaConf", "configs as
Hydra-composable YAML hierarchies". Accordingly `hydra-core` was added as a
dependency and this ADR (and the accompanying notebook) inherited the "hydra"
name. The implementation, however, diverged from that plan — see the
**Decision** below, recorded truthfully.

## Decision

Compose the effective config from **two YAML files merged with plain
`OmegaConf.merge` — there is no Hydra in the composition path.**

`_load_config(name)` (`src/neural_data_decoding/cli.py`) does the entire
composition in one function:

1. Load `configs/base.yaml` (the shared defaults).
2. Load `configs/target_milestone/<name>.yaml` (the experiment overlay,
   selected by `--config-name`).
3. `merged = OmegaConf.merge(base, target)` — a deep merge in which **target
   keys win** and base-only keys carry through untouched.

Both files are required; a missing base or target raises `FileNotFoundError`
with the offending path, so a typo'd `--config-name` fails loudly. An
`assert isinstance(merged, DictConfig)` narrows the return type (OmegaConf
can return a list config).

CLI overrides are layered on **after** the merge by `_apply_cfg_flags`, which
applies `--sweep-index`, `--session`, repeatable `--override KEY=VALUE`, and
`--fold` via `OmegaConf.update`. Precedence, later wins:
`base.yaml` → target overlay → sweep index → session → `--override` → fold.

**What is honestly NOT Hydra, despite the name:**

- `hydra-core>=1.3` is a *declared* dependency in `pyproject.toml`, but there
  is **no `import hydra` anywhere in `src/`**. The only "Hydra" occurrences in
  the source are comments and docstrings.
- There is no `@hydra.main`, no `hydra.compose`, no config-group resolution,
  and no multirun. Composition is the single `OmegaConf.merge` call above.
- The `defaults:` blocks present in the YAML files are **inert** — they are
  never resolved as a Hydra composition list, and are explicitly dropped
  (`if k != "defaults"`) before the resolved config is written back out as
  `EncodingParameters.yaml`.
- `configs/architecture/` and `configs/sweep/` exist but are **empty
  directories** — vestiges of the intended Hydra config groups. Architecture
  is chosen by code registries, not a config group; the sweep is a
  hand-authored dispatcher table, not a YAML group.
- The filename `008_hydra_config_composition.md` (and the notebook
  `09.3_hydra_config_composition`) keep "hydra" for historical/plan-alignment
  reasons only. The label describes the *aspiration*; the code does OmegaConf.

## Consequences

**Positive**

- One `base.yaml` carries the shared defaults once; each experiment overlay
  states only its handful of differences, so defaults cannot drift out of
  sync (change base, everything inherits). This is the declarative twin of the
  MATLAB `PARAMETERS_*` cascade.
- The composition surface is tiny and transparent: two `OmegaConf.load`s and
  one `merge`, readable end-to-end in one function, with no framework magic to
  reason about.
- The CLI-override chain (`_apply_cfg_flags`) gives the SLURM array jobs
  exactly what they need — every task differs only by a couple of flags on top
  of a shared base+target.
- No runtime dependence on Hydra's global state, working-directory rewrites,
  or `@main` decorator, all of which complicate library-style invocation and
  testing.

**Negative**

- A dependency (`hydra-core`) is declared but unused, which is misleading
  until you read the code — precisely the honesty hazard this ADR exists to
  neutralize.
- The name ("hydra") on this ADR, the notebook, and the empty
  `architecture/`/`sweep/` dirs invites the false assumption that Hydra
  features (config groups, `defaults:` composition, multirun, override syntax)
  are available. They are not.
- We forgo Hydra's genuinely useful features (structured config validation,
  built-in multirun, tab-completed overrides). If those are wanted later,
  adopting Hydra is a real migration, not a rename.

## Alternatives considered

1. **Adopt Hydra properly (as the plan specified).** Rejected *for now*: the
   pipeline's needs are met by a two-file merge plus a few CLI flags. Hydra's
   config groups, `@main` CWD management, and multirun add framework surface
   and global state without buying anything the current sweep/dispatch design
   needs (architecture selection is code-registry-driven; the sweep is a
   table). Left as a clean future migration if structured validation or
   built-in multirun become worth it — hence the dependency and naming are
   kept rather than removed.

2. **One flat standalone YAML per experiment.** Rejected: ~70 shared fields
   would be duplicated across every file and drift out of sync; changing a
   shared default would mean editing every file. This is the exact anti-pattern
   the layered base+overlay design removes.

3. **Python `dataclass`/pydantic config objects loaded from YAML.** Rejected
   for the composition layer: OmegaConf already gives deep-merge,
   dotted-path `update`, interpolation, and `.mat`-facing serialization for
   free, and the resolved config must round-trip to a stable-schema
   `EncodingParameters.yaml` (Critical Note #25) that MATLAB analysis reads.
   A typed layer on top remains possible but was not needed to ship.

## References

- Composition: `src/neural_data_decoding/cli.py::_load_config`
  (`OmegaConf.merge(base, target)`; `CONFIG_ROOT = <repo>/configs`).
- CLI override layering: `src/neural_data_decoding/cli.py::_apply_cfg_flags`
  (`OmegaConf.update` for sweep index / session / `--override` / fold); the
  `defaults`-key drop lives near the `EncodingParameters.yaml` write in the
  same module.
- Config files: `configs/base.yaml` and `configs/target_milestone/*.yaml`
  (`A_logistic_synthetic`, `B_gru_classifier_synthetic`,
  `C_optimal_synthetic`, `C_two_stage_synthetic`, `real_data_base`); empty
  vestigial groups `configs/architecture/` and `configs/sweep/`.
- Declared-but-unused dependency: `pyproject.toml` (`hydra-core>=1.3`); no
  `import hydra` exists under `src/`.
- Stable resolved-config output: Critical Note #25 (`docs/PLAN.md`) — every
  run's `EncodingParameters.yaml` must be field-schema-symmetric across a
  sweep so `cgg_plotParameterSweep.m` can scan it.
- Walkthrough notebook:
  `notebooks/09_production_deployment/09.3_hydra_config_composition.ipynb`
  (Section 2.3, "The honest state: OmegaConf, not Hydra").
- External: [OmegaConf docs](https://omegaconf.readthedocs.io/) (what this
  actually uses); [Hydra docs](https://hydra.cc) (what the name refers to,
  for if/when the project adopts it).
