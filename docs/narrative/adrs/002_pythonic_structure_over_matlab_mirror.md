# ADR 002 — Pythonic structure over MATLAB mirror

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline in `Processing_Functions_cgg/` is laid out the way MATLAB
forces code to be laid out: one function per file, a flat directory of
`cgg_*`-prefixed `.m` files, architecture selection done by giant
string-`switch` dispatchers (`PARAMETERS_cgg_constructNetworkArchitecture.m`
lists 47 `ModelName` strings; `cgg_selectClassifier.m` lists 9 `ClassifierName`
strings), configuration threaded through every function as `varargin` pairs
parsed by `CheckVararginPairs`, and a script-vs-function dual-mode boilerplate
header at the top of nearly every file.

A literal port would mirror that layout — a Python module per MATLAB function,
same names, same flat structure. That instinct is wrong for this project. The
migration doctrine is **functional parity, not function-by-function structural
parity**: port the *semantics* (topology, loss math, data contract, output
`.mat` schema), not the *syntax* (file decomposition, naming, dispatch
mechanics). MATLAB's structure exists to satisfy MATLAB's constraints, none of
which apply to a `src/`-layout Python package. Mirroring it would import those
constraints for no benefit and produce un-idiomatic Python that a MATLAB-native
reader would find no more legible than the original.

## Decision

The Python package is organized **by concern**, not by MATLAB provenance.
`src/neural_data_decoding/` contains seven subpackages, each owning one slice of
the pipeline (see the map in `src/neural_data_decoding/__init__.py`):

- `config/` — typed configuration dataclasses (replaces the `CheckVararginPairs`
  varargin pattern; Critical Note #24).
- `data/` — datasets, samplers, stratification, normalization, augmentation,
  `.mat` loading.
- `models/` — encoder / decoder / classifier / composite builders, plus
  `layers/` and `stitching_fusion/` for custom modules.
- `training/` — the loop, lifecycle, checkpointing, freezing, accumulation, plus
  `losses/`, `schedules/`, and `monitoring/` subpackages.
- `interop/` — the bounded MATLAB compatibility surface.
- `sweeps/` — SLURM / sweep launchers and coverage tooling.
- `utils/` — environment detection, seeding, shape/axis converters.

Concretely, the semantics-not-syntax port shows up as:

- MATLAB's 47-way and 9-way string dispatchers become **decorator/dict
  registries** (`models/registry.py`, `models/architecture_registry.py`): a
  builder registers itself with `@register_classifier("...")` and is selected by
  the same config string MATLAB used — same behavior, idiomatic mechanism
  (Critical Note #14).
- One-file-per-function decomposition is collapsed into cohesive modules grouped
  by role; the `cgg_*` prefix is dropped entirely in favor of Python-idiomatic
  names.

Honest nuance — MATLAB coupling is **not eliminated, it is quarantined**. Where
Python output must match MATLAB conventions for the downstream analysis scripts
to consume it, that knowledge is concentrated in `interop/`, whose own docstring
states it is the "bounded MATLAB-compatibility surface" and that "the rest of the
codebase stays free to be idiomatic Python; only this subpackage knows about
MATLAB-side requirements." A handful of modules outside `interop/` still carry
`matlab`/`mat` in their names on purpose — `utils/matlab_axes.py`,
`utils/matlab_source.py`, `data/mat_files.py`, `data/mat_dataset.py` — precisely
to flag where a MATLAB-facing constraint (axis ordering, the v7.3 HDF5 `.mat`
reader, the source-root resolver) crosses into otherwise-Pythonic code. The
package is Pythonic by default with the MATLAB boundary made explicit, not a
MATLAB layout dressed in `.py` extensions.

## Consequences

**Positive**

- A reader navigates by *what the code does* (data, models, training) instead of
  by MATLAB filename trivia; related logic sits together instead of scattered
  across dozens of `cgg_*.m` files.
- The MATLAB boundary is auditable in one place. To find every MATLAB-parity
  requirement, read `interop/` and the handful of `matlab_*`/`mat_*` modules.
- Idiomatic mechanisms (registries, dataclasses, `nn.Module` composition) unlock
  the tooling MATLAB never had: pyright resolves every symbol, IDEs autocomplete,
  and imports document each module's dependencies at its top.

**Negative**

- There is no 1-to-1 file map between the two codebases, so tracing a MATLAB
  function to its Python home requires knowing the concern it belongs to rather
  than grepping a matching filename. Docstrings mitigate this by naming the
  originating `cgg_*.m` source per port.
- "Idiomatic where it improves the codebase" is a judgment call; two reviewers
  can disagree on where the interop boundary should sit and how much a module may
  know about MATLAB before it belongs in `interop/`.

## Alternatives considered

1. **Mirror the MATLAB file/function layout 1-to-1.** Rejected: it imports
   MATLAB's constraints (one function per file, flat directory, `cgg_*` naming)
   into a language that has none of them, producing un-idiomatic Python with no
   offsetting benefit — and structural parity was never a stated goal, functional
   parity was.

2. **Port the string-`switch` dispatchers verbatim as big `if/elif` chains.**
   Rejected: registries give the same string-selected behavior while being
   extensible (Milestone CC adds variants without editing a central switch) and
   introspectable (`list_encoders()` / `list_classifiers()`).

3. **Eliminate the MATLAB coupling entirely and only emit a "clean" native
   format.** Rejected: the downstream analysis scripts (`DATA_cggAllNetwork
   EncoderResults.m` and the sweep plotters) consume specific `.mat` schemas and
   folder hierarchies. The coupling is a real requirement; the right move is to
   *bound* it in `interop/`, not pretend it away.

## References

- Subpackage map and rationale: `src/neural_data_decoding/__init__.py`.
- Bounded MATLAB surface: `src/neural_data_decoding/interop/__init__.py`.
- String dispatch → Python registries: `src/neural_data_decoding/models/registry.py`
  (Critical Note #14).
- Varargin → dataclasses: `src/neural_data_decoding/config/__init__.py`
  (Critical Note #24).
- Migration doctrine (semantics-not-syntax): the "Parity Doctrine: Functional,
  Not Structural" section and the "Structural fidelity" row of the Parity Goals
  table in `docs/PLAN.md`. There is no single numbered Critical Note for the
  doctrine itself; #14 (registries) and #24 (dataclasses) are its concrete
  instances.
- Reader-facing walkthrough of the mental-model shift, including the explicit-
  namespace / no-MATLAB-path contrast and an annotated real production file:
  `notebooks/00_orientation/00.3_the_matlab_to_python_mental_model.ipynb`.
