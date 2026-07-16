# `neural_data_decoding.utils`

Small, dependency-light helpers that the rest of the pipeline leans on but
that don't belong to any one stage: **runtime-environment / path resolution**,
**global RNG seeding**, and **MATLAB `dlarray` ↔ PyTorch axis conversion**.
Plus a helper for locating the external MATLAB source tree used by parity
tests. These modules are imported directly from their submodules; `__init__.py`
is intentionally just a package marker and re-exports nothing.

> **Accuracy note:** the `__init__.py` docstring reads
> "environment detection, seeding, structured logging, MATLAB axis
> converters," but there is **no logging module here** today. Structured
> logging is not implemented in this subpackage — treat that phrase as
> aspirational, not present.

## What's here

| Module | Purpose |
| --- | --- |
| `paths.py` | Detect the host environment (Local / TEBA / ACCRE) and return its input/output/scratch base directories. Ports MATLAB's `cgg_getBaseFolders.m` + `cgg_checkACCREMounted.m`. |
| `seeding.py` | Seed every RNG the pipeline touches (`random`, NumPy legacy, PyTorch CPU/CUDA) in one call. |
| `matlab_axes.py` | Reorder tensor axes between MATLAB `dlarray` format strings (`'SSCTB'`, `'CBT'`, …) and PyTorch layouts. Shape-only; no copy when a view suffices. |
| `matlab_source.py` | Locate the external `Processing_Functions_cgg/` MATLAB tree (for fixture generation / parity gating). |

## Key entry points

- **`get_base_paths(*, environment=None, want_teba=False) -> BasePaths`** —
  resolves `input` / `output` / `temporary` roots for the detected (or forced)
  host. Detection uses hostname + filesystem heuristics and is overridable via
  the `NDD_FORCE_ENV` env var (`local` / `teba` / `accre`).
- **`set_global_seed(seed, *, deterministic_cuda=False) -> int`** — seeds
  `random`, NumPy, and PyTorch (CPU + CUDA) for *intra-Python* determinism.
  Note: this does **not** give bit-exact parity with MATLAB — that is an
  explicit non-goal (ADR 001); `deterministic_cuda` is off by default.
- **`permute_to_pytorch(tensor, *, source_format, target_format)`** /
  **`permute_to_matlab(...)`** — permute a NumPy array or Torch tensor between
  two `dlarray` tag orders (same tag multiset, reordered). The two names are
  directional aliases of one implementation. `parse_matlab_format(fmt)` splits
  a format string into per-axis tags.
- **`find_matlab_source_root() -> Path`** (plus `matlab_source_available()`) —
  finds the MATLAB tree via `$NDD_MATLAB_SOURCE_ROOT`, then the legacy
  parent-dir layout, then a known workstation fallback; raises
  `MatlabSourceNotFoundError` naming every path tried.

## Related ADRs

No ADR directly governs environment/path detection — it is plumbing that
mirrors the MATLAB folder helpers. The two ADRs that touch this subpackage's
behavior:

- [`ADR 001 — Tiered parity, not bit-exact`](../../../docs/narrative/adrs/001_tiered_parity_not_bit_exact.md)
  — why `set_global_seed` targets intra-Python determinism only, not
  MATLAB-identical RNG streams.
- [`ADR 007 — MAT interop surface`](../../../docs/narrative/adrs/007_mat_interop_surface.md)
  — context for the `dlarray` ↔ PyTorch axis conversion used when reading/
  writing `.mat` artifacts.

(For contrast, an unrelated example is
[`ADR 003 — AdamW for L2 weight decay`](../../../docs/narrative/adrs/003_adamw_for_l2_weight_decay.md),
which concerns the optimizer, not these utilities.)

## Learn more

- **User guide:** [`running_on_accre.md`](../../../docs/narrative/user_guide/running_on_accre.md)
  — the multi-environment layout that `get_base_paths` resolves.
- **Notebook module:** [`02_numpy_and_pytorch_basics`](../../../notebooks/02_numpy_and_pytorch_basics/),
  especially [`02.2_array_axis_conventions.ipynb`](../../../notebooks/02_numpy_and_pytorch_basics/02.2_array_axis_conventions.ipynb)
  for the axis-order model behind `matlab_axes.py`.
