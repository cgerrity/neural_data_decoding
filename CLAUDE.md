# CLAUDE.md — neural_data_decoding

Project context for Claude Code sessions. Read `HANDOFF.md` for the full
state snapshot. This file is the concise bootstrap.

## What this is

Python/PyTorch port of the MATLAB neural-decoding pipeline at
`Processing_Functions_cgg/` (variational autoencoder + multi-head
classifier for multi-probe ephys data). Self-contained Python project;
the MATLAB sources are referenced for parity testing only.

## Working with this repo

- **CWD when a Claude Code session starts in this directory IS the
  project root.** No `cd` needed for `source .venv/bin/activate`,
  `pytest`, `python -m neural_data_decoding`, etc.
- **The `.venv/` lives at the project root** — `source .venv/bin/activate`
  works directly.
- **MATLAB sources** are NOT in this project. Set
  `NDD_MATLAB_SOURCE_ROOT` in your shell rc to the directory containing
  `Processing_Functions_cgg/`. The fallbacks try the parent of the project
  root (legacy layout) and a known absolute path; if neither works the
  fixture scripts error clearly.
- **Tests:** `python -m pytest` runs the default 409 tests in ~3s. MATLAB-gated
  parity (`-m needs_matlab`) needs a MATLAB executable + the source root
  resolvable.
- **Static checking:** `pyright` runs project-wide with zero errors. The
  config (`pyproject.toml [tool.pyright]`) silences `reportPrivateImportUsage`
  for torch (lazy-load false positives).
- **Output locations:** smoke-test runs write to `<repo>/results/`
  (gitignored). Don't default to `/tmp/...` without telling the user.

## Status & next step

- Milestones 0 / A / B complete; Milestone C **core + curriculum
  schedules** complete (VAE sampling, NaN mask, ELBO, MIL softmax,
  variational composite, confidence PD-controller, EMA prior
  normalization, variational training integration, dynamic curriculum
  schedules with per-module freeze + live-read augmentation +
  RescaleLossEpoch cadence) and end-to-end smoke-runnable.
- Single-step T2 forward parity against MATLAB: ~1e-9 (composite
  forward), ~1e-10 (confidence kernel), 1e-6 (ELBO + MIL + sampling),
  ~1e-12 (curriculum interpolator + preset library).
- All three target_milestone configs (A / B / C) run end-to-end on
  synthetic data and write both `CM_Table_Validation.mat` (during
  training) and `CM_Table.mat` (from the Optimal weights on the
  held-out test set). C_optimal_synthetic now uses the
  `Soft Three-Stage Curriculum - Shortened` regime and shows visible
  per-epoch ticking of augmentation, weights, and freeze factors.
- **Next step is Milestone C #6 — full two-stage lifecycle** with
  `NumEpochsAutoEncoder > 0` (Stage 1 unsupervised pre-training handing
  off Optimal encoder weights to Stage 2 supervised), plus KL annealing
  wiring (the `KLBaseAnneal` helper exists but isn't yet config-driven),
  and the hardware-aware accumulation table (Note #18). See `HANDOFF.md`
  "Next up".

## Conventions to follow

- NumPy-style docstrings; class docstring carries `Parameters`, not `__init__`.
  Interrogate gate is 100%.
- Disclose any write outside the project (`~/.claude/`, `/tmp/`, etc.)
  before the tool call.
- Don't fabricate parity values — verify empirically against MATLAB
  fixtures. Critical Note #38 was wrong in the plan; the empirical probe
  is what caught it.
- Always check `HANDOFF.md` first for "what was I doing last."

## Reference

- **`HANDOFF.md`** — full state snapshot, current milestone progress,
  next-up checklist.
- **`docs/PLAN.md`** — frozen migration spec (all Critical Notes).
- **`README.md`** — public-facing quickstart + parity status.
