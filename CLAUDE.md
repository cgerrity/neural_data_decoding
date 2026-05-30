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
- **Tests:** `python -m pytest` runs the default 445 tests in ~3s. MATLAB-gated
  parity (`-m needs_matlab`) needs a MATLAB executable + the source root
  resolvable.
- **Static checking:** `pyright` runs project-wide with zero errors. The
  config (`pyproject.toml [tool.pyright]`) silences `reportPrivateImportUsage`
  for torch (lazy-load false positives).
- **Output locations:** smoke-test runs write to `<repo>/results/`
  (gitignored). Don't default to `/tmp/...` without telling the user.

## Status & next step

- Milestones 0 / A / B complete; Milestone C **core + curriculum +
  two-stage + confidence routing** complete (VAE sampling, NaN mask,
  ELBO, MIL softmax kernel, variational composite, confidence
  PD-controller kernel, EMA prior normalization, variational training
  integration, dynamic curriculum schedules with per-module freeze +
  live-read augmentation + RescaleLossEpoch cadence, full two-stage
  Stage 1 → handoff → Stage 2 with config-driven KL annealing,
  TrialConfidenceHead + TaskConfidenceHead grafted into the variational
  composite with Beta P-controller threading per-batch) and end-to-end
  smoke-runnable.
- Single-step T2 forward parity against MATLAB: ~1e-9 (composite
  forward), ~1e-10 (confidence kernel), 1e-6 (ELBO + MIL + sampling),
  ~1e-12 (curriculum interpolator + preset library).
- All four target_milestone configs (A / B / C_optimal /
  C_two_stage) run end-to-end on synthetic data and write both
  `CM_Table_Validation.mat` (during training) and `CM_Table.mat` (from
  the Optimal weights on the held-out test set). C_two_stage_synthetic
  runs 5 epochs unsupervised → loads Optimal autoencoder weights →
  builds composite + classifier → runs 15 epochs supervised, with KL
  annealing visible in Stage 2 epochs 2-5 and the curriculum taking
  over at epochs 10+.
- **Next step is Milestone C polish / cleanup or jump to CC/D.** Four
  options (see `HANDOFF.md` "Next up" for full descriptions):
  - **C #7b**: Eq. 2 interpolated cross-entropy (confidence-weighted CE).
  - **C #8**: MIL softmax pooling in the variational forward path.
  - **C accumulation**: hardware-aware gradient accumulation table.
  - **CC** (extra-credit features) or **D** (cluster deployment).

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
