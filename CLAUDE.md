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
- **Tests:** `python -m pytest` runs the default 296 tests in ~3s. MATLAB-gated
  parity (`-m needs_matlab`) needs a MATLAB executable + the source root
  resolvable.
- **Output locations:** smoke-test runs write to `<repo>/results/`
  (gitignored). Don't default to `/tmp/...` without telling the user.

## Status & next step

- Milestones 0 / A / B complete; Milestone C in progress.
- Single-step T2 forward parity against MATLAB verified to ~1e-9 for the
  full Encoder + Bottleneck + Deep LSTM Classifier composite.
- Milestone C VAE-core (sampling, NaN mask, ELBO, MIL softmax,
  variational composite) is in. **Next step is Milestone C #3 —
  confidence routing**, the highest-risk port (Critical Note #29's five
  subtleties). See `HANDOFF.md` "Next up".

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
