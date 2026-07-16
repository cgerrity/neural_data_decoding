# `neural_data_decoding.sweeps`

Everything needed to turn a **single integer** into a fully-resolved training
run on a SLURM cluster. This subpackage is the Python port of the MATLAB sweep
harness (`SLURMPARAMETERS_cgg_runAutoEncoder_v2.m` +
`cgg_assignSLURMSession.m`): a static table of hyperparameter override bundles,
a `.slurm` script generator, the CLI-side index decomposition, the start-of-run
banner, a user-identity heuristic for mail defaults, and a parameter-coverage
audit doc.

It is pure glue — no torch/Hydra needed to import the dispatcher — so the table
and the SessionRunIDX math stay trivially testable.

## What lives here

| Module | Role |
| --- | --- |
| `dispatcher.py` | The **147-entry sweep table**. Flattens MATLAB's two-level `(SLURMChoice 1-15, SLURMIDX 1-10)` grid into one 1-based `sweep_index`; each entry carries the cfg override bundle (MATLAB CamelCase → Python snake_case) plus caveat `notes`. |
| `slurm_template.py` | Renders/writes a self-contained `.slurm` array script that invokes the **Python** pipeline (not MATLAB). |
| `cli_helpers.py` | `SessionRunIDX` → `(session, fold)` decomposition and the `cfg`-mutating override appliers. Consumed directly by `cli.py`; **not** re-exported from the package root. |
| `banner.py` | Collects + renders the diagnostic run banner (config headline, git SHA, GPU table, dataset shape). |
| `user_identity.py` | Read-only heuristic that fills `--mail-user` only when the caller is identified as the project owner. |
| `parameter_coverage.py` | **Documentation only** — a doc-string support matrix for all 47 MATLAB sweep variables (`__all__ = []`, no runtime code). The actual non-crash audit is the parametrized tests in `tests/integration/test_slurm_sweep_coverage.py`. |

## Key entry points

- **`lookup(sweep_index) -> SweepEntry`** (and the `SWEEP_ENTRIES` tuple /
  `total_sweep_count()`) — resolve one flat index to its override bundle. Grid:
  SC 1-14 have 10 entries each; SC 15 has 7 (its trailing base-filler rows are
  not exposed), for **147** total. `lookup_by_choice(choice, idx)` does the
  reverse from a MATLAB pair.
- **`render_slurm_template(options)` / `write_slurm_template(...)`** — emit an
  sbatch array script (`--array=1-N%1`, N = `num_sessions × num_folds`, default
  250) whose array-task ID *is* the `SessionRunIDX`.
- **`decompose_session_run_idx(session_run_idx, num_sessions)`** (in
  `cli_helpers`) — the fold-across-sessions decomposition
  (`session = (i-1) mod S + 1`, `fold = floor((i-1)/S) + 1`): fold 1 runs across
  *every* session before fold 2 begins, matching `cgg_assignSLURMSession.m`.

Secondary: `collect_banner_data(...)` + `render_banner(...)` for the run banner;
`identify_user()` + `maybe_default_mail()` for the mail-user default.

## Notes on fidelity (read before trusting a title)

- **Resolved literals, not callbacks.** MATLAB entries that referenced
  `cfg.HiddenSizes(end)` are stored with the resolved base-config value (250).
  A non-default base config must re-override via `--override`.
- **The mail heuristic is hardcoded and owner-only.** `user_identity.py`
  recognizes fixed usernames and one email; every other caller gets `None` and
  must pass `--mail-user` explicitly. It never writes the email to disk or git.
- **`parameter_coverage.py` implements nothing.** It is a support-status matrix
  in prose; ~40/47 variables full, 7 partial (data-prep fields the real-data
  loader consumes), 1 N/A (parallelism). Don't call into it.

## Related ADRs

- [ADR 012 — Pre-flight check, no overwrite](../../../docs/narrative/adrs/012_pre_flight_check_no_overwrite.md)
  — the mechanism that makes a re-submitted sweep **idempotent**. Honest scope:
  the predicate (`has_existing_checkpoint`) and the `check-existing` / `--force`
  CLI touch-points live in `training/checkpoint.py` + `cli.py`, **not** in this
  subpackage — sweeps produces the per-point config those gates then check. The
  ADR title is literally true: there is no overwrite path at all, only *abort*
  or *resume*, and `--force` means "proceed → resume from `current_state.pt`,"
  not "wipe and restart."
- [ADR 008 — Hydra config composition](../../../docs/narrative/adrs/008_hydra_config_composition.md)
  — how the override bundles from a `SweepEntry` merge onto a resolved cfg.
- [ADR 004 — Single-session batching](../../../docs/narrative/adrs/004_single_session_batching.md)
  — why runs are decomposed per session × fold.

## See also

- Docs: [user guide — parameter sweeps](../../../docs/narrative/user_guide/parameter_sweeps.md),
  [deployment — SLURM submission](../../../docs/narrative/deployment/slurm_submission.md),
  [cookbook — compare two sweep configs](../../../docs/narrative/cookbook/compare_two_sweep_configs.md),
  [API reference](../../../docs/api/sweeps.rst).
- Notebooks: [`09_production_deployment/`](../../../notebooks/09_production_deployment/)
  — `09.2_slurm_dispatch.ipynb` and `09.4_parameter_sweeps.ipynb`.
