# ADR 012 — Pre-flight check, no overwrite

**Status**: Accepted
**Date**: 2026-07-16

## Context

A training run is expensive — a real sweep is thousands of jobs, each many
GPU-hours. The result directory for a run is derived deterministically from the
resolved config, so re-invoking `train` with the same config resolves to the
*same* directory. Without a guard, a re-submitted job (a fat-fingered re-run, a
SLURM array that got resubmitted, a sweep point that was already computed) would
silently write over a completed run's outputs.

The MATLAB pipeline anticipated this. Critical Note #22 specifies a **pre-flight
existing-network check**: before training, scan the resolved output directory for
existing snapshot files (`*-Current.mat` / `*-Optimal.mat` / `CurrentIteration.mat`),
and if any exist, **abort with an informative error** rather than overwrite. The
user deletes the directory manually if the overwrite was intended. The note's
stated purpose is to prevent silent loss of expensive training runs.

The port needs the same guarantee, plus a non-mutating variant that a sweep
launcher can call to decide whether a given sweep point is already done — so a
re-submitted sweep is idempotent (skip finished points) and restartable (continue
interrupted ones) instead of redoing everything.

## Decision

Implement the pre-flight check as a single predicate plus two CLI touch-points.

**The predicate.** `has_existing_checkpoint(checkpoint_dir)` returns `True` iff
`current_state.pt` **or** `optimal_state.pt` exists in the directory; it returns
`False` (not an error) when the directory does not exist. This is a truthful
divergence from Critical Note #22 worth stating plainly: the Python port persists
two PyTorch `.pt` snapshots, not the three MATLAB `.mat` files the note names, and
it has **no separate iteration file** — the iteration counter lives *inside*
`current_state.pt`. So the check keys off the two `.pt` filenames, nothing else.
It does not inspect `CM_Table*.mat` or `EncodingParameters.yaml`; a directory that
somehow holds only those would read as "empty."

**The `train` guard.** `_cmd_train` resolves the result directory, `mkdir`s it
(harmless — an empty dir is not a checkpoint), then runs
`if has_existing_checkpoint(result_dir) and not args.force:` — printing an error to
stderr and returning **exit code 2** *before writing any output* (the
`EncodingParameters.yaml` write and all training happen after the guard). No code
path deletes existing checkpoints; the guard only ever aborts. A second honest
deviation from the note: the error message prints the *directory path* and the
instruction "Delete them or re-run with `--force`," but does **not** enumerate the
individual offending files the way the note's "listing the files" implies.

**What `--force` actually does.** `--force` bypasses the abort — but it does **not**
wipe and restart. `fit_supervised` / `fit_two_stage` unconditionally call
`load_current_checkpoint` at entry, so proceeding into a directory that already has
`current_state.pt` **resumes** from that checkpoint (start epoch = saved epoch + 1)
rather than starting fresh. So `--force` means "proceed anyway," which in practice
means "resume." A genuine from-scratch re-run still requires manually deleting the
directory — exactly the manual-delete escape hatch the note prescribed. The title's
"no overwrite" is therefore literally true: there is no overwrite path at all, only
abort or resume.

**The `check-existing` subcommand.** `_cmd_check_existing` runs the same predicate
without mutating anything, prints `{"result_dir", "has_existing_checkpoint"}` as
JSON, and returns **exit code 1 when a checkpoint is found, 0 when the directory is
clear**. This inversion of the usual "0 = success" convention is intentional and
load-bearing: a sweep launcher tests the exit code to **skip** already-completed
points (nonzero → done) instead of redoing them. Combined with the resume logic
(ADR 006), a re-submitted sweep is idempotent for finished points and restartable
for interrupted ones.

## Consequences

**Positive**

- Re-submitting a job or a whole sweep array cannot silently destroy a completed
  run — the worst case is a fast, informative abort.
- The guard is genuinely pre-flight: it fires before any output byte is written, so
  an aborted attempt leaves the prior run's files untouched.
- `check-existing` makes big sweeps idempotent (skip done points) and, with resume,
  restartable — the exact property a multi-thousand-job cluster sweep needs when
  jobs get preempted and requeued.
- `--force` is a safe override: it resumes rather than clobbers, so even the
  "escape hatch" preserves prior work.

**Negative**

- Detection is scoped to the two `.pt` filenames. A directory holding only
  `CM_Table*.mat` / `EncodingParameters.yaml` but no `.pt` snapshot reads as empty
  and would not trip the guard — an unlikely state, but the check is narrower than
  "is this directory non-empty."
- The abort message names only the directory, not the specific files, so it is
  slightly less informative than Critical Note #22 asked for. The user still has to
  `ls` the directory to see what is there.
- The inverted exit-code convention on `check-existing` (1 = found) surprises a
  reader who expects 0 = "the thing exists"; it must be read as "0 = clear to run."
- `--force`'s resume-not-restart semantics can surprise someone who reads "force"
  as "overwrite from scratch." A true reset requires deleting the directory first.

## Alternatives considered

1. **Silently overwrite (no guard).** Rejected — the failure mode is catastrophic
   and invisible: a resubmitted job erases hours of results with no signal.

2. **Always resume, never abort.** Rejected as the *default*: resuming into a
   directory the user did not mean to reuse hides config mistakes (they think they
   started fresh but silently continued a stale run). The abort forces an explicit
   acknowledgment; `--force` then opts into the resume behavior deliberately.

3. **Prompt interactively on collision.** Rejected — training runs headless on
   SLURM with no TTY; an interactive prompt would hang the array job.

4. **A single flag with no `check-existing` subcommand.** Rejected — a sweep
   launcher needs to test "is this point done?" *without* starting a run and without
   the side effect of `mkdir`/config-write. A dedicated non-mutating, exit-code-only
   query is what makes the sweep idempotent.

5. **Match the MATLAB filenames literally** (`*-Current.mat` etc.). Rejected — the
   port persists PyTorch `.pt` state, not `.mat`, so the check keys off the actual
   filenames the pipeline writes (ADR 005, ADR 006). Mirroring dead MATLAB names
   would guard against files that never exist.

## References

- Predicate: `src/neural_data_decoding/training/checkpoint.py` — `has_existing_checkpoint`
  (checks `current_state.pt` / `optimal_state.pt`).
- CLI guard: `src/neural_data_decoding/cli.py` — `_cmd_train` (the
  `has_existing_checkpoint(...) and not args.force` abort, exit code 2), the
  `--force` flag registration, and `_cmd_check_existing` (JSON report, exit code 1
  when found else 0).
- Resume behavior that `--force` falls through to:
  `src/neural_data_decoding/training/lifecycle.py` — `fit_supervised` /
  `fit_two_stage` unconditionally call `load_current_checkpoint`.
- Migration spec: Critical Note #22 ("Pre-flight existing-network check") in
  `docs/PLAN.md`.
- Notebook walkthrough:
  `notebooks/09_production_deployment/09.5_debugging_a_failing_run.ipynb`
  (Section 2.5, "Before the run: the pre-flight clobber check").
- Related checkpoint-semantics decisions:
  [ADR 005 — no optimizer state in checkpoints](005_no_optimizer_state_in_checkpoints.md),
  [ADR 006 — resume reads Current, not Optimal](006_resume_reads_current_not_optimal.md).
