# Milestone D — Implementation plan

**Status:** in progress (started 2026-06-02).

**Goal:** Python pipeline runs on ACCRE with the same SLURM sweep model the MATLAB pipeline uses. Single-integer sweep index per configuration; per-session iteration is the primary sweep dimension via a `SessionRunIDX` flat index that decomposes to `(session, fold)`.

**Out of scope (explicit decisions with user):**
- `submitit` / Ray Tune Python-side schedulers — user pattern is bash → sbatch → Python, no Python-side scheduler needed
- MATLAB-side result aggregator (`DATA_cggAllNetworkEncoderResults.m`) — user runs it themselves; we already emit `CM_Table.mat` in the right format
- Base-folder auto-detection for mounted ACCRE paths — deferred until after D ships
- The other ~36 MATLAB ModelName variants not in the SLURM sweep — registry-only additions when needed (CC.1 already covers SLURM-required names)

## Workplan

1. ✅ Fix `mat_files.py` HDF5 detection (was checking offset 0; MATLAB v7.3 files have ASCII header there and version field at offset 124)
2. ✅ Multi-pass MATLAB read of `cgg_loadDataArray.m`, `cgg_loadTargetArray.m`, `cgg_runAutoEncoder.m`, `cgg_assignSLURMSession.m`, `cgg_procAutoEncoder.m`, `cgg_getKFoldPartitions.m`, `SLURMPARAMETERS_cgg_runAutoEncoder_v2.m`, `cgg_getClassifierOutputsFromProbabilities.m`
3. ✅ **D.1 — `MatFileTrialDataset`** in `src/neural_data_decoding/data/mat_dataset.py` (24 tests, real-fixture parity verified by direct indexing)
4. ✅ **D.2 — Sweep dispatcher** in `src/neural_data_decoding/sweeps/dispatcher.py` — 147 entries from the MATLAB SLURMPARAMETERS file, flat sweep_index 1..147 (21 tests)
5. ✅ **D.3 — CLI extensions** to `train` subcommand: `--sweep-index`, `--session-run-idx`, `--session`, `--override` (11 tests in `test_sweep_cli_helpers.py`)
6. **D.4 — Start-of-run banner** matching MATLAB's pattern (cfg dump, datetime, GPU table, session/fold identifier, git SHA, user identifier) — pending
7. ✅ **D.5 — `_identify_user()` helper** in `sweeps/user_identity.py` — `$USER` ∈ {cgerrity, gerritcg} OR git email match → auto-default SLURM `--mail-user` (4 tests)
8. ✅ **D.6 — `.slurm` template generator** via the `sweep-emit-slurm` subcommand — embeds (sweep_index, SC, IDX, SessionRunIDX) in the output filename for MATLAB log cross-reference; `set -euo pipefail` for fail-fast; auto-gated mail-user (17 tests)
9. **D.7 — `configs/target_milestone/real_data_base.yaml`** (real-data analog of `C_optimal_synthetic.yaml`)
10. **D.8 — Smoke run** end-to-end on `results/Decision/Decision_Data_0000011.mat` + `Target_0000011.mat`
11. Commit + push

## Critical facts about the MATLAB pipeline

### Data file shape (verified via MATLAB MCP on `Decision_Data_0000011.mat`)

Per MATLAB `[NumChannels, NumSamples, NumProbes] = size(Data)` (line 14 of `cgg_loadDataArray.m`):

- `(58, 3001, 6)` = `(C=58, NumSamples=3001, A=6)`
- The total-time axis is the **middle** one, **not trailing**. The user's verbal `(C, A, TT)` description had axes swapped; actual order is `(C, TT, A)`.
- Contains `NaN` at removed-channel positions (preserved through the loader, zeroed only at the encoder input per Critical Note #38)
- One `.mat` file = one trial

After windowing per `cgg_loadDataArray.m`: `(C, T=DataWidth, A, W=NumWindows)`. Transposed to `(W, T, A, C)` for the Python convention established by the data-restructure work.

### Target file structure (from `cgg_loadTargetArray.m`)

- Top-level `Target` struct with ~46 fields
- `cfg.Target='Dimension'` (the production default) → `SelectedObjectDimVals(FeatureDimensions=[1,2,3,5])` → 4-element int vector per trial
- Target dispatch table is in `PARAMETERS_cggVariableToData.m`; 25+ types defined. Primary need: `Dimension`. Other supported types in scope: `CorrectTrial` / `Trial Outcome` / `Outcome` (binary), `Dimensionality`, `Gain`, `Loss`, `DataNumber`.

### Per-session iteration (`cgg_assignSLURMSession.m`)

```
SessionIDX = mod(SessionRunIDX - 1, NumSessions) + 1
Fold       = floor((SessionRunIDX - 1) / NumSessions) + 1
cfg.Subset = cfg_Session(SessionIDX).SessionName  (hyphens → underscores)
```

- 25 sessions configured in `DATA_cggAllSessionInformationConfiguration.m` (13 `Wo_Probe_01_*`, 8 `Fr_Probe_02_*`, 4 `Fr_Probe_03_*`)
- 10 folds → `SessionRunIDX = 1..250` covers the full grid per sweep entry
- **MATLAB ordering** is `session-inside-fold` — i.e., `SessionRunIDX=1..25` runs fold 1 across all 25 sessions, then `26..50` runs fold 2 across all sessions, etc.
- **User explicitly requested keeping this order**: lets them see early-fold accuracy across all sessions before deciding to commit further compute. Do not flip.

### `cfg.Subset` semantics (`cgg_runAutoEncoder.m` lines 142-152)

- `true` (default): single-session mode using `cfg.SessionSubset`
- `false` or `'All'`: use all sessions
- `'<SessionName>'`: single-session mode for that specific session
- The SLURM `SessionRunIDX` dispatcher overrides `cfg.Subset` with the specific session name

### Active production config

`PARAMETERS_OPTIMAL_cgg_runAutoEncoder_v3.m` (when `ParameterSetName='Optimal'`). Synthetic test config: `PARAMETERS_OPTIMAL_cgg_runAutoEncoder_SyntheticEasy.m`. The `PaperBase` variant is for paper reproduction only.

### Start-of-run print pattern (`cgg_runAutoEncoder.m` lines 320-323)

```matlab
disp(cfg_Encoder);              % full cfg dump
disp(datetime);                 % timestamp
gpuDeviceTable(["Index","Name","TotalMemory",...])  % GPU info
cgg_getParallelPool;            % parallel pool setup banner
```

Plus `cgg_assignSLURMSession.m` line 23: `>>> Current SLURM Aim is Base Case - Fold N - Session SSS`.

The Python `_print_run_banner` should produce equivalent output plus git SHA and user identifier.

### The 1D-target bug (`cgg_getClassifierOutputsFromProbabilities.m` lines 153, 159)

MATLAB does naked `squeeze()` on the `(NumTrials, NumDims)` true-value table. When `NumDims=1` (e.g. `CorrectTrial` target), the dim axis collapses and the table shape becomes inconsistent with downstream consumers.

Python is **already safe** here: `np.array(list_of_lists)` preserves the singleton dim. The fix is to ensure `MatFileTrialDataset` doesn't pre-squeeze 1D targets — always return shape `(num_dims,)` even when `num_dims=1`.

### Sweep dispatcher source (`SLURMPARAMETERS_cgg_runAutoEncoder_v2.m`)

15 `SLURMChoice` blocks with ~10 `SLURMIDX` entries each (`~147 total non-commented entries`). Each entry is a `Description` + a small `SLURM_struct.<field>` override dict.

For the Python port, flatten the (SLURMChoice, SLURMIDX) pair to a single integer index 1..N. Each `SweepEntry` carries:
- `name` — MATLAB `Description` string
- `overrides` — dict of cfg field → value
- `base_config` — which YAML to start from (defaults to `real_data_base`)
- `notes` — caveats for partial-support fields (e.g. `BottleNeckDepth>1` not yet supported)

## Sample files for testing

In `results/Decision/`:

- `Decision_Data_0000011.mat` — sample trial data, shape `(58, 3001, 6)`
- `Target_0000011.mat` — paired target struct with all 46 fields
- `Autoencoder_SLURMChoice_10_SLURMIDX_1.slurm`, `_10.slurm` — example SLURM scripts to mirror in structure

## CLI design

Extending the existing `train` subcommand:

```
python -m neural_data_decoding train \
    --config-name BASE \
    [--fold K] \                       # existing
    [--sweep-index N] \                # NEW: applies sweep entry override bundle
    [--session-run-idx K] \            # NEW: flat (session, fold) decomposition (MATLAB ordering)
    [--session NAME] \                 # NEW: explicit session filter (alternative to --session-run-idx)
    [--override KEY=VALUE]...          # NEW: ad-hoc escape hatch
```

`SessionRunIDX` decomposition in Python:

```python
session_idx_zero = (session_run_idx - 1) % num_sessions      # 0..NumSessions-1
fold              = (session_run_idx - 1) // num_sessions + 1
session_name      = SESSIONS[session_idx_zero]
```

The .slurm template uses `--array=1-(NumSessions*NumFolds)%1` so each array task gets a unique `SessionRunIDX`.

## User-identification rules

- Detect Charles via `$USER ∈ {cgerrity, gerritcg}` OR `git config user.email == charles.g.gerrity@vanderbilt.edu`
- When detected: default SLURM `--mail-user` to that email
- When not detected: leave SLURM `--mail-user` blank; require explicit `--mail-user` flag
- **Never auto-leak email in git-side actions** — commit author stays `Claude Opus` co-author

## Generated SLURM template (matches `Autoencoder_SLURMChoice_10_SLURMIDX_1.slurm` structure)

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10
#SBATCH --time=48:00:00 --mem=64G
#SBATCH --array=1-250%1   # NumSessions=25 * NumFolds=10, sequential
#SBATCH --output=Output_Files/python_sweep-N-SessionRunIDX-%a.txt
[#SBATCH --mail-user=charles.g.gerrity@vanderbilt.edu --mail-type=ALL]   # only if _is_charles()

cd <repo>
source .venv/bin/activate

python -m neural_data_decoding train \
    --config-name real_data_base \
    --session-run-idx $SLURM_ARRAY_TASK_ID \
    --sweep-index N
```

Output dir `Output_Files/` matches the MATLAB convention (user confirmed; says they can change later if needed since they run the .slurm scripts).

## `real_data_base.yaml` config

Mirrors `C_optimal_synthetic.yaml` but with:
- `data_dir: ???`, `target_dir: ???` (Hydra missing-marker; set via `--override` at the SLURM line)
- `target: Dimension`, `target_dimensions: [1, 2, 3, 4]`, `feature_dimensions: [1, 2, 3, 5]`
- `data_width: 100`, `window_stride: 50`
- `subset: true` (single-session default)
- `num_classes_per_dim` either auto-detected from the dataset's actual `SelectedObjectDimVals` ranges or set explicitly

## Anti-patterns (don't do)

- ❌ Don't add submitit / Ray Tune
- ❌ Don't replicate MATLAB's naked `squeeze()` in CM_Table generation
- ❌ Don't embed user's email in git commits or any auto-applied default beyond SLURM mail-user when user is detected
- ❌ Don't try to read base-folder-mounted ACCRE data paths from this machine
- ❌ Don't write the MATLAB-side aggregator (user runs that)
- ❌ Don't auto-add `--mail-user` to SLURM templates when running as someone else (e.g., from CI)
