# neural_data_decoding — Python Migration Plan for the MATLAB Neural Decoding Pipeline

The new pipeline lives at `neural_data_decoding/` in the repo root, mirroring the parent project's "Neural Data Reading" naming scheme. (The pre-processing / processing code currently in `Processing_Functions_cgg/` will eventually be split into its own sibling repo named in the same family; this plan only concerns the decoder portion.)

*(Auto-generated filename in `.claude/plans/` is non-descriptive; this title is the canonical name. The repo-side copy will be saved as `Plans/neural_data_decoding_plan.md`.)*

## Context

This plan specifies the complete migration of the MATLAB neural decoding pipeline (`Processing_Functions_cgg/`) to a new Python pipeline at `neural_data_decoding/`. **This document is self-contained** — no other plan files or external specifications are required to execute it. The implementer should treat this as the single source of truth for the migration.

The MATLAB pipeline trains variational autoencoder–based decoders on multi-probe ephys data, with multi-task losses (ELBO + multi-head classification + trial/task confidence + learnable augmentation), curriculum-based dynamic parameter scheduling, hierarchically stratified K-fold cross-validation, single-session minibatching (every minibatch comes from one session — see Critical Notes), and a two-stage training lifecycle (unsupervised pre-training → supervised fine-tuning). The new Python pipeline reproduces this functionality using modern Pythonic patterns while writing `.mat`-compatible output where MATLAB-side analysis still consumes it.

**The new pipeline is at `neural_data_decoding/`** as a fresh, self-contained codebase. Nothing else in the repo is modified.

**Authoritative references for understanding the source pipeline:**
- `Codebase_Documentation.md` — full Phase 1–5 reference for every active MATLAB function in the pipeline
- `Execution_Path_Map.md` — active vs. legacy vs. dead-code classification; supported-but-unused options
- `Processing_Functions_cgg/Parameters/PARAMETERS_OPTIMAL_cgg_runAutoEncoder_v3.m` — the target production configuration (referred to as **"Optimal"** throughout this plan; the `_v3` suffix on the MATLAB filename is a historical artifact of three parameter-structuring iterations and is not semantically meaningful — the current optimal is just *the* optimal)
- `Processing_Functions_cgg/Parameters/SLURMPARAMETERS_cgg_runAutoEncoder_v2.m` — the 47-dimension hyperparameter sweep harness that the Python pipeline must support

**MATLAB-side interop scope (narrow):**
The Python pipeline writes two kinds of files MATLAB code consumes:

1. **`CM_Table.mat`** — the per-trial confusion-matrix telemetry written by `cgg_saveValidationCMTable.m`. See `cgg_generateBlankCMTable.m` for the schema (NumWindows, DataNumber, TrueValue, plus per-classifier prediction columns and confidence columns). This is the main per-run output.
2. **`EncodingParameters.yaml`** — a YAML dump of the resolved config used for that run. The downstream `cgg_plotParameterSweep.m` uses these YAML files (together with the CM_Tables) to plot parameter-sweep results. **Critical requirement**: every Python-generated YAML must contain the same field schema across all runs in a sweep, even when individual values differ — `cgg_plotParameterSweep` scans field-by-field across runs and breaks if fields are missing. The MATLAB equivalent guarantees this via `cgg_setBaselineDynamicParameters` (which snapshots all dynamic-parameter fields up-front, ensuring they all appear in the YAML even if unused). The Python implementation must do the same: write a stable-schema YAML that includes every parameter field, even if a given run has a default/unused value.

Other artifacts (network weights, optimizer state, etc.) are PyTorch-native; a small **bidirectional network format converter** (PyTorch ↔ MATLAB `dlnetwork`) is provided for parity comparisons during early milestones but is not the main interop surface.

The downstream MATLAB scripts that consume Python output are:
- `DATA_cggAllNetworkEncoderResults.m` — aggregates the pipeline's results into the plot-ready data files
- `FIGURE_cggAllNetworkEncoderResults.m` (and `FIGURE_SFN_cggAllNetworkEncoderResults.m`) — the actual plotting scripts for this pipeline's paper
- `cgg_plotParameterSweep.m` — sweep-comparison plots; consumes the YAML field schema described above

(`FIGURE_cggPaperFigures.m` is for a different paper that doesn't use this pipeline — not in scope.)

**User-confirmed decisions driving this plan:**

| Decision | Choice |
|----------|--------|
| New pipeline location | `neural_data_decoding/` (fresh codebase, idiomatic Python project layout) |
| Reuse policy | Start completely fresh — nothing copied from earlier scaffolding; no other code in the repo modified |
| MATLAB analysis interop | Python writes `CM_Table.mat` files in the schema MATLAB analysis expects, runs from `DATA_cggAllNetworkEncoderResults` → `FIGURE_cggAllNetworkEncoderResults` unchanged |
| First parity target | Tiered: Logistic Regression → GRU+Classifier → Full Optimal, each as a separate milestone |
| Extra-credit milestone | After Milestone C, implement all currently-Supported-but-Unused options (Stitching/Fusion, Convolutional backbones, PCA full, MAE loss, SGDM) so the SLURM-sweep parameter space is fully functional |
| Structural fidelity | **Functional parity, not function-by-function structural parity.** Use Pythonic patterns where they improve the codebase, as long as parity tests still pass |
| SLURM-sweep parity | Every non-dead parameter in `SLURMPARAMETERS_cgg_runAutoEncoder_v2.m` must be functionally supported by Milestone C+CC end (so any sweep configuration the MATLAB pipeline can run, the Python pipeline can also run) |
| Educational notebooks | First-class deliverable — a complete curriculum that takes a MATLAB-native programmer to expert-level Python/PyTorch fluency on this specific pipeline |
| Implementer | AI agents reviewed by user — plan must be granular with specific files, signatures, and verification commands |

---

## The Optimal Configuration (Single Source of Truth)

This is what the Python pipeline must reproduce at the end of Milestone C. Every other config (Logistic, GRU+Classifier, sweep variants) is a partial subset or perturbation of this.

| Setting | Value | Notes |
|---------|-------|-------|
| `Epoch` | `Decision` (production) / `Synthetic_Easy` (testing) | Behavioral epoch alignment window |
| `Target` | `Dimension` | Classification target |
| `ModelName` | `GRU` | Active encoder architecture string |
| `ClassifierName` | `Deep LSTM - Dropout 0.5` | Active classifier architecture string |
| `HiddenSizes` | `[1000, 500, 250]` | Encoder/decoder hidden widths |
| `ClassifierHiddenSize` | `[250, 100, 50]` | Classifier hidden widths |
| `IsVariational` | `true` | VAE active |
| `EncoderOutputType` | `Stochastic` | Sampling layer placed after bottleneck |
| `MultipleInstanceLearningType` | `MIL` | Multi-axis softmax MIL active |
| `ConfidenceType` | `["Trial", "Task"]` | Both confidence heads active |
| `GradientClipType` | `Global` | Global L2-norm clipping across all params |
| `WeightedLoss` | `Inverse` | Inverse-frequency class weights |
| `Optimizer` | `ADAM` (→ Python `AdamW`) | Decoupled-weight-decay variant |
| `LossType_Decoder` | `MSE` | ELBO with MSE reconstruction |
| `LossType_Classifier` | `CrossEntropy` | |
| `WeightReconstruction` | 100 | |
| `WeightKL` | 1 | |
| `WeightClassification` | 10 | |
| `WeightConfidence` | 1 | |
| `WeightOffsetAndScale` | 0 | Augmentation loss disabled |
| `PriorProportion` | 0.9 | EMA prior normalization weight |
| `RescaleLossEpoch` | 0 | → EMA updates **every iteration** |
| `DynamicParameterSet` | `Soft Three-Stage Curriculum - Shortened` | Active curriculum schedule |
| `NumEpochsAutoEncoder` | 0 | Stage 1 (unsupervised) skipped in production |
| `NumEpochsFull` | 500 | Stage 2 (supervised) total epochs |
| `InitialLearningRate` | 0.001 | |
| `LearningRateEpochRamp` | 0 | No warmup |
| `WeightDelayEpoch` | 2, `WeightEpochRamp` | 3 (KL annealing onset) |
| `StitchingAndFusionLayer` | `''` | Disabled (enabled in Milestone CC) |

## First Commands After Plan Approval

These are the very first commands the implementer should run, in order, on approval:

```bash
# 1. Save the plan into the repo (this is what the user requested)
cp /Users/cgerrity/.claude/plans/that-s-exactly-what-i-noble-ripple.md \
   "/Users/cgerrity/Documents/MATLAB/Neural Data Reading/Plans/neural_data_decoding_plan.md"

# 2. Create the new pipeline directory
cd "/Users/cgerrity/Documents/MATLAB/Neural Data Reading"
mkdir -p neural_data_decoding/{src/neural_data_decoding,tests/{parity,unit,fixtures},configs,scripts,notebooks}
cd neural_data_decoding

# 3. Initialize Python project
python -m venv .venv
source .venv/bin/activate
# (write pyproject.toml per Milestone 0 spec, then:)
pip install -e ".[dev]"

# 4. Initialize git tooling
pre-commit install
nbstripout --install

# 5. Verify environment
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m neural_data_decoding --help
```

## Parity Doctrine: Functional, Not Structural

**The Python pipeline does not need to mirror the MATLAB file/function structure.** What it must mirror is:

- The architecture topology (same layer counts, same activations, same VAE sampling, same multi-head classifier, same MIL pooling)
- The loss components and their relative weights (same ELBO, same classification, same confidence PD, same EMA priors)
- The data pipeline contract (same K-fold strata, same augmentation magnitudes per epoch, same session-balanced batching)
- The output contract (`.mat` files in the exact folder hierarchy the MATLAB analysis scripts expect)

What it **does not** need to mirror:

- The 1-file-per-function decomposition (group related code Pythonically)
- The `cgg_*` naming convention (use Python-idiomatic names)
- The `CheckVararginPairs` varargin pattern (use dataclasses or pydantic configs)
- The script-vs-function dual-mode boilerplate at the top of every MATLAB file
- The flat-table parameter struct (use Hydra config composition properly)
- The bespoke monitor system (use W&B + a minimal `.mat` dump for the MATLAB analysis bridge)

Where there is a better Python practice, use it — provided the parity tests still pass. For example:
- Use `pytorch_lightning.LightningModule` for the training loop if it cleanly captures the two-stage lifecycle; otherwise plain PyTorch.
- Use `torch.utils.data.Sampler` for session-balanced batching, not a hand-rolled table.
- Use `nn.ModuleDict` for the multi-head classifier rather than a list of layerGraphs.
- Use `dataclasses.dataclass` (or pydantic `BaseModel`) for `cfg_Encoder`-shaped configs; load from YAML via Hydra/OmegaConf.
- Use `torch.optim.lr_scheduler` primitives where they cover the MATLAB schedule (custom subclass where they don't).
- Use `huggingface accelerate` or native `torch.distributed` for multi-GPU rather than reimplementing `cgg_procGradientAggregation`'s parfor.

The plan's file-layout suggestions below are **starting points, not mandates** — the implementer should refactor freely as long as the parity gates (below) still pass.

## Parity Goals (Tiered)

| Tier | Goal | Verification |
|------|------|--------------|
| **T1 — Design parity** | Same architecture topology, same loss components, same hyperparameter surface, same data preprocessing pipeline | Code review + architecture-graph diffing |
| **T2 — Single-step numerical parity** | Forward pass on identical input + weights matches within tolerance (1e-5 fp32, 1e-3 after BN) | `pytest` golden-vector tests; weights ported from MATLAB `.mat` checkpoints |
| **T3 — Statistical parity** | Same convergence behavior, accuracy distributions match across seeds (KS-test or paired-bootstrap) | 10-seed runs each side, compared per-epoch validation accuracy |
| **T4 — Output-format parity** | Python output `.mat` files load cleanly into MATLAB analysis scripts and produce same downstream figures | Run MATLAB `DATA_cggAllNetworkEncoderResults` + `FIGURE_cggPaperFigures` on Python output; visual diff |

**Bit-exact parity is explicitly NOT a goal** — ADAM, BatchNorm, RNG, and floating-point reductions differ between MATLAB and PyTorch in ways that cannot be made identical without rewriting native ops. Setting bit-exact as the bar would produce endless false-positive bug hunts.

---

## First-Target Configurations (Tiered Milestones)

Each milestone freezes a specific `cfg_Encoder` to target. Configs are kept as YAML in `neural_data_decoding/configs/` and exactly mirror the corresponding MATLAB parameter set.

### Target A — Logistic Regression on `Synthetic_Easy`
```
ModelName='Logistic Regression', Epoch='Synthetic_Easy', Target='Dimension'
HiddenSizes=[],  ClassifierName='Logistic',  LossType_Decoder='None'
IsVariational=false, ConfidenceType='', MultipleInstanceLearningType='None'
DynamicParameterSet='None',  Optimizer='ADAM',  WeightedLoss='Inverse'
GradientClipType='SubNetwork',  WeightConfidence=0,  PriorProportion=0
```
*Exercises:* data pipeline, classifier head, stratification, optimizer step, save/resume, .mat output writer.
*Skips:* encoder/decoder, VAE, curriculum, multi-head, confidence, MIL.

### Target B — GRU + Deep LSTM Classifier on `Synthetic_Easy`
```
ModelName='GRU', IsVariational=false, EncoderOutputType='Deterministic'
ClassifierName='Deep LSTM - Dropout 0.5', HiddenSizes=[256,128,64]
LossType_Decoder='None'  (classifier-only)
ConfidenceType='', MultipleInstanceLearningType='None'
DynamicParameterSet='No Dynamic Parameters'
```
*Exercises:* encoder dispatcher (`Simple` branch), RNN block builder, session-balanced batching, two-stage lifecycle (Stage 1 still degenerate since no decoder).
*Skips:* VAE sampling, KL annealing, confidence, MIL, curriculum.

### Target C — Full Optimal on `Synthetic_Easy` then `Decision`
```
PARAMETERS_OPTIMAL_cgg_runAutoEncoder_v3 verbatim:
ModelName='GRU', IsVariational=true, EncoderOutputType='Stochastic'
HiddenSizes=[1000,500,250], ClassifierHiddenSize=[250,100,50]
ConfidenceType=["Trial","Task"], MultipleInstanceLearningType='MIL'
GradientClipType='Global', WeightedLoss='Inverse'
WeightReconstruction=100, WeightKL=1, WeightClassification=10, WeightConfidence=1
PriorProportion=0.9, RescaleLossEpoch=0
DynamicParameterSet='Soft Three-Stage Curriculum - Shortened'
NumEpochsAutoEncoder=0  (skip Stage 1 in current optimal)  → NumEpochsFull=500
```
*Exercises:* everything in active path — full VAE, Stochastic sampling, dual confidence heads, MIL pooling, EMA-normalized multi-task loss, dynamic curriculum, gradient accumulation, two-stage lifecycle (degenerate Stage 1 in current Optimal since `NumEpochsAutoEncoder=0`).

**Moved to Milestone CC ("Extra Credit") — implemented after Milestone C as part of this plan:**
- Stitching/Fusion bridge (all variants)
- `LossType_Decoder='MAE'` and `'None'` (classifier-only) variants
- All Convolutional/ResNet `ModelName` variants
- PCA backbone (full implementation)
- SGDM optimizer
- `WeightedLoss=''` (unweighted classification)
- Learnable offset/scale augmentation layers

**Fully dropped from scope — not implemented at all:**
- `LossType_Classifier='CTC'` (MIL replaces CTC functionality in this pipeline)
- Legacy `NetworkTrainingVersion='Version 1'` path (`cgg_trainAllAutoEncoder.m` without `_v2`) — fully superseded; not implemented
- **Legacy v1 freeze mechanism only**: the `Freeze_cfg` parameter struct and `cgg_setFrozenNetwork.m` (v1, no version suffix) are NOT ported. The active **v2** path — `cgg_setFrozenNetwork_v2.m` together with the `FreezeParameters` schedule class from `cgg_generateFreezeParameters` — IS ported (see Milestone C). The v2 function applies the schedule's current value at the start of each epoch to actually freeze/unfreeze layers. Only the obsolete v1 variant is dropped.
- Legacy `cgg_lossNetwork_v2.m` / `cgg_lossComponentsParallel.m` (dead code in MATLAB, never wired)
- Legacy `cgg_trainNetworkParallel.m` / `cgg_trainCustomTrainingParallel*.m` (legacy parallel paths)

---

## Starting Point

The `neural_data_decoding/` directory does not exist yet. All files are new. The implementer should reference the MATLAB source code (`Processing_Functions_cgg/`) and `Codebase_Documentation.md` for the source-of-truth behavior — no other Python code is to be consulted or copied.

---

## Target Directory Structure (Pythonic)

This is a modern Python project layout: `src/` package layout (avoids implicit-namespace-package issues), tests at top level, configs as Hydra-composable YAML hierarchies, notebooks as a sibling deliverable, MATLAB-interop kept in a clearly bounded subpackage. The layout below is the implementer's starting suggestion — refactor freely to whatever is most Pythonic for the codebase, as long as parity gates still pass.

```text
neural_data_decoding/
├── pyproject.toml                         # PEP 621 — project metadata, deps, build config (Poetry / hatch / setuptools — pick one)
├── README.md                              # Quickstart, parity-status badges, links to notebooks
├── .gitignore
├── .pre-commit-config.yaml                # ruff + black + nbstripout
├── src/
│   └── neural_data_decoding/              # Top-level Python package (same name as the repo directory)
│       ├── __init__.py                    # Package metadata, version
│       ├── cli.py                         # CLI entry point (replaces run_autoencoder.py — use Typer or Hydra @main)
│       ├── config/                        # Config dataclasses (replaces MATLAB's CheckVararginPairs varargin pattern)
│       │   ├── __init__.py
│       │   ├── encoder_config.py          # Dataclass mirroring cfg_Encoder fields
│       │   ├── session_config.py          # Dataclass mirroring cfg_Session
│       │   └── slurm_config.py            # SLURM sweep parameter dataclass
│       ├── data/                          # Data pipeline
│       │   ├── __init__.py
│       │   ├── dataset.py                 # PyTorch Dataset, single-trial loading + augmentation
│       │   ├── samplers.py                # SingleSessionBatchSampler (every minibatch from ONE session)
│       │   ├── stratification.py          # Stratified K-fold partition manager
│       │   ├── normalization.py           # Per-trial normalization recipes
│       │   ├── augmentation.py            # Channel offset, white noise, random walk, time shift
│       │   ├── target_mapping.py          # Behavioral target encoding
│       │   └── mat_files.py               # HDF5-aware .mat I/O
│       ├── models/                        # Architecture
│       │   ├── __init__.py
│       │   ├── registry.py                # Architecture string → builder registry pattern
│       │   ├── encoder.py                 # Encoder builders (Simple/RNN/Conv)
│       │   ├── decoder.py                 # Decoder builders
│       │   ├── bottleneck.py              # Bottleneck builders
│       │   ├── classifier.py              # Multi-head classifier + confidence routing + MIL
│       │   ├── layers/                    # Custom layers
│       │   │   ├── __init__.py
│       │   │   ├── sampling.py            # VAE reparameterization layer
│       │   │   ├── pca.py                 # Frozen PCA encode/decode (CC)
│       │   │   └── mil_softmax.py         # Multi-axis softmax for MIL
│       │   └── stitching_fusion/          # Currently unused — implemented in Milestone CC
│       │       ├── __init__.py
│       │       ├── module.py              # Gemini cascade module
│       │       └── dispatcher.py          # Option-set dispatcher
│       ├── training/                      # Training engine
│       │   ├── __init__.py
│       │   ├── lifecycle.py               # Two-stage state machine (unsupervised → supervised)
│       │   ├── loop.py                    # Single-stage training loop
│       │   ├── gradient_accumulation.py   # Sub-batch chunking
│       │   ├── hardware.py                # Device detection, multi-GPU readiness
│       │   ├── checkpoint.py              # Resume-from-interruption (current state only, no optimizer state)
│       │   ├── pre_flight.py              # Existing-network detection + safety warning
│       │   ├── losses/
│       │   │   ├── __init__.py
│       │   │   ├── elbo.py                # MSE-ELBO (MAE variant in Milestone CC)
│       │   │   ├── classification.py      # Weighted cross-entropy
│       │   │   ├── confidence.py          # Trial + Task confidence PD-controller
│       │   │   ├── offset_and_scale.py    # Augmentation loss (Milestone CC)
│       │   │   ├── weight_decay.py        # AdamW-aware or grad-hook L2
│       │   │   └── multi_objective.py     # EMA prior normalizer + total-loss aggregator
│       │   ├── schedules/                 # Curriculum learning
│       │   │   ├── __init__.py
│       │   │   ├── base.py                # Schedule abstract base + piecewise-linear interpolation
│       │   │   ├── load.py                # Augmentation magnitudes (read by Dataset live)
│       │   │   ├── weights.py             # Per-loss-component weights
│       │   │   └── freeze.py              # Per-network freeze magnitudes
│       │   └── monitoring/
│       │       ├── __init__.py
│       │       ├── wandb_logger.py        # W&B integration
│       │       ├── cm_table.py            # The .mat CM_Table writer (MATLAB-readable)
│       │       └── memory_probe.py        # Proactive OOM detection (replaces cgg_getMemoryInformation)
│       ├── interop/                       # MATLAB ↔ Python bridge
│       │   ├── __init__.py
│       │   ├── cm_table_format.py         # CM_Table schema (matches cgg_generateBlankCMTable.m)
│       │   ├── folder_hierarchy.py        # Deterministic deeply-nested result dir (matches cgg_generateEncoderSubFolders_v3.m)
│       │   ├── network_converter.py       # Bidirectional PyTorch <-> MATLAB dlnetwork weight conversion
│       │   └── parameter_yaml.py          # Read/write .yaml of resolved cfg (matches MATLAB's EncodingParameters.yaml)
│       ├── sweeps/                        # SLURM sweep harness (replaces SLURMPARAMETERS_cgg_runAutoEncoder_v2.m)
│       │   ├── __init__.py
│       │   ├── parameter_space.py         # 47-dim parameter space definition
│       │   ├── submitit_launcher.py       # submitit + Hydra sweep submission
│       │   └── ray_tune_launcher.py       # Alternative: Ray Tune (chosen at runtime)
│       └── utils/
│           ├── __init__.py
│           ├── paths.py                   # Environment detection (Local / TEBA / ACCRE)
│           ├── seeding.py                 # Global seed control
│           ├── logging.py                 # Structured logging
│           └── matlab_axes.py             # Shape conversions (MATLAB SSCTB ↔ PyTorch NCHW)
├── configs/                               # Hydra config root (composable, not flat)
│   ├── base.yaml                          # Defaults from PARAMETERS_cgg_runAutoEncoder
│   ├── optimal.yaml                    # Defaults override for Optimal
│   ├── synthetic_easy.yaml                # SyntheticEasy override
│   ├── target_milestone/                  # One per Milestone target
│   │   ├── A_logistic_synthetic.yaml
│   │   ├── B_gru_classifier_synthetic.yaml
│   │   └── C_optimal_synthetic.yaml
│   ├── architecture/                      # Composable; one per ModelName string
│   │   ├── logistic_regression.yaml
│   │   ├── gru.yaml
│   │   └── variational_gru_dropout_0_5.yaml
│   ├── schedule/                          # Curriculum schedules
│   │   ├── none.yaml
│   │   └── soft_three_stage_curriculum_shortened.yaml
│   └── sweep/                             # SLURM sweep configs
│       └── optimal_sweep.yaml
├── tests/                                 # Top-level (Pythonic standard)
│   ├── conftest.py
│   ├── parity/                            # Tests against MATLAB reference
│   │   ├── test_stratification.py
│   │   ├── test_augmentation_seeded.py
│   │   ├── test_loss_forward.py
│   │   ├── test_dynamic_schedules.py
│   │   ├── test_cm_table_format.py        # T4 — round-trip
│   │   └── test_e2e_milestone_a/b/c/cc.py # T3 — convergence parity per milestone
│   ├── unit/
│   │   ├── test_single_session_sampler.py
│   │   ├── test_checkpoint_resume.py
│   │   ├── test_pre_flight_check.py
│   │   ├── test_oom_probe.py
│   │   └── test_network_converter.py
│   └── fixtures/
│       ├── golden_weights/                # MATLAB-trained .mat checkpoints
│       ├── golden_batches/                # (X, T, expected_loss) tuples
│       ├── reference_partitions/          # MATLAB K-fold partition .mat files
│       └── reference_cm_tables/           # MATLAB-generated CM_Table .mat files
├── notebooks/                             # First-class deliverable — see Milestone E for full curriculum
│   ├── README.md
│   ├── 00_orientation/
│   ├── 01_python_for_matlab_users/
│   ├── 02_numpy_and_pytorch_basics/
│   ├── 03_data_pipeline/
│   ├── 04_architecture/
│   ├── 05_training_loop/
│   ├── 06_loss_orchestration/
│   ├── 07_dynamic_curriculum/
│   ├── 08_output_and_analysis/
│   └── 09_production_deployment/
├── scripts/                               # Standalone utility scripts (not part of package)
│   ├── prepare_golden_fixtures.py         # Runs MATLAB in batch mode to generate parity fixtures
│   └── verify_mat_roundtrip.py            # CI helper for T4 gate
└── docs/                                  # First-class deliverable — see Milestone F
    ├── narrative/                         # MkDocs Material — user guides, cookbook, ADRs, glossary
    │   ├── index.md
    │   ├── quickstart.md
    │   ├── user_guide/
    │   ├── cookbook/
    │   ├── concepts/
    │   ├── deployment/
    │   ├── glossary.md
    │   ├── troubleshooting.md
    │   ├── contributing.md
    │   └── adrs/                          # Architecture Decision Records
    ├── api/                               # Sphinx — auto-generated API reference
    │   ├── conf.py
    │   ├── index.rst
    │   ├── data.rst
    │   ├── models.rst
    │   ├── training.rst
    │   ├── interop.rst
    │   └── sweeps.rst
    ├── mkdocs.yml                         # MkDocs config (lives here, not at repo root, to keep root clean)
    └── build/                             # Generated output (gitignored)
```

**Notes on the Pythonic layout:**

- `src/` layout (PEP 517/518 standard) — protects against import-shadowing during testing and forces `pip install -e .` for development, which catches packaging bugs early.
- The Python package is `neural_data_decoding` (same name as the repo directory — clean, no suffix mismatch).
- Configs are Hydra-composable: `optimal.yaml` composes `base.yaml` + `architecture/variational_gru_dropout_0_5.yaml` + `schedule/soft_three_stage_curriculum_shortened.yaml`. This replaces the MATLAB pattern of "one flat parameter file overrides another".
- `tests/` is at the top level (Python convention), not nested in `src/`. Fixtures stay in `tests/fixtures/`.
- `interop/` is the bounded MATLAB-compatibility subpackage — everything that has to match MATLAB lives there, so the rest of the codebase stays free to be idiomatic Python.
- `sweeps/` exposes both Submitit (low-friction MATLAB equivalent) and Ray Tune (fancier hyperparameter optimization), chosen at run-time per the user's preference.
- `scripts/` holds standalone Python utility scripts not part of the importable package.

---

## Milestone Sequence

**Path conventions used in this section:** unless otherwise noted, paths in the task tables are relative to `src/neural_data_decoding/` (e.g., `data/dataset.py` means `neural_data_decoding/src/neural_data_decoding/data/dataset.py`). Test paths are relative to `tests/`. Config paths are relative to `configs/`. Script paths are relative to `scripts/`.



### Milestone 0 — Foundation (1 week)

**Goal:** save this plan into the repo, dependency lock, scaffold the new pipeline, set up test harness so subsequent milestones can verify parity continuously.

**Step 0 (first action on plan approval):** Copy this plan file to `Plans/neural_data_decoding_plan.md` in the repo so the active specification lives in the codebase. This becomes the implementation reference for all subsequent milestones.

All paths relative to `neural_data_decoding/`.

| Task | File(s) | Notes |
|------|---------|-------|
| Scaffold project | `pyproject.toml`, `src/neural_data_decoding/__init__.py`, all subpackage `__init__.py`s | PEP 621 project. Core deps: torch>=2.1, numpy, scipy, h5py, mat73, pandas, scikit-learn, pyyaml, hydra-core, omegaconf, wandb. Dev extras `[dev]`: pytest, pytest-cov, ruff, black, pre-commit, nbstripout, jupyter, interrogate. Docs extras `[docs]`: mkdocs-material, mkdocstrings, mike, sphinx, sphinx-autodoc-typehints, myst-parser. Cluster extras `[cluster]`: submitit, ray\[tune\]. |
| Pre-commit & CI scaffolding | `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (if applicable) | ruff + black + nbstripout |
| Environment detection | `src/neural_data_decoding/utils/paths.py` | Detect Local / TEBA / ACCRE via `os.environ`; return base data directories |
| .mat file I/O | `src/neural_data_decoding/data/mat_files.py` | Auto-detect v7.3 (HDF5) vs older format from 128-byte header; use `mat73` for HDF5, `scipy.io.loadmat` otherwise |
| Normalization recipes | `src/neural_data_decoding/data/normalization.py` | Implement the `cgg_selectNormalization.m` switch, including the Optimal recipe |
| Dataset | `src/neural_data_decoding/data/dataset.py` | PyTorch Dataset; reads `.mat` per trial; applies augmentation reading live `LoadSchedule` state; PCA hook for Milestone CC |
| Augmentation | `src/neural_data_decoding/data/augmentation.py` | Channel offset, white noise, random walk, time shift — re-randomized per `__getitem__` |
| Single-session sampler | `src/neural_data_decoding/data/samplers.py::SingleSessionBatchSampler` | Every minibatch contains trials from **one session only** — this is intentional, supports future per-session stitching layers (see Critical Notes) |
| Stratification | `src/neural_data_decoding/data/stratification.py` | Recursive hierarchical K-fold splitter matching MATLAB strata IDs exactly |
| Hardware detection | `src/neural_data_decoding/training/hardware.py` | Detect GPU/CPU; report device; multi-GPU readiness hook (initially single-GPU only) |
| Memory probe | `src/neural_data_decoding/training/monitoring/memory_probe.py` | Proactive OOM detection via `torch.cuda.mem_get_info()` + headroom check; replaces `cgg_getMemoryInformation` |
| Pre-flight existing-network check | `src/neural_data_decoding/training/pre_flight.py` | Before training, scan target output folder; if any existing checkpoint files match the resolved config, **abort with informative error** listing the files; require user to manually delete before re-running |
| Seeding | `src/neural_data_decoding/utils/seeding.py` | `set_global_seed(seed)` covering torch, numpy, random, CUDA |
| Axis converters | `src/neural_data_decoding/utils/matlab_axes.py` | Round-trip-tested converters between MATLAB SSCTB and PyTorch (N, C, H, W) / (N, T, C) shapes |
| CLI entry | `src/neural_data_decoding/cli.py` | `python -m neural_data_decoding ...` or Hydra `@main` — loads composed config, dispatches to training |
| Test harness | `tests/conftest.py` | Pytest fixtures including: MATLAB reference data loaders, golden-weights loader, golden-batch loader, seed-control fixture |
| Stratification parity test | `tests/parity/test_stratification.py` | Generates a MATLAB partition once via `scripts/prepare_golden_fixtures.py`, then asserts Python strata IDs match per trial |
| .mat round-trip baseline test | `tests/parity/test_cm_table_format.py` | Write a minimal CM_Table from Python; load in MATLAB via batch; assert structure matches `cgg_generateBlankCMTable.m` output exactly |
| Fixture preparation script | `scripts/prepare_golden_fixtures.py` | Runs MATLAB in batch mode once to generate all the reference `.mat` files the parity tests need |
| MkDocs scaffold | `docs/narrative/`, `docs/mkdocs.yml` | Empty pages with placeholder content; navigation structure matching the Milestone F inventory |
| Sphinx scaffold | `docs/api/`, `docs/api/conf.py` | Autodoc + autosummary + napoleon + myst-parser configured; one autosummary entry per top-level subpackage |
| Unified docs build script | `scripts/build_docs.sh` | Builds both MkDocs and Sphinx, places outputs under `docs/build/{narrative,api}/` |
| Docs CI gates | `.github/workflows/docs.yml` (or local pre-commit) | `mkdocs build --strict`, `sphinx-build -W`, `interrogate --fail-under=100 --omit-covered-files` |
| First ADR | `docs/narrative/adrs/001_tiered_parity_not_bit_exact.md` | Template the ADR format with the first decision recorded |

**Verification:**
```bash
cd neural_data_decoding
pip install -e ".[dev,docs]"
pre-commit install
python scripts/prepare_golden_fixtures.py            # one-time MATLAB run
pytest tests/parity/test_stratification.py -v        # T3 strata parity
pytest tests/parity/test_cm_table_format.py -v       # T4 round-trip baseline
pytest tests/unit/                                   # all unit tests
bash scripts/build_docs.sh                           # builds both narrative + API docs
interrogate -vv src/ --fail-under=100                # docstring coverage gate
python -m neural_data_decoding cli +target_milestone=A_logistic_synthetic --dry-run
```

### Milestone A — Logistic Regression Tracer Bullet (2 weeks)

**Goal:** end-to-end training of the simplest configuration; lock down the orchestration / loss / checkpoint / .mat-output backbone. No VAE, no MIL, no curriculum.

| Task | File(s) | Notes |
|------|---------|-------|
| Config | `configs/target_milestone/A_logistic_synthetic.yaml` | Exactly mirrors a MATLAB run with `ModelName='Logistic Regression'` |
| Architecture dispatcher | `models/registry.py` | Initial table contains only `'Logistic Regression'`; extensible map for B/C |
| Logistic builder | `models/classifier.py::build_logistic_classifier()` | Mirrors `cgg_selectClassifier.m::'Logistic'` branch |
| Multi-head + class weights | `models/classifier.py::ClassifierHead` | Per-dim cross-entropy with `WeightedLoss='Inverse'` class weights |
| Loss orchestrator (minimal) | `training/losses/multi_objective.py` | Just classification loss + L2; no EMA priors yet |
| Single-stage training loop | `training/loop.py` | Mirrors `cgg_trainNetwork.m` skeleton: epoch loop, mini-batch loop, optimizer step, validation, save |
| Two-stage orchestrator | `training/lifecycle.py` | Implement the state machine but Stage 1 is no-op (`NumEpochsAutoEncoder=0`) |
| Gradient accumulation | `training/gradient_accumulation.py` | Mirror `cgg_procGradientAggregation`; `parfor` → `for` initially (no multi-worker yet) |
| Session-balanced sampler | `data/samplers.py::SessionBalancedBatchSampler` | Replaces stock `DataLoader` batching; mirrors `cgg_procAllSessionMiniBatchTable` |
| Checkpoint state machine | `training/checkpoint.py` | Loads/saves `model.state_dict() + optimizer.state_dict() + epoch + iteration + best_val_acc`; mirrors `cgg_get/saveIterationInformation` |
| Folder hierarchy | `analysis/folder_hierarchy.py` | Generates the deterministic deeply-nested result dir matching `cgg_generateEncoderSubFolders_v3.m` |
| .mat output writer | `analysis/mat_output.py` | Writes `Encoder-Current.mat`, `Classifier-Current.mat`, `Encoder-Optimal.mat`, `IterationInformation.mat`, `ValidationCMTable.mat` with field names/struct shapes matching MATLAB |
| Wire CLI | `run_autoencoder.py` | Loads config, builds, trains, writes .mat output |
| Parity tests | `tests/parity/test_end_to_end_milestone_A.py` | T3: convergence on Synthetic_Easy across 5 seeds matches MATLAB within 2σ |
| .mat round-trip test | `tests/unit/test_mat_output_format.py` | T4: Python writes → MATLAB loads → asserts structure matches reference |

**Verification:**
```bash
# Pre-flight check (will abort if existing checkpoints found — user manually deletes if intentional)
python -m neural_data_decoding check-existing +target_milestone=A_logistic_synthetic --fold 1

# Train Python
python -m neural_data_decoding train +target_milestone=A_logistic_synthetic --fold 1

# Train MATLAB reference (one-time, results saved to separate folder via the network converter)
matlab -batch "cgg_runAutoEncoder(1, 'Epoch', 'Synthetic_Easy', 'ModelName', 'Logistic Regression')"

# Convert MATLAB weights to PyTorch via the bidirectional converter for T2 forward-pass parity
python scripts/convert_matlab_to_pytorch.py --from <matlab_dir> --to <fixture_dir>

# Compare outputs
pytest tests/parity/test_e2e_milestone_a.py -v

# Run MATLAB analysis on Python output (T4 verification — produces plot-ready data)
matlab -batch "DATA_cggAllNetworkEncoderResults()"
matlab -batch "FIGURE_cggAllNetworkEncoderResults()"   # Should render plots from Python results
```

### Milestone B — GRU + Classifier (2 weeks)

**Goal:** add the encoder pathway and most of the training-loop sophistication. Still no VAE, no confidence, no MIL, no curriculum.

| Task | File(s) | Notes |
|------|---------|-------|
| Config | `configs/target_milestone/B_gru_classifier_synthetic.yaml` | `ModelName='GRU'`, `IsVariational=false`, `LossType_Decoder='None'` |
| Architecture YAML | `configs/architectures/gru.yaml` | Mirrors the `'GRU'` case from `PARAMETERS_cgg_constructNetworkArchitecture` |
| Encoder builder | `models/encoder.py` | Simple-branch path: feedforward / GRU / LSTM stacks; mirrors `cgg_selectEncoder` + `cgg_constructSimpleCoder` |
| Bottleneck builder | `models/bottleneck.py` | Mirrors `cgg_selectBottleNeck` Simple branch (flatten + FC) |
| Decoder builder (no-op stub) | `models/decoder.py` | Just enough to support `LossType_Decoder='None'`; full impl in Milestone C |
| Architecture dispatcher expansion | `models/registry.py` | Adds `'GRU'`, `'LSTM'`, `'Feedforward - ReLU'` cases from PARAMETERS table |
| Classifier with weighted loss | `models/classifier.py::DeepLSTMClassifier` | Mirrors `cgg_selectClassifier::'Deep LSTM - Dropout 0.5'` |
| Save Encoder-Current/Optimal | `analysis/mat_output.py` | Extend writer for the encoder state |
| Single-step parity test | `tests/parity/test_loss_forward.py` | T2: load MATLAB-trained GRU encoder weights into Python, forward-pass identical input, assert classification logits match within 1e-5 |
| Golden weight fixtures | `tests/fixtures/golden_weights/` | Save MATLAB encoder/classifier weights as `.mat`; loader in `conftest.py` |
| Convergence parity test | `tests/parity/test_end_to_end_milestone_B.py` | T3: 5-seed convergence comparison |

**Verification:**
```bash
pytest tests/parity/test_loss_forward.py -v             # T2 single-step
pytest tests/parity/test_e2e_milestone_b.py -v          # T3 convergence
python -m neural_data_decoding check-existing +target_milestone=B_gru_classifier_synthetic --fold 1
python -m neural_data_decoding train +target_milestone=B_gru_classifier_synthetic --fold 1
matlab -batch "DATA_cggAllNetworkEncoderResults(); FIGURE_cggAllNetworkEncoderResults()"  # T4 dashboard renders
```

### Milestone C — Full Optimal (4 weeks)

**Goal:** complete the active production path. VAE sampling, dual confidence heads, MIL pooling, EMA-normalized multi-task loss, dynamic curriculum, full two-stage lifecycle, hardware-aware accumulation. After this milestone the pipeline is production-equivalent for Optimal.

| Task | File(s) | Notes |
|------|---------|-------|
| Config | `configs/target_milestone/C_optimal_synthetic.yaml`, then `_decision.yaml` | Verbatim mirror of `PARAMETERS_OPTIMAL_cgg_runAutoEncoder_v3.m` |
| VAE sampling layer | `models/layers/sampling.py` | Reparameterization trick; supports `'Stochastic'` placement (after bottleneck) and `'Deterministic'` placement (pre-decoder) per `EncoderOutputType` |
| Decoder builder | `models/decoder.py` | Simple branch + symmetric expansion; mirrors `cgg_selectDecoder` |
| Variational architecture | `models/registry.py` | Add `'Variational GRU - Dropout 0.5'` etc. cases; correctly insert sampling layer per `EncoderOutputType` |
| Multi-head classifier | `models/classifier.py::MultiHeadClassifier` | Shared input → N parallel heads; mirrors `cgg_constructClassifierArchitecture` |
| Confidence routing | `models/classifier.py::add_confidence_heads()` | Adds Trial Confidence regression branch and Task Confidence branch (`cgg_addTaskConfidenceToClassifier`) |
| MIL softmax pooling | `models/classifier.py::MILSoftmaxLayer` | Mirrors `cgg_softmaxLayer.m`, gated by `MultipleInstanceLearningType='MIL'` |
| ELBO loss (MSE) with **NaN-masked reconstruction** | `training/losses/elbo.py` | Mirrors `cgg_lossELBO_v2.m`. Reconstruction term uses `Mask_NaN = ~torch.isnan(target)` so removed-channel positions contribute zero to the loss. **Two-tensor flow:** encoder receives the NaN-zeroed input; the loss receives the original NaN-preserving target. KL term emitted separately. See Critical Note #38. |
| Two-tensor input plumbing | `data/dataset.py`, `training/loop.py` | Dataset emits both `x_input_to_encoder` (NaN→0) and `x_reconstruction_target` (NaN preserved) per trial. Training loop passes the first to the encoder and the second to the reconstruction loss. |
| NaN-replacement at encoder input | `models/layers/nan_to_zero.py` (or inline in encoder builder) | Mirrors the `RemoveNaNFunc = @(x) cgg_setNaNToValue(x,0)` set as `sequenceInputLayer`'s `Normalization` function. Implement as a leading `nn.Module` in the encoder graph. |
| Classification loss with weights | `training/losses/classification.py` | Cross-entropy + per-class weights + optional `WantBatchCorrection` per-batch prior correction |
| Confidence PD controller | `training/losses/confidence.py` | Interpolates predictions toward baseline by self-reported confidence; accumulates confidence-budget regularizer. **Must match `cgg_lossConfidence.m` precisely** — fixture-based T2 test required |
| Loss orchestrator (full) | `training/losses/multi_objective.py` | Reconstruction + KL + Classification + Confidence; EMA prior normalization with `PriorProportion=0.9` mirroring `cgg_getLossInformation` |
| L2 weight decay (Pythonic) | `training/losses/weight_decay.py` | Use **`torch.optim.AdamW`** by default (decoupled weight decay — the modern standard, mathematically cleaner than MATLAB's grad-side L2 on Adam). For SGDM use `SGD(weight_decay=...)`. Document in notebook 06.8 why this differs from MATLAB and why the AdamW path is preferred. |
| Schedule base class | `training/schedules/base.py` | `Schedule` abstract base with `.update(epoch)` and `.current_value` property; piecewise-linear interpolation mirroring `cgg_calculateDynamicValue.m` |
| Augmentation schedule | `training/schedules/load.py` | Stateful object exposing `current_STDChannelOffset`, `current_STDWhiteNoise`, etc. **Dataset must read live state at `__getitem__` time**, not at epoch start. Mirrors `cgg_generateLoadParameters_v2`. |
| Loss-weight schedule | `training/schedules/weights.py` | Exposes `current_WeightReconstruction`, `current_WeightKL`, etc.; loss orchestrator reads at each forward pass. Mirrors `cgg_generateLossWeights_v2`. |
| Freeze schedule | `training/schedules/freeze.py` | Per-network freeze magnitude; applied at start of each epoch via an `apply_freeze_schedule(net, schedule, epoch)` function. Mirrors `cgg_generateFreezeParameters` + `cgg_setFrozenNetwork_v2`. |
| KL annealing | `training/loop.py` | `cgg_annealWeight` for KL term, applied via `WeightParameters.WeightKL = annealed_value` |
| Learning-rate schedule | `training/loop.py` | Step-decay + warmup ramp mirroring `cgg_getLearningRate.m` |
| Global gradient clip | `training/loop.py` | `GradientClipType='Global'` → `torch.nn.utils.clip_grad_norm_`; `'SubNetwork'` branch deferred |
| Hardware-aware accumulation | `training/accumulation_size.py` | Replicates the 3 conditional rewrites in `PARAMETERS_cgg_runAutoEncoder` (StitchingAndFusionLayer / ClassifierHiddenSize / Synthetic) |
| Two-stage lifecycle | `training/lifecycle.py` | Full implementation: Stage 1 unsupervised → optimal-weight handoff → Stage 2 supervised with classifier added. Checkpoint state machine reads `CurrentIteration` for resume (NOT Optimal); see Critical Note #2. The MATLAB decision tree at `cgg_trainAllAutoEncoder_v2.m:171–221` is the reference. |
| Monitor save (.mat compat) | `training/monitoring/cm_table.py` | Writes `CM_Table.mat` per validation pass in the schema from `cgg_generateBlankCMTable.m`. Also writes `CurrentIteration.mat` + `OptimalIteration.mat` for resume + high-water-mark tracking. **Does NOT save optimizer state** (matches MATLAB; intentional). W&B logging runs in parallel for live telemetry. |
| Augmentation per-call contract | `data/dataset.py`, `data/augmentation.py` | Augmentation **re-randomized on every `__getitem__`** (not cached), reading current `LoadParameters` state |
| Curriculum YAMLs | `configs/schedule/soft_three_stage_curriculum_shortened.yaml` | Mirrors the corresponding case in `PARAMETERS_cgg_selectDynamicParameters.m` |
| T2 tests for each loss | `tests/parity/test_loss_forward.py` | Golden-vector test: load MATLAB-saved (X, T, weights), forward to each loss component, assert match |
| T2 test for confidence | `tests/parity/test_confidence_pd_controller.py` | Specifically verifies the PD-controller behavior since this is the highest-risk port |
| T2 test for dynamic params | `tests/parity/test_dynamic_schedules.py` | Verify each schedule produces the same per-epoch values as MATLAB |
| End-to-end convergence | `tests/parity/test_end_to_end_milestone_C.py` | T3: 10-seed comparison on Synthetic_Easy, then a single multi-day run on `Decision` |

**Verification:**
```bash
# Single-step parity
pytest tests/parity/ -v

# Pre-flight + full Synthetic_Easy run
python -m neural_data_decoding check-existing +target_milestone=C_optimal_synthetic --fold 1
python -m neural_data_decoding train +target_milestone=C_optimal_synthetic --fold 1

# Real-data: Decision epoch, single fold (multi-day run; supports resume-from-interruption)
python -m neural_data_decoding check-existing +target_milestone=C_optimal_decision --fold 1
python -m neural_data_decoding train +target_milestone=C_optimal_decision --fold 1

# Interrupt + resume test (kill the process mid-training, restart same command — should pick up from last save)
# T4 dashboard compatibility
matlab -batch "DATA_cggAllNetworkEncoderResults(); FIGURE_cggAllNetworkEncoderResults(); FIGURE_SFN_cggAllNetworkEncoderResults()"
```

### Milestone CC — Extra-Credit Feature Implementation (4–5 weeks)

**Goal:** implement all currently-Supported-but-Unused options in the active SLURM sweep parameter space, so the Python pipeline supports every non-dead parameter combination the MATLAB pipeline supports. After this milestone the Python pipeline reaches feature-completeness for parameter-sweep experiments.

This milestone is treated as a first-class part of the plan, not a deferral.

| Sub-milestone | Topic | Notes |
|---------------|-------|-------|
| CC.1 | Convolutional / ResNet architectures | Implement the full set from `PARAMETERS_cgg_constructNetworkArchitecture.m` — the 25+ ConvX/ResnetX/Multi-Filter variants. Includes split-area handling, ResNet path merging, post-decoder convolution, pre-decoder convolution, learnable offset/scale augmentation layers. |
| CC.2 | PCA backbone | Full implementation of frozen PCA encode/decode layers (`models/layers/pca.py`). Pre-compute PCA components via `sklearn.decomposition.PCA` once per fold; inject as `nn.Linear` weights with `requires_grad=False`. |
| CC.3 | MAE reconstruction loss | Implement the `LossType_Decoder='MAE'` branch via a Mean Absolute Error ELBO kernel (mirrors `cgg_lossELBO_MAE.m`); selectable via config. |
| CC.4 | SGDM optimizer | Add SGDM as a configurable optimizer choice; verify all schedules / freeze logic work identically with both ADAM and SGDM. |
| CC.5 | Stitching & Fusion bridge | Implement all four named option-sets from `PARAMETERS_cgg_constructStitchingAndFusionNetwork.m` (`Default`, `Feedforward`, the three Gemini variants). Connects to the encoder/decoder graphs via the pre-encoder / post-decoder hook described in `Codebase_Documentation.md`. **This is significant work** — the Gemini module is a cascaded multi-area fusion architecture with multiple option dimensions (kernel sets, reduction methods, cascade strides). |
| CC.6 | Learnable offset/scale augmentation | Full implementation of the `cgg_lossOffsetAndScale.m` augmentation loss + the corresponding decoder-side learnable augmentation block. Gated by `WantLearnableOffset`/`WantLearnableScale` flags. |
| CC.7 | `WeightedLoss=''` (unweighted) path | Implement the disabled-class-weighting branch as a config-selectable alternative to `'Inverse'`. |
| CC.8 | Full SLURM sweep parameter coverage | Audit `SLURMPARAMETERS_cgg_runAutoEncoder_v2.m`'s 47-dim sweep; ensure every non-dead value is supported by the Python pipeline. Add integration tests that sweep representative slices of the parameter space and verify training succeeds (not parity — just non-crash). |

**Verification:**
```bash
# Per-feature unit tests
pytest tests/parity/test_loss_forward.py -v -k "mae or sgdm or offset_scale"

# Sweep coverage smoke test
python -m neural_data_decoding sweep +sweep=optimal_sweep --dry-run    # Just enumerate; no training
python -m neural_data_decoding sweep +sweep=optimal_sweep --max-runs 4 # Actually train 4 randomly-selected configs

# End-to-end with a Convolutional architecture
python -m neural_data_decoding train +target_milestone=CC_conv_resnet_synthetic --fold 1

# End-to-end with Stitching/Fusion enabled
python -m neural_data_decoding train +target_milestone=CC_gemini_synthetic --fold 1
```

### Milestone D — Cluster Deployment & SLURM Sweep (1 week)

**Goal:** make the Python pipeline usable on ACCRE with the same SLURM sweep model the MATLAB pipeline uses.

| Task | File(s) | Notes |
|------|---------|-------|
| SLURM submission helper | `src/neural_data_decoding/sweeps/submitit_launcher.py` | Wraps `submitit` to enqueue runs; mirrors the SLURMChoice/SLURMIDX dispatch model from `SLURMPARAMETERS_cgg_runAutoEncoder_v2.m` |
| Ray Tune alternative | `src/neural_data_decoding/sweeps/ray_tune_launcher.py` | Optional: smarter hyperparameter search (BOHB, ASHA) for sweeps |
| Hydra sweep config | `configs/sweep/optimal_sweep.yaml` | Encodes the active hyperparameter sweep (full 47-dim space from `SLURMPARAMETERS_cgg_runAutoEncoder_v2.m`) |
| ACCRE detection | `src/neural_data_decoding/utils/paths.py` | Verify env-var detection works; integration test on actual cluster |
| Result aggregator | (existing MATLAB scripts) | Verify MATLAB `DATA_cggAllNetworkEncoderResults` cleanly aggregates Python sweep output |

### Milestone E — Educational Curriculum (first-class deliverable, runs in parallel)

**Goal:** anyone with MATLAB experience and no Python background should be able to work through these notebooks and emerge able to extend, debug, and operate the `neural_data_decoding` pipeline as an expert. The notebooks are not an afterthought — they are an equal-weight deliverable to the production code.

**Authoring rule:** every concept the production code uses must be teachable from these notebooks. If a reader can't trace a feature from "first principles" to "working in production code" through the curriculum, the curriculum has a gap.

**Authoring cadence:** notebooks are authored alongside the milestone that introduces the corresponding code. Milestone A code → modules 00–03 + the relevant Milestone-A-specific notebooks; B → 04–05; C → 06–08; D → 09. No code module ships without its companion notebook.

**Every notebook follows the same template:**
1. **What MATLAB does** — paste the actual `cgg_*` MATLAB code, explain in plain English what it does and why.
2. **The Python concept(s) you need** — explain the underlying Python/PyTorch concept from first principles, with worked-out micro-examples.
3. **The neural_data_decoding implementation** — show the actual production code from the new pipeline, annotated line-by-line, with cross-references to the MATLAB source.
4. **Hands-on exercises** — small problems the reader implements themselves, with hidden-cell solutions.
5. **Diagnostic / debugging walkthrough** — common errors a MATLAB-native programmer will hit, what they look like, how to fix them.
6. **Further reading** — links to PyTorch docs, key Python style guides, the relevant section of `Codebase_Documentation.md`.

#### Module 00 — Orientation (no prerequisites)

| # | Notebook | Topic |
|---|----------|-------|
| 00.1 | `welcome.ipynb` | Tour of the curriculum, prerequisite graph, how to use Jupyter |
| 00.2 | `set_up_your_environment.ipynb` | Install Python, set up venv / conda, install `neural_data_decoding`, run first hello-world cell |
| 00.3 | `the_matlab_to_python_mental_model.ipynb` | The single biggest mindset shifts (everything-is-an-object, 0-indexing, indentation, mutable vs immutable, namespaces, modules) |

#### Module 01 — Python for MATLAB Users

| # | Notebook | MATLAB analog |
|---|----------|---------------|
| 01.1 | `syntax_basics.ipynb` | scripts, functions, variables |
| 01.2 | `control_flow.ipynb` | if/else/for/while — and why Python uses indentation |
| 01.3 | `functions_and_lambdas.ipynb` | function declaration, `varargin` vs `*args/**kwargs`, lambda vs anonymous function |
| 01.4 | `classes_and_oop.ipynb` | classdef, inheritance, methods — Python equivalents |
| 01.5 | `modules_and_imports.ipynb` | MATLAB path vs Python packages; `from x import y` |
| 01.6 | `error_handling.ipynb` | try/catch → try/except; how to read a Python traceback |
| 01.7 | `dataclasses_and_typed_configs.ipynb` | replacing `CheckVararginPairs` with `@dataclass` and pydantic |
| 01.8 | `the_python_standard_library_for_matlab_users.ipynb` | `os`, `pathlib`, `json`, `yaml`, `logging` |

#### Module 02 — NumPy & PyTorch Basics

| # | Notebook | MATLAB analog |
|---|----------|---------------|
| 02.1 | `numpy_vs_matlab_arrays.ipynb` | array creation, slicing, broadcasting, view vs copy — every subtle gotcha |
| 02.2 | `array_axis_conventions.ipynb` | MATLAB's `'SSCTB'` vs PyTorch's `(N, C, H, W)` — what each dim means and how to convert (`utils/matlab_compat.py` walkthrough) |
| 02.3 | `loading_mat_files.ipynb` | `scipy.io.loadmat` vs `mat73` vs `h5py` — when each fails and how to debug |
| 02.4 | `pytorch_tensors_intro.ipynb` | `torch.Tensor` vs `np.ndarray`, device placement (CPU/GPU), dtype |
| 02.5 | `autograd_basics.ipynb` | the `requires_grad` flag, `.backward()`, computational graphs — what MATLAB's `dlfeval` does and how PyTorch does it differently |
| 02.6 | `nn_module_vs_layergraph.ipynb` | `layerGraph` and `dlnetwork` vs `nn.Module` / `nn.Sequential` / `nn.ModuleDict` |
| 02.7 | `optimizers_and_learning_rates.ipynb` | `trainNetwork` options vs `torch.optim.Adam` |
| 02.8 | `nan_handling.ipynb` | MATLAB's implicit NaN tolerance vs PyTorch's strict NaN propagation; `torch.nan*` ops, masking patterns |

#### Module 03 — Data Pipeline (companion to Milestone 0 & A)

| # | Notebook | Reference to |
|---|----------|--------------|
| 03.1 | `dataset_vs_filedatastore.ipynb` | `cgg_loadDataArray` ↔ `neural_data_decoding.data.dataset` |
| 03.2 | `dataloader_and_collation.ipynb` | how MATLAB iterates fileDatastores vs how PyTorch's `DataLoader` does it |
| 03.3 | `the_session_balanced_sampler.ipynb` | `cgg_procAllSessionMiniBatchTable` ↔ `data.samplers.SessionBalancedBatchSampler` |
| 03.4 | `kfold_stratification_deep_dive.ipynb` | `cgg_getKFoldPartitions` recursive splitting ↔ Python implementation; parity test walkthrough |
| 03.5 | `normalization_recipes.ipynb` | `cgg_selectNormalization` string-driven dispatch ↔ Pythonic implementation |
| 03.6 | `augmentation_per_call_contract.ipynb` | why augmentation must re-randomize on every `__getitem__` (the silent-parity-loss trap) |

#### Module 04 — Architecture (companion to Milestone B)

| # | Notebook | Reference to |
|---|----------|--------------|
| 04.1 | `architecture_string_dispatcher.ipynb` | how `PARAMETERS_cgg_constructNetworkArchitecture`'s 47 string options work; how `neural_data_decoding.models.registry` implements the registry pattern |
| 04.2 | `building_a_simple_encoder.ipynb` | walk through `cgg_constructSimpleCoder` and its Python equivalent |
| 04.3 | `rnn_building_blocks.ipynb` | GRU/LSTM in MATLAB vs PyTorch; batch_first, hidden state handling |
| 04.4 | `convolutional_backbones.ipynb` | (foreshadow — currently out of scope; introduces the patterns) |
| 04.5 | `the_bottleneck.ipynb` | flatten + FC, why MATLAB has this exact structure |
| 04.6 | `multi_head_classifier.ipynb` | `nn.ModuleDict` for the multi-head case; replicate the MATLAB structure |
| 04.7 | `weighted_classification_loss.ipynb` | `WeightedLoss='Inverse'` mechanics |
| 04.8 | `weight_initialization_he_vs_pytorch_defaults.ipynb` | why we explicitly call `nn.init.kaiming_normal_` instead of trusting PyTorch defaults; how MATLAB's `'he'` differs from PyTorch's `nn.Linear` default; a parity demo |

#### Module 05 — Training Loop (companion to Milestone B/C)

| # | Notebook | Reference to |
|---|----------|--------------|
| 05.1 | `the_custom_training_loop.ipynb` | walk through `cgg_trainNetwork` and the Python equivalent end-to-end |
| 05.2 | `gradient_accumulation.ipynb` | why MATLAB needs `cgg_procGradientAggregation`; PyTorch's native pattern |
| 05.3 | `gradient_clipping.ipynb` | `Global` vs `SubNetwork` clip; `torch.nn.utils.clip_grad_norm_` |
| 05.4 | `learning_rate_scheduling.ipynb` | step-decay + warmup; `torch.optim.lr_scheduler` |
| 05.5 | `checkpoint_resume_state_machine.ipynb` | the `cgg_trainAllAutoEncoder_v2.m:171–221` decision tree; the Python equivalent |
| 05.6 | `the_two_stage_lifecycle.ipynb` | Stage 1 (unsupervised) → Stage 2 (supervised) — what changes between them and why |
| 05.7 | `batch_norm_state_synchronization.ipynb` | `cgg_updateState` vs PyTorch's automatic running-mean updates |

#### Module 06 — Loss Orchestration (companion to Milestone C)

| # | Notebook | Reference to |
|---|----------|--------------|
| 06.1 | `multi_task_losses_overview.ipynb` | ELBO + classification + confidence + offset/scale; how MATLAB weights them; the EMA prior normalization |
| 06.2 | `vae_and_the_elbo.ipynb` | KL divergence intuition, reparameterization trick, `cgg_lossELBO_v2` mathematics, the `cgg_samplingLayer` |
| 06.3 | `stochastic_vs_deterministic_placement.ipynb` | the two graph topologies; why Optimal uses Stochastic |
| 06.4 | `the_ema_prior_normalization_deep_dive.ipynb` | `cgg_getLossInformation` + `cgg_processLossComponent`; `PriorProportion=0.9` mechanics |
| 06.5 | `mil_softmax_pooling.ipynb` | Multiple Instance Learning intuition; why multi-axis softmax; `cgg_softmaxLayer` |
| 06.6 | `confidence_routing.ipynb` | Trial vs Task confidence; what each head outputs; `cgg_addTaskConfidenceToClassifier` |
| 06.7 | `the_confidence_pd_controller.ipynb` | **highest-risk port — full mathematical derivation, MATLAB code, Python implementation, parity test walkthrough** |
| 06.8 | `l2_inside_the_loss_kernel.ipynb` | why MATLAB applies L2 as `grad + L2Factor*param` and why PyTorch's `weight_decay` on Adam is NOT equivalent |
| 06.9 | `per_batch_prior_correction.ipynb` | the `WantBatchCorrection` flag — what it does, when to use it |
| 06.10 | `nan_masked_reconstruction.ipynb` | the two-layered NaN handling (input-side zero substitution + loss-side masking); why both are needed; how to verify the mask is correct; the silent-parity-loss trap |
| 06.11 | `single_total_loss_three_subnetworks.ipynb` | the gradient-flow topology: how one `total_loss.backward()` correctly distributes to all three subnetworks via autograd; why `Loss_Decoder` and `Loss_Classifier` in the orchestrator are intermediate sums, not gradient roots |
| 06.12 | `ema_prior_normalization_deep_dive.ipynb` | how the cross-component normalization works; why classification is the reference; first-iteration degeneracy; relationship between `RescaleLossEpoch`, `PriorProportion`, and `BatchFraction` |
| 06.13 | `sampling_layer_deterministic_at_inference.ipynb` | the `self.training`-branched sampling layer; why deterministic at eval time is the right behavior; how this differs from textbook VAE implementations; a hands-on demonstration showing same input → same classification across multiple inference passes |

#### Module 07 — Dynamic Curriculum (companion to Milestone C)

| # | Notebook | Reference to |
|---|----------|--------------|
| 07.1 | `curriculum_learning_intuition.ipynb` | why neural decoding benefits from staged training; the schedule families |
| 07.2 | `piecewise_linear_schedules.ipynb` | `cgg_calculateDynamicValue` — interpolation between (epoch, magnitude) waypoints |
| 07.3 | `load_parameters.ipynb` | `cgg_generateLoadParameters_v2` → Python `LoadParameters` class; how the Dataset reads live curriculum state |
| 07.4 | `loss_weights_curriculum.ipynb` | `cgg_generateLossWeights_v2`; KL annealing |
| 07.5 | `freeze_unfreeze_curriculum.ipynb` | `cgg_setFrozenNetwork_v2` ↔ PyTorch `requires_grad` management |
| 07.6 | `walkthrough_soft_three_stage_curriculum_shortened.ipynb` | end-to-end trace of the active Optimal curriculum |

#### Module 08 — Output & Analysis (companion to Milestone C/D)

| # | Notebook | Reference to |
|---|----------|--------------|
| 08.1 | `folder_hierarchy_generation.ipynb` | `cgg_generateEncoderSubFolders_v3` deterministic naming; the Python replication |
| 08.2 | `writing_mat_files_for_matlab.ipynb` | how to produce `.mat` files MATLAB analysis scripts can consume — nested structs, dtypes, cell vs numeric |
| 08.3 | `monitor_table_compatibility.ipynb` | what fields MATLAB monitors expect; how to write them from Python |
| 08.4 | `the_mat_round_trip_test.ipynb` | walkthrough of the T4 parity gate |
| 08.5 | `weights_and_biases_integration.ipynb` | W&B as the modern equivalent of the MATLAB monitor system |
| 08.6 | `running_matlab_analysis_on_python_output.ipynb` | hands-on: train in Python, aggregate with `DATA_cggAllNetworkEncoderResults`, plot with `FIGURE_cggPaperFigures` |

#### Module 09 — Production Deployment (companion to Milestone D)

| # | Notebook | Reference to |
|---|----------|--------------|
| 09.1 | `environment_detection.ipynb` | how `cgg_getBaseFolders` detects ACCRE/TEBA/local; the Python equivalent |
| 09.2 | `submitit_and_slurm.ipynb` | submitting jobs from Python; equivalent to MATLAB's SLURM dispatch model |
| 09.3 | `hydra_config_composition.ipynb` | replacing the MATLAB parameter switch with composable configs |
| 09.4 | `parameter_sweeps.ipynb` | replacing the 47-dim `SLURMPARAMETERS_cgg_runAutoEncoder_v2` sweep |
| 09.5 | `debugging_a_failing_run.ipynb` | troubleshooting cookbook: NaN losses, OOM, divergent training, parity-test failures |
| 09.6 | `extending_the_pipeline.ipynb` | how to add a new architecture, a new loss component, a new curriculum schedule, a new target task |

**Total: ~55 notebooks across 10 modules.**

#### Notebook execution + CI

- Notebooks are version-controlled in the repo as `.ipynb` with outputs stripped (use `nbstripout` git filter).
- A CI job runs `jupyter nbconvert --execute --to notebook` on all notebooks weekly to catch drift between notebook code and production code.
- Hidden solution cells use `nbgrader` so readers see exercises blank by default and reveal solutions via a button.
- A separate `notebooks/README.md` provides the curriculum map, prerequisite graph, and "I'm coming from X background, where do I start?" guidance.

#### Estimated authoring effort

| Module | # notebooks | Effort (weeks) |
|--------|-------------|----------------|
| 00 Orientation | 3 | 0.5 |
| 01 Python for MATLAB users | 8 | 1.5 |
| 02 NumPy & PyTorch basics | 8 | 1.5 |
| 03 Data pipeline | 6 | 1.0 |
| 04 Architecture | 8 | 1.7 |
| 05 Training loop | 7 | 1.5 |
| 06 Loss orchestration | 13 | 2.5 |
| 07 Dynamic curriculum | 6 | 1.0 |
| 08 Output & analysis | 6 | 1.0 |
| 09 Production deployment | 6 | 1.0 |
| **Total** | **~60** | **~13.2 weeks** |

This effort is in parallel with code milestones, not sequential. With AI authoring and reviewer feedback, the on-elapsed-calendar overhead is roughly +4–6 weeks beyond the code milestones (most notebook content is grounded in already-written production code).

### Milestone F — Reference Documentation (first-class deliverable, runs in parallel)

**Goal:** comprehensive code documentation so anyone landing in the repo can understand, use, navigate, debug, extend, and deploy the pipeline without needing the original author present. Documentation is a peer to the educational notebooks (Milestone E) and to the code itself.

The notebooks (Milestone E) are *pedagogical* — they take a reader through learning the codebase from scratch. This milestone produces *reference* documentation — what someone uses for lookup and navigation once they already understand the basics. The two complement each other.

**Authoring rule:** every public function/class/module must have a NumPy-style docstring before its code merges. Documentation drift is treated as a bug — a CI job builds the docs on every PR and fails if any public symbol lacks a docstring.

**Authoring cadence:** documentation grows alongside each code milestone. API docs are auto-generated from docstrings, so they appear as soon as code lands. Narrative and ADRs are written deliberately at decision time, not retroactively.

**Toolchain (two parallel builds):**
- **MkDocs Material** at `docs/narrative/` — narrative docs (quickstart, user guides, cookbook, concepts, glossary, ADRs, troubleshooting, contributing, deployment).
- **Sphinx + autodoc + autosummary + myst-parser + napoleon** at `docs/api/` — API reference auto-generated from NumPy-style docstrings.
- A unified build script (`scripts/build_docs.sh`) builds both and stitches them into a single site with a shared navigation bar. Hosted on GitHub Pages or ReadTheDocs.
- Both sites have versioned builds (`mike` for MkDocs, Sphinx's own versioning for API).

**Cross-reference policy (per user choice — minimal):** Python API docs read like a standard PyTorch library. MATLAB origin is only mentioned in:
- The project README
- The `docs/narrative/glossary.md` (a single one-shot reference table mapping MATLAB names to Python counterparts)
- The relevant ADR (where the MATLAB design influenced the Python choice)

Per-function docstrings do **not** include "MATLAB equivalent" sections. The Python pipeline is documented as a standalone library going forward.

**Docstring style:** NumPy-style throughout. Every public function/class has `Parameters`, `Returns`, `Raises`, `Examples`, and `Notes` sections as appropriate. Private helpers (leading underscore) need only a single-line summary.

**Documentation inventory:**

#### Narrative (MkDocs)

| Section | Contents |
|---------|----------|
| `index.md` | Landing page; project overview; current parity status; quick links |
| `quickstart.md` | Install + run a training in 10 minutes |
| `user_guide/running_a_training.md` | How to launch a single training run |
| `user_guide/parameter_sweeps.md` | Submitit / Ray Tune / Hydra sweeps |
| `user_guide/resuming_an_interrupted_run.md` | How resume works; what's saved, what's reinitialized |
| `user_guide/running_on_accre.md` | ACCRE-specific setup and quirks |
| `user_guide/inspecting_results.md` | Reading CM_Tables, running MATLAB analysis |
| `cookbook/add_a_new_architecture.md` | Register a new ModelName |
| `cookbook/add_a_new_loss_component.md` | Wire a new term into the multi-objective loss |
| `cookbook/add_a_new_curriculum_schedule.md` | Add a new DynamicParameterSet |
| `cookbook/add_a_new_target_task.md` | Wire a new behavioral target |
| `cookbook/debug_a_failing_run.md` | NaN losses, OOM, divergent training, parity failures |
| `cookbook/compare_two_sweep_configs.md` | Use `cgg_plotParameterSweep` against Python output |
| `concepts/the_training_lifecycle.md` | The two-stage state machine |
| `concepts/multi_objective_losses.md` | ELBO + classification + confidence + EMA priors |
| `concepts/dynamic_curriculum.md` | The schedule classes and how they feed the dataloader live |
| `concepts/single_session_batching.md` | Why one session per minibatch |
| `concepts/vae_sampling.md` | Reparameterization, Stochastic vs Deterministic |
| `concepts/the_confidence_pd_controller.md` | The most subtle component |
| `deployment/slurm_submission.md` | submitit + Hydra |
| `deployment/cluster_quickstart_accre.md` | Step-by-step on the actual cluster |
| `deployment/monitoring_a_running_job.md` | W&B + filesystem checks |
| `deployment/recovering_from_failure.md` | What to do when a sweep run dies |
| `glossary.md` | MATLAB ↔ Python name mapping (one-shot reference) |
| `troubleshooting.md` | Common errors with paste-able solutions |
| `contributing.md` | Pre-commit setup, test conventions, docstring requirements, ADR process |

#### Architecture Decision Records (ADRs in `docs/narrative/adrs/`)

Each ADR is a short markdown file (1–2 pages) with sections: **Context**, **Decision**, **Consequences**, **Alternatives Considered**. Numbered sequentially; once accepted, never deleted (superseded ADRs link to the new one).

| # | ADR | Decision being recorded |
|---|-----|-------------------------|
| 001 | `tiered_parity_not_bit_exact.md` | Why bit-exact parity is the wrong bar; the four-tier model |
| 002 | `pythonic_structure_over_matlab_mirror.md` | Why we don't mirror the MATLAB file layout |
| 003 | `adamw_for_l2_weight_decay.md` | Why AdamW instead of Adam + grad-side L2 |
| 004 | `single_session_batching.md` | Why minibatches contain trials from only one session |
| 005 | `no_optimizer_state_in_checkpoints.md` | Why we match MATLAB's intentional omission (file-size tradeoff) |
| 006 | `resume_reads_current_not_optimal.md` | How resume semantics differ from "best model" tracking |
| 007 | `mat_interop_surface.md` | Why CM_Table + stable-schema YAML and not full MATLAB output mirroring |
| 008 | `hydra_config_composition.md` | Why composable YAML instead of flat parameter files |
| 009 | `ema_prior_cadence_via_rescale_loss_epoch.md` | The three-mode cadence behavior |
| 010 | `augmentation_per_getitem.md` | Why augmentation re-randomizes every read, never cached |
| 011 | `validation_per_epoch_default.md` | Default cadence change vs MATLAB's per-iteration |
| 012 | `pre_flight_check_no_overwrite.md` | Abort-on-existing instead of auto-overwrite |
| 013 | `memory_probe_via_cuda_mem_get_info.md` | Replacement for `cgg_getMemoryInformation` |
| 014 | `single_gpu_default_accelerate_for_multi.md` | Why we don't replicate MATLAB's `canUseGPU` quirk |
| 015 | `two_doc_toolchains_mkdocs_plus_sphinx.md` | Why both, and what each is responsible for |
| 016 | `minimal_matlab_cross_referencing_in_api_docs.md` | Why Python docs read standalone |
| 017 | `nan_masked_reconstruction_loss.md` | Two-tensor input plumbing + masked reconstruction loss; why we don't penalize the decoder on removed-channel positions |
| 018 | `layer_block_order_dropout_before_norm.md` | Why `Transform → Dropout → Norm → Activation` is preserved even though unconventional |
| 019 | `single_total_loss_three_subnetworks.md` | Why all three subnetworks gradient-flow from one `Loss_Encoder` scalar |
| 020 | `confidence_loss_five_subtleties.md` | Multiplicative conjunction, ConfidenceDropout, prediction-toward-truth interpolation, stop-grad on historical EMA, BatchFraction-governed cadence |
| 021 | `ema_prior_normalized_to_classification.md` | Why classification's prior is the reference for cross-component normalization |
| 022 | `he_initialization_explicit.md` | Why `nn.init.kaiming_normal_` is set explicitly on FC layers |
| 023 | `augmentation_loss_auto_activated_by_topology.md` | Why offset/scale loss is gated by layer presence in the Decoder, not by config flag |
| 024 | `sampling_layer_deterministic_at_inference.md` | Why the VAE sampling layer returns `mu` deterministically in `eval` mode |
| 025+ | (additional ADRs added at decision-time during Milestones A–CC) | |

#### API reference (Sphinx)

Auto-generated from docstrings. Module tree:
- `docs/api/data.rst` — Dataset, samplers, stratification, normalization, augmentation, mat_files
- `docs/api/models.rst` — Encoder, decoder, classifier, layers, registry, stitching_fusion
- `docs/api/training.rst` — Lifecycle, loop, losses, schedules, monitoring, hardware, checkpoint
- `docs/api/interop.rst` — CM_Table format, folder hierarchy, network converter, parameter YAML
- `docs/api/sweeps.rst` — Parameter space, launchers

#### Subpackage READMEs

Each top-level subpackage gets a short `README.md` (~1 page) explaining:
- What this subpackage is for
- The 2–3 most important entry-point functions / classes
- Links to the relevant ADRs
- Pointers to the corresponding cookbook entries

Example: `src/neural_data_decoding/training/losses/README.md` would link to ADRs 003, 005, 009 and to the `concepts/multi_objective_losses.md` page.

#### Estimated effort

| Track | Effort (weeks) | Calendar overhead |
|-------|---------------|-------------------|
| API docs (auto-generated from docstrings) | ~0.5 (just CI setup) | included in Milestone 0 |
| Docstring authoring | ~0.1 per code module (~3 weeks total) | woven into A/B/C/CC code work |
| Narrative pages (~25 pages) | ~2 weeks | +1–2 weeks elapsed beyond code |
| ADRs (~16 baseline, more as decisions arise) | ~1 week | written at decision-time |
| **Total** | **~6 weeks effort** | **~+2 weeks elapsed calendar** |

#### CI gates

- `mkdocs build --strict` on every PR — fails on broken links or missing pages.
- `sphinx-build -W` on every PR — fails on missing docstrings or broken refs.
- `interrogate` (docstring coverage tool) configured to require 100% coverage on public symbols.
- Notebook execution from Milestone E shares this CI (`nbconvert --execute`) — keeps notebook code in sync with library code.

---

## Critical Implementation Notes (must be in implementer's face)

These are high-risk items where silent parity loss is most likely if implemented carelessly. Each must be implemented as specified — the implementer should resist "improvements" without first establishing that parity holds:

1. **Two-stage training lifecycle** — `cgg_trainAllAutoEncoder_v2` calls `cgg_trainNetwork` twice. **Stage 1 explicitly does not pass a Classifier argument** (line 232 of `cgg_trainAllAutoEncoder_v2.m`); `cgg_trainNetwork` then sets `Classifier=[]` and `HasClassifier=false`, completely skipping all classification forward/backward code. Stage 1 is purely unsupervised reconstruction. Stage 2 then builds the Classifier via `cgg_constructClassifierArchitecture`, loads optimal autoencoder weights from Stage 1, and runs supervised training. Current Optimal has `NumEpochsAutoEncoder=0` so Stage 1 is degenerate (no actual training), but the orchestrator MUST handle the general case where Stage 1 runs. The Python lifecycle should reflect this: Stage 1 instantiates only the encoder/decoder; Stage 2 instantiates the classifier on top of the loaded encoder.

2. **Checkpoint state machine (resume from interruption)** — checkpointing exists strictly so that an interrupted training run can pick up from where it left off (or the most recent save point). Not for tracking "best model" — that's a separate concern. The Python implementation should mirror MATLAB's "resume reads `CurrentIteration.mat` and `Encoder-Current.mat`" semantics: **resume always uses Current state, never Optimal**. (Optimal checkpoints are a separate high-water-mark snapshot used for downstream evaluation, not for resume.) See `cgg_getIterationInformation.m` for confirmation.

3. **Optimizer state is NOT saved by MATLAB** — `cgg_saveIterationInformation.m` commented out the line that would save `OptimizerVariables.mat`. On resume, the optimizer is reinitialized. The user chose this deliberately to reduce file sizes. Match this behavior in the Python pipeline: **save model weights + epoch + iteration + monitor state, but NOT optimizer state**. This means resume after an interruption will produce slightly different trajectories than uninterrupted training — parity tests for "interrupt + resume" must account for this small expected drift.

4. **Freeze v1 is legacy — do NOT port it; v2 IS ported.** Two freeze mechanisms exist in MATLAB:
   - **v1 (legacy, drop)**: the `Freeze_cfg` parameter struct + `cgg_setFrozenNetwork.m` (no version suffix). Do NOT port these.
   - **v2 (active, port)**: the `FreezeParameters` schedule class from `cgg_generateFreezeParameters` + `cgg_setFrozenNetwork_v2.m`. **This IS the active path** — `cgg_setFrozenNetwork_v2` reads the schedule's current per-epoch magnitude and applies it to actually freeze/unfreeze each network's layers at the start of each epoch. Port this as `FreezeSchedule` class + a `apply_freeze_schedule(net, schedule, epoch)` function.

   The Python implementation should use only the v2-style schedule-driven freeze; never reference `Freeze_cfg`.

5. **L2 weight decay — use the modern Python approach.** MATLAB applies L2 as `grad + L2Factor*param` after `dlgradient`, which is correct for SGD-like optimizers but yields different behavior than `torch.optim.Adam(weight_decay=...)` because Adam's adaptive moments couple with weight decay. The standard modern solution is **`torch.optim.AdamW`** (decoupled weight decay). For non-Adam optimizers, `torch.optim.SGD(weight_decay=...)` is mathematically equivalent to MATLAB's L2-in-loss. Implementation: use `AdamW` by default for the ADAM-equivalent path; for SGDM use `SGD(weight_decay=...)`; document both in a notebook so the user understands why `Adam(weight_decay=...)` is NOT what they want. (Optional escape hatch: a gradient-hook-based L2 that works with any optimizer, for users who specifically want the MATLAB grad-side semantics.)

6. **EMA prior normalization is controlled by `RescaleLossEpoch`** — `cgg_getLossInformation` keeps running EMA of each loss component magnitude (with `PriorProportion=0.9` meaning new value contributes `(1-0.9)=0.1` per update). The **update cadence** is controlled by `cfg_Encoder.RescaleLossEpoch`:
   - `RescaleLossEpoch == 0` (Optimal): update **every iteration**
   - `RescaleLossEpoch == 1`: update once **per epoch**
   - `RescaleLossEpoch > 1`: update every `N` epochs

   Implement as a callback whose `should_update(epoch, iteration)` reads this config.

7. **Augmentation re-randomization per `__getitem__`** — the augmented Dataset must re-roll noise on every read. MATLAB does this implicitly via `fileDatastore` re-invoking the read function each access. PyTorch users sometimes cache augmented tensors — DO NOT.

8. **LoadSchedule live read by Dataset** — the Dataset's augmentation magnitudes are controlled by the *current epoch's* `LoadSchedule` (Python name for `LoadParameters`). The Dataset needs a reference to the LoadSchedule object and reads `load_schedule.current_STDChannelOffset` etc. on every `__getitem__`. NOT set once per epoch.

9. **Single-session sampler — every minibatch is from ONE session.** This is the **opposite** of what a naive reading might suggest. `cgg_procAllSessionMiniBatchTable.m` partitions trials by session and ensures each minibatch contains trials from **only one session**. The reason: future per-session stitching/fusion layers will require all trials in a minibatch to come from the same recording session so their session-specific transform can be applied uniformly. Implement `SingleSessionBatchSampler` accordingly; the unit test in `tests/unit/test_single_session_sampler.py` must verify that every emitted minibatch contains trials from exactly one session.

10. **MIL pooling is multi-axis softmax across Space-Channel-Time** — `cgg_softmaxLayer.m` computes softmax across these axes simultaneously. Match this. (User may try alternative MIL formulations later, but for now this is the implementation.)

11. **Confidence PD controller** — `cgg_lossConfidence.m` interpolates predictions toward ground truth based on self-reported confidence, then adds a budget regularizer pushing confidence toward 1. **See Critical Note #29 for the full breakdown** (five subtleties: multiplicative conjunction, internal ConfidenceDropout, prediction-toward-truth interpolation, stop-gradient on historical EMA, BatchFraction-governed cadence). The exact mathematical form must be preserved — golden-vector tests are mandatory at the Milestone C boundary. It's not a generic "uncertainty estimate".

12. **`WantBatchCorrection` corrects the confidence loss** — when true, makes up for the small loss that comes from using *total* (running) confidence (i.e., the EMA of current-batch + previous-batches' confidence) instead of *single-batch* confidence. Affects the confidence loss kernel only; not classification. Implement as a flag inside the confidence loss component.

13. **Sampling-layer placement (Stochastic vs Deterministic)** — In MATLAB this is two different graph topologies because of how Encoder/Decoder/Classifier are split into separate `dlnetwork`s (the Classifier always takes Encoder output). In Python, with the cleaner option of composing the sampling step inside a single `nn.Module`, both behaviors can typically be expressed via a single layer with a placement flag. If the Python version preserves the same input/output behavior as MATLAB (verified by T2 forward-pass parity), the implementation detail can be more Pythonic. Document in a notebook why MATLAB has two topologies and Python doesn't need to.

14. **Architecture string registry** — there are 47 named `ModelName` strings in `PARAMETERS_cgg_constructNetworkArchitecture.m` and 9 named `ClassifierName` strings in `cgg_selectClassifier.m`. Implement both as Python registries (decorator-based or dict-based) in `models/registry.py` so any of these architecture strings can be selected via config. Milestones A/B/C populate only the ones they need; Milestone CC fills in the rest of the architecture registry. The classifier registry should be fully populated by end of Milestone C (all 9 classifier variants registered, even if only 1–2 are exercised by Optimal).

15. **Folder hierarchy must match exactly** — `cgg_generateEncoderSubFolders_v3.m` produces a deterministic deeply-nested path from hyperparameters. If Python uses different folder conventions, the MATLAB results aggregator can't find Python output. See `src/neural_data_decoding/interop/folder_hierarchy.py`.

16. **CM_Table is the primary MATLAB-interop output** — the `.mat` file with the per-trial confusion-matrix telemetry, written by `cgg_saveValidationCMTable.m` and consumed by `DATA_cggAllNetworkEncoderResults`. Inspect `cgg_generateBlankCMTable.m` to see the exact field schema (NumWindows, DataNumber, TrueValue, plus per-classifier prediction columns and confidence columns). Other `.mat` artifacts (network weights, etc.) are only needed for parity-comparison fixtures, not for the analysis pipeline.

17. **HDF5 `.mat` files** — MATLAB v7.3+ saves as HDF5. `scipy.io.loadmat` fails on these. Use `mat73` or `h5py` for v7.3, `scipy.io.loadmat` for older. Auto-detect via 128-byte header in `data/mat_files.py`.

18. **Hardware-aware accumulation table** — `PARAMETERS_cgg_runAutoEncoder.m:296–338` rewrites the AccumulationInformation table three different ways based on `StitchingAndFusionLayer`, `ClassifierHiddenSize`/`HiddenSizes(1)`, and `ParameterSetName='Synthetic'`. Replicate all three branches.

19. **Memory probe** — replace MATLAB's `cgg_getMemoryInformation` (which was used to predict imminent OOM errors) with a Python equivalent using `torch.cuda.mem_get_info()` for proactive headroom checking, plus a try/except around `dlfeval`-equivalent calls to catch any OOM and report it cleanly. Don't replicate MATLAB's 6×-per-sub-batch probing — call it once per minibatch.

20. **Hardware detection — single-GPU default, multi-GPU readiness via `accelerate`** — the MATLAB code disabled `parfor` on a single GPU because MATLAB's stock parallelism makes single-GPU slower. Python doesn't have this oddity. Just detect the hardware, use single-GPU by default, and provide a clear `accelerate`-based path for multi-GPU when the user has more than one device. Don't replicate the `canUseGPU` flip behavior — it's a MATLAB artifact.

21. **Validation timing — make Python-idiomatic** — MATLAB runs validation every `ValidationFrequency` iterations (default 25) because per-iteration scheduling was easier in MATLAB. In Python, default to **per-epoch validation** (more idiomatic) but expose an option `validate_every_n_iterations` for users who want the MATLAB cadence. Document the default change clearly.

22. **Pre-flight existing-network check** — Before training, scan the resolved-config output directory for existing `*-Current.mat`/`*-Optimal.mat`/`CurrentIteration.mat` files. If any exist, **abort with an informative error** listing the files. User manually deletes if intended. This prevents silent overwrites of expensive training runs.

23. **Min-trial-class filter** — `cgg_getDataIndexToRemoveFromDataStore` runs once after K-fold assignment to drop trials whose target class is < `ClassLowerCount` (default 20). Must happen *after* the partition is loaded, *before* the train/val/test subset.

24. **CheckVararginPairs idiom → dataclasses or pydantic** — every MATLAB function uses `CheckVararginPairs` for varargin parsing. Replace with `@dataclass` config objects (or pydantic models) loaded from Hydra-composed YAML. Single consistent pattern across the codebase.

25. **Stable YAML field schema for parameter-sweep plotting** — `cgg_plotParameterSweep.m` reads the `EncodingParameters.yaml` file from each run in a sweep and scans field-by-field to identify which hyperparameters varied. **Every run's YAML must have the same set of fields**, even when individual values are defaults or zeros. MATLAB guarantees this via `cgg_setBaselineDynamicParameters` which snapshots all dynamic-parameter fields up-front into a `BaselineDynamicParameters` struct that always exists in the saved YAML. Python equivalent: build a `dump_full_schema_yaml(cfg)` helper that emits **every** config field (including unused ones) so the sweep YAMLs are field-symmetric across the parameter space. The implementer should inspect a MATLAB-generated `EncodingParameters.yaml` to mirror the exact field set.

26. **Docstrings are mandatory and CI-enforced** — every public function, class, and module must carry a NumPy-style docstring before its code merges. The Sphinx build (`docs/api/`) is wired into CI with `-W` so missing docstrings or broken references fail the PR. `interrogate` is configured to require 100% docstring coverage on public symbols. Private helpers (leading underscore) need only a single-line summary. This is treated with the same rigor as test coverage — undocumented code is broken code. See Milestone F for the full documentation deliverable.

27. **Layer-block order is `Transform → Dropout → Norm → Activation`** — not the conventional `Transform → Norm → Activation → Dropout`. See `cgg_generateSimpleBlock.m:116–121`. This order is **intentional** even though it's non-standard. Implementer should preserve it exactly; "fixing" it to the conventional order is a silent training-dynamics change. Document in an ADR.

28. **All three networks backprop from the same total loss** — `cgg_lossComponents.m:491–504`: `Gradients.{Encoder,Decoder,Classifier} = dlgradient(LossInformation.Loss_Encoder, .Learnables)`. Inside the loss orchestrator:
    - `Loss_Decoder = Loss_Reconstruction + Loss_KL + Loss_OffsetAndScale`  *(intermediate sum, NOT the gradient root)*
    - `Loss_Classifier = Loss_Classification + Loss_Confidence`  *(intermediate sum, NOT the gradient root)*
    - `Loss_Encoder = Loss_Decoder + Loss_Classifier`  *(the actual gradient root)*

    The names "Loss_Decoder" and "Loss_Classifier" are misleading — they are **telemetry** intermediate sums, not subnetwork-specific gradients. All three networks gradient-flow from `Loss_Encoder`. The implication: in PyTorch, do `total_loss.backward()` once over `model.parameters()` (concatenated encoder+decoder+classifier params) — autograd handles per-parameter flow. Do **not** call `.backward()` separately on `Loss_Decoder` and `Loss_Classifier`; that would double-count gradients.

29. **Confidence loss internals — five subtleties packed into one kernel** (`cgg_lossConfidence.m`). The plan's earlier note about "PD controller" is correct in spirit but undersells the mechanics. The full kernel does:

    **(a) Multiplicative conjunction**: `TotalConfidence = TaskConfidence .* TrialConfidence` (element-wise multiply, line 53). When both heads are active.

    **(b) Internal `ConfidenceDropout` (default 0.5)**: separate from network dropout. Randomly resets confidence to 1 for ~50% of trials before prediction interpolation (lines 39–41, 49–51). The *dropped* version is used for prediction interpolation; the *un-dropped* version is used for the budget regularizer. Two parallel paths.

    **(c) Prediction interpolation toward ground truth** (line 75, "Eq. 2"): `Y = TotalConfidence_Dropped .* Y + (1 - TotalConfidence_Dropped) .* T`. Low-confidence trials have their classification target pulled toward the truth, so the classifier loss for those trials approaches zero. The corresponding confidence-budget regularizer (push confidence toward 1) prevents the trivial "always predict 0 confidence → no loss" solution. **This is the mechanism**, not a postprocessing step. Losing it would entirely change training behavior.

    **(d) Stop-gradient on historical confidence** (line 115): `cgg_extractData(HistoricalDatasetConfidence)` detaches from autograd. PyTorch equivalent: `.detach()`. Only the current batch's contribution flows gradient through the EMA.

    **(e) EMA cadence governed by `BatchFraction`** (lines 116, 121): `UpdatedDatasetConfidence = Historical * (1 - gamma) + BatchMean * gamma` where `gamma = BatchFraction`. So the EMA update rate is proportional to how much of the dataset the current batch represents — NOT a fixed coefficient. Distinct from the `PriorProportion=0.9` used for multi-objective loss normalization (#31).

    Mandatory T2 golden-vector tests for each of these subtleties (see Risks table).

30. **EMA prior normalization uses Classification's prior as the reference** — `cgg_getLossInformation.m:128–134`. The `Rescale_Value` for normalizing every loss component defaults to `Prior_Loss_Classification`, falling back to `Prior_Loss_Reconstruction` if classification is inactive. All five loss components (Reconstruction / KL / Classification / OffsetAndScale / Confidence) get normalized relative to this single reference. The first iteration's prior is degenerate (loss/loss = 1) — the EMA only becomes meaningful from iteration 2+. Implementer should match: initialize all priors with the first batch's actual loss values, then EMA-update from there.

31. **He initialization is explicit on `fullyConnectedLayer`** — `cgg_generateSimpleBlock.m:63` and `cgg_selectBottleNeck.m:62`: `"WeightsInitializer", "he"`. PyTorch's `nn.Linear` default is Kaiming-uniform with `fan_in` and `a=sqrt(5)`, which is mathematically different from MATLAB's `'he'` initializer (Kaiming-normal with `fan_in`, `nonlinearity='relu'`). Implementer should explicitly set `nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')` on every FC layer to match MATLAB. GRU/LSTM layers in MATLAB use default initialization (Glorot) — PyTorch's default also differs but is closer. Run a single-step parity check with random seeds matched to verify.

32. **Augmentation loss is auto-activated by graph topology, not config** — `cgg_lossComponents.m:368–375`: `if any(contains({Decoder.Layers(:).Name}, "reshape_offset_Augment")) || any(contains({Decoder.Layers(:).Name}, "reshape_scale_Augment"))`. So `cgg_lossOffsetAndScale` is invoked only when the Decoder contains specifically-named augmentation layers. Python equivalent: check for the relevant `nn.Module` instance in the Decoder (e.g., `isinstance(child, LearnableOffsetScale)`) before invoking the loss. Optimal config doesn't activate this (`WeightOffsetAndScale=0` in Optimal), but the auto-detection pattern is what Milestone CC will need.

33. **Per-area reconstruction loss is detached for telemetry only** — `cgg_lossComponents.m:115`: `cgg_extractData(Loss_Reconstruction_PerArea)`. The per-area variant is logged for monitoring but does NOT contribute to the gradient. The total reconstruction loss (sum across areas) is what backprops. Don't mistakenly add the per-area losses to the gradient path.

34. **`forward` vs `predict` distinction maps to `.train()` / `.eval()` plus `.no_grad()`** — `cgg_lossComponents.m:334-337, 356-360, 394-398`: `wantPredict=true` calls `predict()` (no gradient computation, no state update on BatchNorm running stats); `wantPredict=false` calls `forward()` (gradient + BN state update). In PyTorch:
    - Training: `model.train()` + autograd active → equivalent to MATLAB `forward()`
    - Validation/testing: `model.eval()` + `with torch.no_grad()` → equivalent to MATLAB `predict()`

    Running BatchNorm stats only update during Training, never during Validation. Standard PyTorch idiom; just don't forget to wrap validation passes in `model.eval()` + `torch.no_grad()`.

35. **Sampling layer is deterministic in `predict()`, probabilistic in `forward()`** — `cgg_samplingLayer.m`. The `forward` method (line 94, used during training) draws `epsilon = randn(...)` and computes `Z = epsilon * sigma + mu` (proper reparameterization sampling). The `predict` method (line 44, used during validation/testing) computes `epsilon = randn(...); epsilon = epsilon * 0;` followed by `Z = epsilon * sigma + mu` — which simplifies to `Z = mu`, **deterministic**, returns just the mean.

    This is **intentional and important**: it means the trained model produces consistent outputs for identical inputs at inference time, while still using stochastic gradient flow during training. Without this distinction, the same input at inference would yield different classification probabilities on each call.

    PyTorch implementation: the sampling layer must check `self.training`:
    ```python
    class SamplingLayer(nn.Module):
        def forward(self, x):
            mu, logvar = x.chunk(2, dim=self.channel_dim)
            if self.training:
                eps = torch.randn_like(mu)
                return mu + eps * (0.5 * logvar).exp(), mu, logvar
            else:
                return mu, mu, logvar  # deterministic at eval time
    ```
    This is **not the standard reparameterization-trick implementation** — most PyTorch VAE tutorials sample in both modes. The implementer needs to deliberately add the training-flag branch.

36. **Confidence heads emit sequence outputs; the last time-step is used** — `cgg_lossConfidence.m:37, 47, 63` calls `cgg_getLastSequenceValue(confidence)` to extract `confidence[:, :, -1]` (last time step). Python: `confidence[..., -1]` along the time axis. Standard practice but easy to overlook.

37. **`'SoftSign'` activation name actually instantiates a `softplusLayer`** — `cgg_generateSimpleBlock.m:88–90`. This is a naming bug in MATLAB (the names don't match the layers). Do NOT propagate this confusion. In Python config, use the correct name (`'softplus'`); document the MATLAB-name mapping in the glossary. This is not in any active parameter set; only a curiosity from the supported-but-unused architecture variants.

38. **Removed-channel NaN handling is two-layered — don't drop either layer.** Channels removed during preprocessing are represented as `NaN` in the on-disk `.mat` files. The pipeline handles these in two places that must both be ported:

   **(a) Input pathway — replace NaN with 0:**
   The encoder's input layer applies a normalization function `cgg_setNaNToValue(x, 0)` (see `cgg_constructNetworkArchitecture.m:127–129`), converting NaN entries to 0 *before* the encoder forward pass. This is correct because the data was Z-scored upstream, so 0 is the population mean — a neutral substitute that does not bias the encoder. Python: apply this transform inside the Dataset's `__getitem__` (or via a `nn.Module` at the encoder input) before any augmentation that depends on finite values.

   **(b) Reconstruction loss — mask out NaN positions:**
   Both `cgg_lossELBO_v2.m` (MSE) and `cgg_lossELBO_MAE.m` (MAE) build `Mask_NaN = ~isnan(T)` from the **original target** `T` (which still contains NaN for removed channels — the NaN-to-0 substitution happens only on the encoder input path) and pass it as `Mask=Mask_NaN` to MATLAB's `l2loss` / `l1loss`. This means the decoder is **not penalized** for whatever it predicts at removed-channel positions — those positions contribute zero to the loss.

   The per-area reconstruction loss (used for telemetry) applies the same mask slice per channel/area.

   PyTorch implementation: PyTorch has no built-in masked `l2loss` / `l1loss`. Implement as:
   ```python
   def masked_mse(y_pred, y_target):
       mask = ~torch.isnan(y_target)
       diff = (y_pred - y_target) * mask        # masked positions contribute 0
       return 0.5 * (diff ** 2).sum() / mask.sum().clamp(min=1)
   ```
   (For MAE, use `diff.abs()` and drop the `0.5` and `**2`.) **Verify against the MATLAB output on a synthetic batch with known-NaN positions before trusting the implementation** — it is the single highest-risk silent-parity-loss point in the reconstruction path.

   Note: the *input* `X` going into the encoder has NaN already zeroed (by the input-layer normalization function). The *target* `T` for reconstruction loss is the **pre-zero-substitution** original data — so the loss kernel sees NaN positions and masks them. The Python implementation must preserve this distinction: pass two versions of the input through — the NaN-zeroed one to the encoder, the NaN-preserving one to the loss as target. The MATLAB code arranges this implicitly because the input-layer normalization only affects the encoder's view, not the target tensor held in the dataloader minibatch.

---

## Verification Strategy

Per milestone, the verification gates are:

All `pytest` commands run from `neural_data_decoding/`.

| Gate | What | Tools | When |
|------|------|-------|------|
| **G1 — Stratification parity** | Strata for every trial match MATLAB | `pytest tests/parity/test_stratification_parity.py` | Milestone 0 onwards |
| **G2 — Augmentation parity** | Given seed, per-trial augmented output matches MATLAB within 1e-6 | `pytest tests/parity/test_dataset_augmentation.py` | Milestone 0 |
| **G3 — Single-step forward parity** | Forward pass on identical weights+input matches per loss component within 1e-5 | `pytest tests/parity/test_loss_forward.py` | Milestone B (basic), Milestone C (full) |
| **G4 — Dynamic schedule parity** | Each schedule's per-epoch value matches MATLAB exactly (deterministic) | `pytest tests/parity/test_dynamic_schedules.py` | Milestone C |
| **G5 — Checkpoint resume** | Train, kill mid-epoch, resume, verify weights/optimizer state continuity | `pytest tests/unit/test_checkpoint_resume.py` | Milestone A onwards |
| **G6 — Convergence parity** | 5-10 seed runs both sides; mean validation accuracy curves overlap within paired-bootstrap 95% CI | `pytest tests/parity/test_end_to_end_milestone_*.py` | End of each Milestone A/B/C |
| **G7 — .mat round-trip** | Python writes → MATLAB loads → MATLAB analysis script runs without error and produces same numeric aggregate | `matlab -batch ...` from pytest | Milestone A onwards |
| **G8 — Visual dashboard parity** | MATLAB `FIGURE_cggAllNetworkEncoderResults` and `FIGURE_SFN_cggAllNetworkEncoderResults` produce visually equivalent plots on Python output | manual visual diff | Milestone C |

**CI cadence:** G1, G2, G3, G4, G5 run on every PR. G6 runs nightly (long-running). G7 and G8 are manual gates at milestone boundaries.

---

## Critical Files (New & Reference)

**All new files created in `neural_data_decoding/`** — nothing else in the repo is modified.

**Key MATLAB files to mirror (reference, don't modify):**
- `Processing_Functions_cgg/Encoder Functions/cgg_runAutoEncoder.m`
- `Processing_Functions_cgg/Decoder Functions/cgg_procAutoEncoder.m`
- `Processing_Functions_cgg/Encoder Functions/cgg_trainAllAutoEncoder_v2.m` — two-stage state machine
- `Processing_Functions_cgg/ANN Functions/cgg_trainNetwork.m` — single-stage loop
- `Processing_Functions_cgg/ANN Functions/cgg_lossComponents.m` — loss orchestrator
- `Processing_Functions_cgg/ANN Functions/cgg_constructClassifierArchitecture.m` — multi-head + confidence + MIL classifier builder
- `Processing_Functions_cgg/Encoder Functions/cgg_selectClassifier.m` — classifier-name string dispatcher (9 named variants — must be ported as config options)
- `Processing_Functions_cgg/ANN Functions/Training Functions/cgg_procGradientAggregation.m`
- `Processing_Functions_cgg/ANN Functions/Training Functions/cgg_procAllSessionMiniBatchTable.m` — single-session batching reference
- `Processing_Functions_cgg/ANN Functions/Training Functions/cgg_getIterationInformation.m` — resume reads `CurrentIteration.mat`, not Optimal
- `Processing_Functions_cgg/ANN Functions/Training Functions/cgg_saveIterationInformation.m` — confirms optimizer state is intentionally NOT saved
- `Processing_Functions_cgg/Classification Functions/cgg_generateBlankCMTable.m` — schema for the primary `.mat` output
- `Processing_Functions_cgg/Decoder Functions/cgg_getKFoldPartitions.m` + helpers — stratification reference
- `Processing_Functions_cgg/Parameters/PARAMETERS_OPTIMAL_cgg_runAutoEncoder_v3.m` — Milestone C target spec
- `Processing_Functions_cgg/Parameters/PARAMETERS_cgg_constructNetworkArchitecture.m` — full ModelName enum (Milestone CC)
- `Processing_Functions_cgg/Parameters/PARAMETERS_cgg_selectDynamicParameters.m` — curriculum schedules
- `Processing_Functions_cgg/Parameters/SLURMPARAMETERS_cgg_runAutoEncoder_v2.m` — 47-dim sweep harness
- `Processing_Functions_cgg/Parameters/PARAMETERS_cgg_constructStitchingAndFusionNetwork.m` — S&F option-set definitions (Milestone CC)
- `Processing_Functions_cgg/Figures/FIGURE_cggAllNetworkEncoderResults.m` and `FIGURE_SFN_cggAllNetworkEncoderResults.m` — the actual plotting scripts that consume Python output
- `Processing_Functions_cgg/All Data Processing/DATA_cggAllNetworkEncoderResults.m` — aggregator that runs between training and plotting

**Reference docs:**
- `Codebase_Documentation.md` — Phase 2 has the full data-pipeline contract; Phase 4 has the loss-orchestration details
- `Execution_Path_Map.md` — active path + Supported-but-Unused classification

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Confidence PD controller mathematically diverges from MATLAB | Medium | High (silent training-quality loss) | Mandatory T2 golden-vector test at Milestone C boundary |
| Reconstruction loss penalizes NaN-masked positions (forgotten or wrong mask) | High | High (silent training-quality loss — the decoder will be pushed to predict 0 at removed-channel positions, biasing the latent space) | Critical Note #38. Dedicated T2 parity test: build a fixture batch with hand-placed NaN channels, compute MATLAB-side `cgg_lossELBO_v2(Y, T, mu, logSigmaSq)` once, then assert the Python `masked_mse` produces the same value within 1e-5. Apply same test for MAE. Test both total and per-area reconstruction loss. |
| Layer-block order "fixed" to conventional `Transform → Norm → Activation → Dropout` instead of MATLAB's `Transform → Dropout → Norm → Activation` | Medium | Medium (silent training-dynamics change) | Critical Note #27; ADR 018; documented in `04_architecture/` notebook. Code reviewer must check block-build code against MATLAB's exact order. |
| Sampling layer left probabilistic at inference (forgetting the `self.training` branch) | High | High (classification of identical inputs gives different answers across calls; breaks reproducibility of test-set evaluation) | Critical Note #35; ADR 024; notebook 06.13. Unit test: instantiate sampling layer, call twice on identical input in `.eval()` mode, assert outputs are bit-identical. |
| Confidence loss missing one of its five subtleties (multiplicative conjunction / dropout / interpolation / stop-grad / cadence) | High | High (training behavior diverges silently; the budget regularizer no longer balances the loss attenuation, so confidence collapses to 0 or 1) | Critical Note #29; ADR 020. **Mandatory** golden-vector parity tests at Milestone C boundary for each of the five subtleties individually, plus an end-to-end one. |
| Backprop called separately on `Loss_Decoder` / `Loss_Classifier` instead of `Loss_Encoder` | Medium | High (gradient double-counting or under-counting depending on which `.backward()` is called) | Critical Note #28; ADR 019; notebook 06.11. Code reviewer must verify only one `.backward()` call on the single total scalar. |
| He initialization not explicitly applied | Low | Low–Medium (slower convergence in early epochs; not catastrophic) | Critical Note #31; ADR 022; notebook 04.8. T2 parity check: load matched MATLAB weights, verify forward-pass output identical. |
| EMA prior normalization implemented wrong (esp. `RescaleLossEpoch` cadence) | Medium | High (training instability) | Snapshot MATLAB's EMA state mid-training, replay in Python, compare. Test all three cadence modes (0, 1, >1). |
| `CM_Table` `.mat` schema mismatch | High | Medium (MATLAB analysis scripts crash on Python output) | T4 round-trip test starting at Milestone A; inspect `cgg_generateBlankCMTable.m` output as the reference schema |
| MATLAB v7.3 `.mat` files break `scipy.io.loadmat` | High | Medium (data won't load) | `data/mat_files.py` auto-detects format; tested on real fixture files in Milestone 0 |
| Single-session sampler mis-implemented as cross-session | High | High (training dynamics differ; future stitching layers break) | Unit test: emit 100 minibatches from a 3-session toy dataset; assert every minibatch has exactly one session |
| LoadSchedule not properly threaded to Dataset | Medium | High (curriculum doesn't actually affect augmentation) | Integration test: snapshot Dataset output across epochs, verify augmentation magnitudes change |
| Folder hierarchy mismatch | High | Medium (MATLAB aggregator can't find Python output) | Test runs MATLAB `DATA_cggAllNetworkEncoderResults` against Python output directly |
| Stage 1 + Stage 2 lifecycle bugs only manifest with non-zero NumEpochsAutoEncoder | Low (current Optimal has 0) | High (when activated later, silent regression) | Build a separate test config with `NumEpochsAutoEncoder>0` and run it as part of Milestone C |
| Interrupt + resume gives different trajectory than uninterrupted (optimizer reinitialized) | Expected (matches MATLAB) | Low | Parity tests for "interrupt + resume" use loose tolerances; document the expected drift |
| Pre-flight check accidentally aborts on legitimate re-runs | Medium | Low (annoying) | Provide a `--force` flag with a clear warning; document the workflow |
| AdamW vs Adam+L2 confusion when users port MATLAB hyperparameters | Medium | Medium (silently different training dynamics) | Notebook 06.8 explicitly walks through this; default config uses AdamW |
| Hardware-aware accumulation table mis-applied | Low | Medium (OOM on certain configs) | Snapshot the MATLAB-applied `MaxBatchSize` values; assert Python applies same values for matching configs |
| OOM probe causes overhead in training loop | Low | Low | Probe once per minibatch (not 6× as MATLAB does); use try/except as backup |

---

## Timeline Estimate

| Milestone | Calendar weeks (AI-implementer, single-developer review cycle) |
|-----------|--------|
| 0 — Foundation (includes docs CI setup) | 1 week |
| A — Logistic tracer | 2 weeks |
| B — GRU+Classifier | 2 weeks |
| C — Full Optimal | 4 weeks |
| CC — Extra-credit feature implementation | 4–5 weeks |
| D — Cluster/SLURM | 1 week |
| E — Educational curriculum (~60 notebooks) | parallel, +4–6 weeks elapsed (~13.2 weeks of effort) |
| F — Reference documentation (narrative + API + ADRs) | parallel, +2 weeks elapsed (~6 weeks of effort) |
| **Total** | **~20–23 weeks calendar** |

Risk-adjusted: 26–30 weeks accounting for parity-debug cycles (particularly Milestone C's PD controller and EMA priors), the Gemini stitching implementation complexity in Milestone CC, and revision rounds for notebooks and documentation based on reviewer feedback.

Both the curriculum (E) and the reference documentation (F) are first-class deliverables — their calendar contribution is non-trivial and should not be cut to compress the schedule. A user landing in the repo with no Python background must be able to become productive by reading and doing the notebooks (E); a user already comfortable with Python must be able to use the pipeline as a library and extend it by reading the reference docs (F). Both are explicit success criteria for this plan.
