# Educational Curriculum — `neural_data_decoding`

Hands-on Jupyter notebooks that take a MATLAB-native programmer to
expert-level Python/PyTorch fluency on this specific pipeline. The
curriculum is structured so every concept the production code uses
is teachable from these notebooks — if a feature can't be traced from
"first principles" to "working in production code" through the
curriculum, the curriculum has a gap (open an issue).

The notebooks are an **equal-weight deliverable** to the production
code, not documentation that gets written after the fact.

## Quickstart

```bash
# From the repo root, with the project's venv activated:
source .venv/bin/activate

# One-time setup: activate the nbstripout git filter so notebook outputs
# stay out of commits (see "Workflow conventions" below).
nbstripout --install --attributes .gitattributes
```

Then open any `.ipynb` under this directory. **Notebook
[00.1 welcome](00_orientation/00.1_welcome.ipynb#Section-2-—-How-to-actually-run-this-notebook)
walks through three editor options** — VS Code, JupyterLab, Classic
Jupyter — including how to attach the venv's Python as the kernel.

The production code that the curriculum references is in
[`src/neural_data_decoding/`](../src/neural_data_decoding/). The
notebooks import directly from this package — re-run any cell as you
read to see live values change.

## Workflow conventions

### Outputs stay out of commits

When you execute a notebook (any IDE, any kernel) Jupyter writes the
cell outputs back into the `.ipynb` JSON. The repo uses
[`nbstripout`](https://github.com/kynan/nbstripout) as a git
clean-filter to remove outputs at commit time so the diffs stay
readable. The filter is configured in `.gitattributes` but
**every contributor must run `nbstripout --install` once after
cloning** — that's what wires the filter into the local
`.git/config`.

`nbstripout` is already in the project's `pip install -e .[dev]`
extras (it's a declared dependency), so the activation is the only
manual step:

```bash
source .venv/bin/activate
nbstripout --install --attributes .gitattributes
```

After that, you can run notebooks freely — the outputs only exist in
your working tree, never in commits. Verify with `nbstripout --status`.

If you really want to commit a notebook with its outputs intact (rare
— useful for tutorial demos), pass `--no-verify` to skip the filter
on that one commit, or use `git add --no-renormalize`.

### Cell IDs

The `_build_notebook.py` helper assigns each cell a short UUID so
NotebookEdit operations have a stable handle to target. `nbstripout`
strips these IDs at commit time (they're regenerated on next
authoring), which is fine — IDs only matter during interactive
editing, not in the committed history.

## Notebook template (every notebook follows this structure)

1. **What MATLAB does** — the actual `cgg_*` MATLAB code, in plain
   English, with a pointer to where it lives in
   `Processing_Functions_cgg/`.
2. **The Python concept(s) you need** — the underlying
   Python / PyTorch concept from first principles, with worked-out
   micro-examples.
3. **The `neural_data_decoding` implementation** — the production
   code, annotated line-by-line, with cross-references to the MATLAB
   source.
4. **Hands-on exercises** — small problems for the reader, with
   hidden-cell solutions.
5. **Diagnostic / debugging walkthrough** — common errors a
   MATLAB-native programmer will hit, what they look like, how to fix
   them.
6. **Further reading** — PyTorch docs, key Python style guides, the
   relevant section of `Codebase_Documentation.md` (MATLAB-side).

## "I'm coming from X background — where do I start?"

| Background | Start here | Then |
|---|---|---|
| MATLAB only, no Python | [00.1 welcome](00_orientation/00.1_welcome.ipynb) | Walk through 00 and 01 sequentially before moving to 02 |
| Python basics, no PyTorch | [02.1 numpy vs MATLAB arrays](02_numpy_and_pytorch_basics/02.1_numpy_vs_matlab_arrays.ipynb) | Cover all of Module 02, then jump into the topic you need |
| Python + PyTorch, new to this codebase | 03.1 dataset vs filedatastore *(not yet authored — start with [02.2 axis conventions](02_numpy_and_pytorch_basics/02.2_array_axis_conventions.ipynb) meanwhile)* | Read the modules in order of the milestone you're working on |
| Just want to extend the production pipeline | 09.6 extending the pipeline *(not yet authored — see [HANDOFF.md](../HANDOFF.md) meanwhile)* | Drop back into the relevant module when a concept is unfamiliar |
| Maintaining MATLAB↔Python parity tests | 08.4 the .mat round-trip test *(not yet authored — see `tests/parity/` meanwhile)* | Cross-reference Module 06 (loss orchestration) for the deeper parity points |

## Prerequisite graph

```
00 Orientation ──┬──> 01 Python for MATLAB users ──> 02 NumPy & PyTorch ──┐
                 │                                                         │
                 └──> 09.1 environment detection (if just deploying)       │
                                                                           ▼
                                                   03 Data pipeline ──> 04 Architecture
                                                          │                   │
                                                          │                   ▼
                                                          │            05 Training loop
                                                          │                   │
                                                          │                   ▼
                                                          └──────────> 06 Loss orchestration
                                                                              │
                                                                              ▼
                                                                       07 Dynamic curriculum
                                                                              │
                                                                              ▼
                                                                       08 Output & analysis
                                                                              │
                                                                              ▼
                                                                  09 Production deployment
```

Modules 03 — 09 can also be read in any order once Modules 00 — 02
are done; the arrows above show the recommended sequence.

## Curriculum map

### Module 00 — Orientation (no prerequisites)
| # | Notebook | Topic |
|---|---|---|
| 00.1 | [welcome.ipynb](00_orientation/00.1_welcome.ipynb) | Tour of the curriculum, prerequisite graph, how to use Jupyter |
| 00.2 | [set_up_your_environment.ipynb](00_orientation/00.2_set_up_your_environment.ipynb) | Install Python, set up venv, install `neural_data_decoding`, hello-world cell |
| 00.3 | [the_matlab_to_python_mental_model.ipynb](00_orientation/00.3_the_matlab_to_python_mental_model.ipynb) | The biggest mindset shifts (everything-is-an-object, 0-indexing, indentation, namespaces) |
| 00.4 | [ide_deep_dive.ipynb](00_orientation/00.4_ide_deep_dive.ipynb) | Foolproof setup for VS Code / JupyterLab / Classic, with troubleshooting and decision matrix |

### Module 01 — Python for MATLAB users
| # | Notebook | MATLAB analog |
|---|---|---|
| 01.1 | `syntax_basics.ipynb` | scripts, functions, variables |
| 01.2 | `control_flow.ipynb` | `if/else/for/while` — and why Python uses indentation |
| 01.3 | `functions_and_lambdas.ipynb` | `varargin` vs `*args/**kwargs`; lambda vs anonymous function |
| 01.4 | `classes_and_oop.ipynb` | `classdef`, inheritance, methods |
| 01.5 | `modules_and_imports.ipynb` | MATLAB path vs Python packages; `from x import y` |
| 01.6 | `error_handling.ipynb` | `try/catch` → `try/except`; reading a Python traceback |
| 01.7 | `dataclasses_and_typed_configs.ipynb` | replacing `CheckVararginPairs` with `@dataclass` |
| 01.8 | `the_python_standard_library_for_matlab_users.ipynb` | `os`, `pathlib`, `json`, `yaml`, `logging` |

### Module 02 — NumPy & PyTorch basics
| # | Notebook | MATLAB analog |
|---|---|---|
| 02.1 | [numpy_vs_matlab_arrays.ipynb](02_numpy_and_pytorch_basics/02.1_numpy_vs_matlab_arrays.ipynb) | array creation, slicing, broadcasting, view vs copy |
| 02.2 | [array_axis_conventions.ipynb](02_numpy_and_pytorch_basics/02.2_array_axis_conventions.ipynb) | MATLAB's `'SSCTB'` vs PyTorch's `(N, C, H, W)` vs the codebase's `(W, T, A, C)` |
| 02.3 | [loading_mat_files.ipynb](02_numpy_and_pytorch_basics/02.3_loading_mat_files.ipynb) | `scipy.io.loadmat` vs `mat73` vs `h5py` |
| 02.4 | [pytorch_tensors_intro.ipynb](02_numpy_and_pytorch_basics/02.4_pytorch_tensors_intro.ipynb) | `torch.Tensor` vs `np.ndarray`, device, dtype |
| 02.5 | `autograd_basics.ipynb` | `requires_grad`, `.backward()`, computational graphs |
| 02.6 | `nn_module_vs_layergraph.ipynb` | `layerGraph` / `dlnetwork` vs `nn.Module` / `nn.Sequential` |
| 02.7 | `optimizers_and_learning_rates.ipynb` | `trainNetwork` options vs `torch.optim.Adam` |
| 02.8 | `nan_handling.ipynb` | MATLAB's implicit NaN tolerance vs PyTorch's strict propagation |

### Module 03 — Data pipeline (companion to Milestone 0 & A)
| # | Notebook | References |
|---|---|---|
| 03.1 | `dataset_vs_filedatastore.ipynb` | `cgg_loadDataArray` ↔ `data.dataset.SyntheticTrialDataset` / `data.mat_dataset.MatFileTrialDataset` |
| 03.2 | `dataloader_and_collation.ipynb` | how MATLAB iterates `fileDatastores` vs PyTorch's `DataLoader` |
| 03.3 | `the_session_balanced_sampler.ipynb` | `cgg_procAllSessionMiniBatchTable` ↔ `data.samplers.SingleSessionBatchSampler` |
| 03.4 | `kfold_stratification_deep_dive.ipynb` | `cgg_getKFoldPartitions` recursive splitting ↔ Python `data.stratification` |
| 03.5 | `normalization_recipes.ipynb` | `cgg_selectNormalization` string dispatch ↔ `data.normalization` registry |
| 03.6 | `augmentation_per_call_contract.ipynb` | why augmentation must re-randomize per `__getitem__` (the silent-parity-loss trap) |

### Module 04 — Architecture (companion to Milestone B)
| # | Notebook | References |
|---|---|---|
| 04.1 | `architecture_string_dispatcher.ipynb` | `PARAMETERS_cgg_constructNetworkArchitecture` ↔ `models.registry` |
| 04.2 | `building_a_simple_encoder.ipynb` | `cgg_constructSimpleCoder` and its Python equivalent |
| 04.3 | `rnn_building_blocks.ipynb` | GRU/LSTM in MATLAB vs PyTorch; batch_first, hidden state |
| 04.4 | `convolutional_backbones.ipynb` | Conv1d / Resnet1d / Multi-Filter — Milestone CC.1 walkthrough |
| 04.5 | `the_bottleneck.ipynb` | flatten + FC; why MATLAB has this exact structure |
| 04.6 | `multi_head_classifier.ipynb` | `nn.ModuleDict` for the multi-head case |
| 04.7 | `weighted_classification_loss.ipynb` | `WeightedLoss='Inverse'` mechanics |
| 04.8 | `weight_initialization_he_vs_pytorch_defaults.ipynb` | why we explicitly call `nn.init.kaiming_normal_` |

### Module 05 — Training loop (companion to Milestone B/C)
| # | Notebook | References |
|---|---|---|
| 05.1 | `the_custom_training_loop.ipynb` | walk through `cgg_trainNetwork` and the Python equivalent end-to-end |
| 05.2 | `gradient_accumulation.ipynb` | `cgg_procGradientAggregation` ↔ PyTorch's native pattern |
| 05.3 | `gradient_clipping.ipynb` | `Global` vs `SubNetwork` clip; `torch.nn.utils.clip_grad_norm_` |
| 05.4 | `learning_rate_scheduling.ipynb` | step-decay + warmup; `torch.optim.lr_scheduler` |
| 05.5 | `checkpoint_resume_state_machine.ipynb` | `cgg_trainAllAutoEncoder_v2.m:171–221` decision tree |
| 05.6 | `the_two_stage_lifecycle.ipynb` | Stage 1 (unsupervised) → Stage 2 (supervised) |
| 05.7 | `batch_norm_state_synchronization.ipynb` | `cgg_updateState` vs PyTorch's automatic running-mean updates |

### Module 06 — Loss orchestration (companion to Milestone C)
| # | Notebook | References |
|---|---|---|
| 06.1 | `multi_task_losses_overview.ipynb` | ELBO + classification + confidence + offset/scale; EMA prior normalization |
| 06.2 | `vae_and_the_elbo.ipynb` | KL intuition, reparameterization trick, `cgg_lossELBO_v2` |
| 06.3 | `stochastic_vs_deterministic_placement.ipynb` | the two graph topologies; why Optimal uses Stochastic |
| 06.4 | `the_ema_prior_normalization_deep_dive.ipynb` | `cgg_getLossInformation` + `cgg_processLossComponent` |
| 06.5 | `mil_softmax_pooling.ipynb` | Multiple Instance Learning intuition; multi-axis softmax |
| 06.6 | `confidence_routing.ipynb` | Trial vs Task confidence; `cgg_addTaskConfidenceToClassifier` |
| 06.7 | `the_confidence_pd_controller.ipynb` | **highest-risk port** — full derivation + parity test walkthrough |
| 06.8 | `l2_inside_the_loss_kernel.ipynb` | why MATLAB's grad-side L2 ≠ PyTorch's `weight_decay` on Adam |
| 06.9 | `per_batch_prior_correction.ipynb` | the `WantBatchCorrection` flag |
| 06.10 | `nan_masked_reconstruction.ipynb` | the two-layered NaN handling (input + loss) |
| 06.11 | `single_total_loss_three_subnetworks.ipynb` | gradient-flow topology |
| 06.12 | `ema_prior_normalization_deep_dive.ipynb` | how cross-component normalization works |
| 06.13 | `sampling_layer_deterministic_at_inference.ipynb` | `self.training`-branched sampling |

### Module 07 — Dynamic curriculum (companion to Milestone C)
| # | Notebook | References |
|---|---|---|
| 07.1 | `curriculum_learning_intuition.ipynb` | why neural decoding benefits from staged training |
| 07.2 | `piecewise_linear_schedules.ipynb` | `cgg_calculateDynamicValue` waypoint interpolation |
| 07.3 | `load_parameters.ipynb` | `cgg_generateLoadParameters_v2` ↔ Python `LoadParameters` |
| 07.4 | `loss_weights_curriculum.ipynb` | `cgg_generateLossWeights_v2`; KL annealing |
| 07.5 | `freeze_unfreeze_curriculum.ipynb` | `cgg_setFrozenNetwork_v2` ↔ `requires_grad` management |
| 07.6 | `walkthrough_soft_three_stage_curriculum_shortened.ipynb` | end-to-end Optimal-curriculum trace |

### Module 08 — Output & analysis (companion to Milestone C/D)
| # | Notebook | References |
|---|---|---|
| 08.1 | `folder_hierarchy_generation.ipynb` | `cgg_generateEncoderSubFolders_v3` ↔ `interop.folder_hierarchy_matlab` |
| 08.2 | `writing_mat_files_for_matlab.ipynb` | producing `.mat` files MATLAB scripts can consume |
| 08.3 | `monitor_table_compatibility.ipynb` | MATLAB monitor field expectations |
| 08.4 | `the_mat_round_trip_test.ipynb` | the T4 parity gate walkthrough |
| 08.5 | `weights_and_biases_integration.ipynb` | W&B as the modern equivalent of the MATLAB monitor system |
| 08.6 | `running_matlab_analysis_on_python_output.ipynb` | train in Python → aggregate with `DATA_cggAllNetworkEncoderResults` |

### Module 09 — Production deployment (companion to Milestone D)
| # | Notebook | References |
|---|---|---|
| 09.1 | `environment_detection.ipynb` | `cgg_getBaseFolders` ↔ Python equivalent |
| 09.2 | `slurm_dispatch.ipynb` | the `sweep-emit-slurm` subcommand + bash → sbatch model |
| 09.3 | `hydra_config_composition.ipynb` | replacing the MATLAB parameter switch with composable configs |
| 09.4 | `parameter_sweeps.ipynb` | replacing the 47-dim `SLURMPARAMETERS_cgg_runAutoEncoder_v2` sweep |
| 09.5 | `debugging_a_failing_run.ipynb` | troubleshooting cookbook: NaN losses, OOM, divergent training, parity-test failures |
| 09.6 | `extending_the_pipeline.ipynb` | how to add a new architecture, loss component, curriculum, or target task |

## Conventions

- **Notebook files** are `.ipynb` with outputs stripped via `nbstripout` so
  diffs stay readable.
- **Code cells** import directly from the installed `neural_data_decoding`
  package — re-run any cell as you read to verify the result on your machine.
- **MATLAB sources** referenced by the notebooks live in the separate
  `Processing_Functions_cgg/` directory pointed to by
  `$NDD_MATLAB_SOURCE_ROOT` (see [`CLAUDE.md`](../CLAUDE.md)).
- **Exercises** use hidden solution cells — try the exercise yourself
  before unhiding.

## Contributing

If you spot a gap, a stale reference, or a confusing explanation:

1. Open an issue describing the gap.
2. Reference the production code (file + line) the notebook should
   teach.
3. PRs welcome — keep the 6-section template, and rerun the notebook
   end-to-end before pushing.
