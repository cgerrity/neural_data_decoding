# Architecture Decision Records

ADRs are short markdown documents recording **why** significant design decisions
were made. They are immutable — once accepted, an ADR is never deleted; if a
later decision supersedes it, the new ADR links back to the old one.

Each ADR follows the same template:

1. **Context** — the situation that prompted the decision.
2. **Decision** — what was chosen.
3. **Consequences** — what this commits us to (positive and negative).
4. **Alternatives considered** — options that were weighed and rejected.
5. **References** — links to relevant code, plan sections, external sources.

## Index

| # | Title | Status |
|---|-------|--------|
| [001](001_tiered_parity_not_bit_exact.md) | Tiered parity, not bit-exact | Accepted |
| [002](002_pythonic_structure_over_matlab_mirror.md) | Pythonic structure over MATLAB mirror | Accepted |
| [003](003_adamw_for_l2_weight_decay.md) | AdamW for L2 weight decay | Accepted |
| [004](004_single_session_batching.md) | Single-session batching | Accepted |
| [005](005_no_optimizer_state_in_checkpoints.md) | No optimizer state in checkpoints | Accepted |
| [006](006_resume_reads_current_not_optimal.md) | Resume reads Current, not Optimal | Accepted |
| [007](007_mat_interop_surface.md) | MAT interop surface | Accepted |
| [008](008_hydra_config_composition.md) | Composable YAML config (OmegaConf) | Accepted |
| [009](009_ema_prior_cadence_via_rescale_loss_epoch.md) | EMA prior cadence via RescaleLossEpoch | Accepted |
| [010](010_augmentation_per_getitem.md) | Augmentation per `__getitem__` | Accepted |
| [011](011_validation_per_epoch_default.md) | Validation per epoch by default | Accepted |
| [012](012_pre_flight_check_no_overwrite.md) | Pre-flight check, no overwrite | Accepted |
| [013](013_memory_probe_via_cuda_mem_get_info.md) | Memory probe via `cuda.mem_get_info` | Accepted |
| [014](014_single_gpu_default_accelerate_for_multi.md) | Single-GPU default, accelerate for multi | Accepted |
| [015](015_two_doc_toolchains_mkdocs_plus_sphinx.md) | Two doc toolchains (MkDocs + Sphinx) | Accepted |
| [016](016_minimal_matlab_cross_referencing_in_api_docs.md) | Minimal MATLAB cross-referencing in API docs | Accepted |
| [017](017_nan_masked_reconstruction_loss.md) | NaN-masked reconstruction loss | Accepted |
| [018](018_layer_block_order_dropout_before_norm.md) | Layer block order: dropout before norm | Accepted |
| [019](019_single_total_loss_three_subnetworks.md) | Single total loss, three subnetworks | Accepted |
| [020](020_confidence_loss_five_subtleties.md) | Confidence loss: five subtleties | Accepted |
| [021](021_ema_prior_normalized_to_classification.md) | EMA prior normalized to classification | Accepted |
| [022](022_he_initialization_explicit.md) | He initialization, explicit | Accepted |
| [023](023_augmentation_loss_auto_activated_by_topology.md) | Augmentation loss auto-activated by topology | Accepted |
| [024](024_sampling_layer_deterministic_at_inference.md) | Sampling layer deterministic at inference | Accepted |
