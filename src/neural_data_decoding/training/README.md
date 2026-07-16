# `neural_data_decoding.training`

The training engine: the custom fit loop and per-iteration kernels, the
two-stage (unsupervised → supervised) lifecycle, the loss kernels and the
multi-objective aggregator, curriculum schedules, per-module freezing,
checkpoint/resume, and hardware-aware gradient accumulation. It ports the
behavior of MATLAB's `cgg_trainNetwork` / `cgg_lossComponents` /
`cgg_getLossInformation` family (parity-verified, not literal-mirrored).

## Layout

| Path | What it holds |
|------|---------------|
| `lifecycle.py` | Epoch-driving fit loops + two-stage orchestration. |
| `loop.py` | Per-epoch train / validate kernels (the mini-batch body). |
| `losses/` | Loss kernels (classification, ELBO, confidence, offset/scale) + the multi-objective aggregator. |
| `schedules/` | Curriculum schedules (piecewise-linear anneal, weight/freeze/load bundles). |
| `freezing.py` | Per-module optimizer param-groups + per-group LR scaling. |
| `checkpoint.py` | Current / Optimal snapshot state machine for resume. |
| `accumulation.py` | Micro-batch chunking for gradient accumulation. |

## Main entry points

- **`fit_two_stage(...)`** (`lifecycle.py`) — runs Stage 1 unsupervised
  autoencoder pre-training, hands the Optimal encoder/decoder weights off
  into the composite, then runs Stage 2 supervised fine-tuning. Also
  exported as the two single-stage loops it composes: `fit_supervised`
  (accuracy-maximizing, writes Optimal on best val accuracy) and
  `fit_unsupervised` (loss-minimizing Stage 1). These loops own epoch
  iteration, resume offset, curriculum update + freeze application,
  EMA-prior cadence, and per-epoch checkpointing.
- **`train_one_epoch(...)` / `validate(...)`** (`loop.py`) — the
  per-iteration kernel. One forward → one aggregated scalar loss → one
  `.backward()` → optimizer step, per mini-batch; also drives the
  unsupervised variants and the MIL / accumulation / confidence paths.
- **`aggregate_total_loss(...)` / `aggregate_normalized_losses(...)`**
  (`losses/multi_objective.py`) — assemble the single gradient-root scalar
  from the active components. `aggregate_total_loss` is the plain weighted
  sum; `aggregate_normalized_losses` applies the EMA-prior normalization
  before the weighted sum.

## Honest notes on current behavior

- The module docstrings in `lifecycle.py` and `loop.py` are written from a
  Milestone-A ("classifier only, curriculum/VAE/confidence are future
  work") perspective and are now **stale**. The implemented functions
  already thread curriculum, VAE reconstruction + KL, confidence routing,
  MIL, and gradient accumulation. Trust the function signatures, not those
  file-top narratives.
- **Optimizer choice / ADR 003:** `resolve_optimizer_factory` maps the
  config's `"ADAM"` to `torch.optim.AdamW` (decoupled weight decay) and
  `"SGDM"` to `torch.optim.SGD(momentum=0.9)`. AdamW's decoupled decay is a
  deliberate divergence from MATLAB's coupled L2 gradient — see the ADR.
- **Resume drops optimizer state / ADR 005–006:** checkpoints save model
  weights but **not** optimizer state, and resume reads the *Current*
  snapshot, never *Optimal*. After a resume, AdamW's moment estimates
  restart from zero, so the first post-resume iterations differ slightly
  from an uninterrupted run.
- **"PD controller" (confidence) / ADR 020:** `losses/confidence.py` is
  titled a PD-controller port, but what it actually implements is
  confidence routing — multiplicative Task×Trial conjunction (Eq. 1),
  prediction-to-truth interpolation of the classifier loss (Eq. 2),
  ConfidenceDropout, a budget regularizer, and a stop-gradient EMA history
  with BatchFraction-governed cadence. Read it as "confidence routing +
  budget regularizer," not a literal PD loop.
- **Freezing:** PyTorch has no `setLearnRateFactor`; freezing is emulated
  by one optimizer param-group per submodule with per-group
  `lr = base_lr * factor` (factor 0 = frozen, 1e-2 = "slow but momentum
  alive"). Requires `freeze_base_lr` to be passed, else the freeze
  schedule is computed but not applied.

## Related ADRs

- [003 — AdamW for L2 weight decay](../../../docs/narrative/adrs/003_adamw_for_l2_weight_decay.md)
- [005 — No optimizer state in checkpoints](../../../docs/narrative/adrs/005_no_optimizer_state_in_checkpoints.md)
- [006 — Resume reads Current, not Optimal](../../../docs/narrative/adrs/006_resume_reads_current_not_optimal.md)
- [009 — EMA prior cadence via RescaleLossEpoch](../../../docs/narrative/adrs/009_ema_prior_cadence_via_rescale_loss_epoch.md)
- [019 — Single total loss, three subnetworks](../../../docs/narrative/adrs/019_single_total_loss_three_subnetworks.md)
- [020 — Confidence loss: five subtleties](../../../docs/narrative/adrs/020_confidence_loss_five_subtleties.md)
- [021 — EMA prior normalized to classification](../../../docs/narrative/adrs/021_ema_prior_normalized_to_classification.md)

## Learn more

- Concepts: [the training lifecycle](../../../docs/narrative/concepts/the_training_lifecycle.md),
  [multi-objective losses](../../../docs/narrative/concepts/multi_objective_losses.md),
  [dynamic curriculum](../../../docs/narrative/concepts/dynamic_curriculum.md),
  [the confidence PD controller](../../../docs/narrative/concepts/the_confidence_pd_controller.md).
- Cookbook: [add a new loss component](../../../docs/narrative/cookbook/add_a_new_loss_component.md),
  [add a new curriculum schedule](../../../docs/narrative/cookbook/add_a_new_curriculum_schedule.md).
- Notebooks: [`05_training_loop/`](../../../notebooks/05_training_loop/),
  [`06_loss_orchestration/`](../../../notebooks/06_loss_orchestration/),
  [`07_dynamic_curriculum/`](../../../notebooks/07_dynamic_curriculum/).
