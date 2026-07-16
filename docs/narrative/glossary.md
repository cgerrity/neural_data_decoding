# Glossary

Terms used across the codebase and docs.

## Models and losses

**VAE (variational autoencoder)**
: An autoencoder whose encoder outputs a *distribution* (`mu`, `logvar`) over the
  latent rather than a point. See [VAE sampling](concepts/vae_sampling.md).

**ELBO (evidence lower bound)**
: The VAE training objective — a reconstruction term plus a KL term.

**KL divergence**
: A term pulling the latent posterior toward a unit Gaussian; annealed in to
  avoid *posterior collapse*.

**Posterior collapse**
: Degenerate VAE state where the latent carries no information (the KL term won
  before the encoder learned). Prevented by KL annealing.

**MIL (multiple instance learning)**
: Pooling per-window predictions into one per-trial prediction. This project uses
  a joint softmax over (time × class), then marginalizes over time.

**Confidence controller**
: A pure P-controller that keeps mean self-estimated confidence at 0.5. Named
  "PD-controller" in MATLAB but has no derivative term. See
  [The confidence controller](concepts/the_confidence_pd_controller.md).

**EMA prior normalization**
: Dividing each loss component by its running-average magnitude before summing,
  so the *weights* — not raw scales — set the balance.

## Training and curriculum

**Two-stage lifecycle**
: Unsupervised autoencoder pre-training (Stage 1) → handoff → supervised training
  (Stage 2). See [The training lifecycle](concepts/the_training_lifecycle.md).

**Curriculum / dynamic parameters**
: Per-epoch schedules for augmentation, loss weights, and freeze factors. See
  [The dynamic curriculum](concepts/dynamic_curriculum.md).

**Freeze factor**
: A per-subnetwork *learning-rate multiplier* (1.0 = trainable, 0.0 = frozen,
  1e-2 = slow-learn) — not a `requires_grad` toggle.

**Current vs Optimal**
: `current_state.pt` (every epoch, resume point) vs `optimal_state.pt`
  (best validation, model selection).

## Data and deployment

**Session**
: One recording from one set of probes. Every minibatch comes from a single
  session — see [Single-session batching](concepts/single_session_batching.md).

**Fold**
: A cross-validation split. A run trains one `(config, session, fold)`.

**SessionRunIDX**
: The SLURM array task ID, decomposed into `(session, fold)` in
  fold-across-sessions order.

**Sweep index**
: A single integer (1..147) naming one curated config variation.

**CM_Table**
: The MATLAB-compatible `.mat` results file (per-trial predictions, targets,
  confidences). `CM_Table.mat` = test, `CM_Table_Validation.mat` = validation.

**ACCRE**
: Vanderbilt's HPC cluster — one of the three runtime environments (Local /
  TEBA / ACCRE) the pipeline detects.

**Parity tiers**
: T2 = numeric forward-pass parity (~1e-6 tolerance); T4 = output-format parity
  (bit-exact `.mat` round-trip). MATLAB-spawning tests are gated by the
  `needs_matlab` marker.

**Critical Note**
: A numbered parity subtlety from the migration plan. Cited throughout the code
  where a MATLAB behavior must be reproduced exactly.
