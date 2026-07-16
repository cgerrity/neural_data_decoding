# neural_data_decoding

Python port of the MATLAB neural decoding pipeline.

!!! info "Project status: all milestones complete"
    Milestones 0/A/B/C/CC/D (the full pipeline), the 76-notebook educational
    curriculum (E), and this reference documentation (F) are complete and
    runnable end-to-end on synthetic data, with single-step forward-pass parity
    against MATLAB verified to ~1e-9. See `ROADMAP.md` at the project root for
    what remains (scientific convergence validation, a deferred MATLAB-parity
    long tail, and polish) and `docs/PLAN.md` for the full migration spec.

## What this pipeline does

Trains a variational autoencoder + multi-head classifier on multi-probe ephys
data, with:

- ELBO + multi-head classification + trial/task confidence losses
- Curriculum-based dynamic parameter scheduling
- Hierarchical stratified K-fold cross-validation
- Single-session minibatching (every minibatch from one session — see [Single-Session Batching](concepts/single_session_batching.md))
- Two-stage training lifecycle (unsupervised pre-training → supervised fine-tuning)

The port reproduces the active MATLAB production path using modern PyTorch
patterns while writing `.mat`-compatible output where MATLAB-side analysis
still consumes it.

## Quick navigation

| If you want to … | Start with |
|------------------|-----------|
| **Get the pipeline running** | [Quickstart](quickstart.md) |
| **Understand how training works** | [The training lifecycle](concepts/the_training_lifecycle.md) |
| **Add a new architecture or loss** | [Cookbook](cookbook/add_a_new_architecture.md) |
| **Debug a failing run** | [Troubleshooting](troubleshooting.md) |
| **See why a design decision was made** | [Decision records](adrs/index.md) |
| **Look up a Python API symbol** | API reference — built separately by Sphinx; run `bash scripts/build_docs.sh` then open `docs/build/api/index.html` |
| **Learn Python coming from MATLAB** | Educational notebooks — in the repo at `neural_data_decoding/notebooks/` (Jupyter) |

## Parity status

See the README in the repo root for the current milestone-by-milestone status.
