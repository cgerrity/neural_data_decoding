# ADR 014 — Single-GPU default, accelerate for multi

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline treats multi-GPU as the normal case. Its parallel
trainer `cgg_trainCustomTrainingParallelMultipleOutput.m` opens with a
`canUseGPU` branch: when a GPU is present it sets
`executionEnvironment = "gpu"`, calls
`numberOfGPUs = gpuDeviceCount("available")`, and spins up
`parpool(numberOfGPUs)`. Training then runs inside an `spmd` block that
partitions each mini-batch across workers
(`partition(InDataStore, numWorkers, ...)`), multiplies `miniBatchSize`
by `numWorkers`, and reduces per-worker gradients with `spmdPlus` — a
hand-rolled data-parallel all-reduce living inside
`cgg_procGradientAggregation.m`. `cgg_generateProgressMonitor.m` mirrors
the same `canUseGPU` / `ExecutionEnvironment="auto"` flip just for its
telemetry line.

Two MATLAB-specific artifacts fall out of this:

- The `canUseGPU` flip plus the `parpool` / `spmd` scaffolding exist
  largely because MATLAB's stock training path does not transparently use
  an available GPU — the code has to opt in explicitly and reshape the
  batch to do so.
- On a **single** GPU the parallel path is often slower than just running
  on the one device, so the MATLAB code carries conditional logic to
  avoid the `parfor` / `spmd` machinery in that case. That is a MATLAB
  performance quirk, not a modeling decision.

PyTorch has none of this. A tensor and a module move to a device with
`.to(device)`, and single-device execution is the default fast path — no
pool, no worker partition, no manual all-reduce. Replicating `canUseGPU`,
`gpuDeviceCount`, `parpool`, and the `spmdPlus` reductions would port a
MATLAB workaround into an environment that does not need it.

## Decision

Target a **single compute device** and keep the training kernels
device-parametric, rather than reproducing the MATLAB multi-GPU / `spmd`
topology. What the code actually does today, precisely:

- The per-epoch kernels in `training/loop.py` (`train_one_epoch`,
  `validate`, `train_unsupervised_epoch`, `validate_unsupervised`) take a
  `device: torch.device` argument and move each batch with
  `x.to(device, non_blocking=True)`. They are device-agnostic — hand them
  a `cuda:0` device and they run single-GPU unchanged.
- The CLI (`cli.py`) does **not** yet do device selection. Both training
  entry points — the `fit_supervised(...)` call and the two-stage path in
  `_dispatch_two_stage` → `fit_two_stage(...)` — pass the literal
  `device=torch.device("cpu")`. So an actual run today executes on **CPU**.
  There is no `--device` flag, no `torch.cuda.is_available()` check driving
  placement, and no MPS handling at the CLI layer.
- The **only** live GPU-awareness in the training path is
  `get_accumulation_size_for_current_system` in `training/accumulation.py`.
  It queries `torch.cuda.is_available()` / `get_device_name(i)` /
  `device_count()` to look up the per-device micro-batch cap from
  `cfg.accumulation_information`, taking the `min` across visible GPUs.
  This sizes gradient accumulation for memory headroom; it never moves a
  model or tensor onto a GPU.
- There is no `DataParallel`, no `DistributedDataParallel`, no `accelerate`
  import, and no `hardware.py` module. `accelerate` is not a project
  dependency.

So the honest state is: the pipeline is architected for single-device
execution and the kernels are already device-generic, but the CLI
currently pins CPU. Promoting a run to single-GPU is a localized change —
resolve one `torch.device` (CPU / CUDA / MPS) at the CLI, `model.to(device)`,
and thread that device into the two `fit_*` calls in place of the
hardcoded `torch.device("cpu")`. Multi-GPU is deferred to a future
`accelerate`-based path (wrap model / optimizer / dataloaders in an
`Accelerator`), explicitly **not** a reimplementation of the MATLAB `spmd`
all-reduce. The `canUseGPU` single-vs-multi flip is deliberately not
replicated.

## Consequences

**Positive**

- The kernels stay a single simple code path — no pool lifecycle, worker
  partitioning, or manual gradient all-reduce to maintain or test. The
  "one `backward()` per iteration" invariant (Critical Note #28) survives
  verbatim.
- Because the kernels are device-parametric, single-GPU is a one-line CLI
  change and is already exercisable in tests by passing a `cuda` device to
  the kernel directly.
- Avoids porting a MATLAB-only performance workaround (the `canUseGPU` /
  single-GPU `parfor`-avoidance dance) that has no PyTorch analogue.
- Memory sizing stays hardware-aware via the accumulation table, so a
  large model can be fit on a small GPU without any multi-GPU machinery.
- When multi-GPU genuinely matters, `accelerate` supplies DDP/FSDP in a
  few lines instead of a bespoke `spmdPlus` reduction to debug.

**Negative**

- Today's runs are CPU-only — GPU acceleration is not yet realized despite
  the kernels supporting it. Large real-data runs will be slow until the
  CLI device wire-up lands.
- Multi-GPU data parallelism is unavailable, so single-run throughput is
  capped at one device. Cohort-scale work leans on SLURM array fan-out
  (one device per array task) rather than intra-run parallelism.
- The accumulation table's CUDA detection can resolve a GPU-specific
  micro-batch size even while the model runs on CPU — a latent
  inconsistency if a GPU is visible but unused. Harmless today (entries
  key on device name; CPU falls back to the `"CPU"` entry), but worth
  flagging.
- The title can mislead a future reader: "single-GPU default" is the
  intended posture, not the literal current default, which is CPU.

## Alternatives considered

1. **Replicate MATLAB's `spmd` / `parpool` multi-GPU plus the `canUseGPU`
   flip.** Rejected: it ports a MATLAB workaround into an environment
   where single-GPU is already the fast path, and the manual `spmdPlus`
   all-reduce is exactly what `accelerate` / `torch.distributed` exist to
   replace — the migration spec explicitly says to prefer those over
   reimplementing `cgg_procGradientAggregation`'s parfor.

2. **Adopt `accelerate` up front, now.** Rejected for this milestone: it
   adds a dependency and an `Accelerator` wrapper around every train /
   validate loop before any real multi-GPU workload exists. Single-device
   already covers synthetic smoke runs and SLURM-per-task cohort runs;
   defer until a real >1-GPU need appears.

3. **Auto-select CUDA/MPS at the CLI immediately.** Rejected only as a
   sequencing choice, not on principle: milestone work has been CPU-bound
   synthetic and parity runs where deterministic CPU execution keeps the
   parity fixtures reproducible. The device resolver is a small, separable
   follow-up rather than something to bake in mid-parity-work.

4. **Use `pytorch_lightning` to abstract device / multi-GPU.** Rejected:
   heavier framework buy-in than the custom two-stage loop needs. The plan
   lists Lightning as optional, and the bespoke lifecycle (Stage 1 →
   handoff → Stage 2, EMA priors, confidence Beta threading) is already
   hand-managed, so the abstraction would fight the existing structure.

## References

- Device-parametric kernels: `src/neural_data_decoding/training/loop.py`
  (`device` parameter + `x.to(device, non_blocking=True)` across
  `train_one_epoch` / `validate` / `train_unsupervised_epoch` /
  `validate_unsupervised`).
- CPU pinned at the CLI: `src/neural_data_decoding/cli.py`
  (`fit_supervised(..., device=torch.device("cpu"))` and, via
  `_dispatch_two_stage`, `fit_two_stage(..., device=torch.device("cpu"))`).
- Only live GPU detection today (memory sizing, not placement):
  `src/neural_data_decoding/training/accumulation.py`
  (`get_accumulation_size_for_current_system`).
- MATLAB reference behavior (parity only, not ported):
  `cgg_trainCustomTrainingParallelMultipleOutput.m` (`canUseGPU` →
  `parpool` → `spmd`), `cgg_procGradientAggregation.m` (`spmdPlus`
  reduction), `cgg_generateProgressMonitor.m`
  (`canUseGPU` / `ExecutionEnvironment` flip).
- Migration spec: `docs/PLAN.md` Critical Note #20 (single-GPU default,
  multi-GPU readiness via `accelerate`; don't replicate the `canUseGPU`
  flip). Related: Critical Note #18 (hardware-aware accumulation table),
  Critical Note #19 (memory probe via `torch.cuda.mem_get_info()`).
- Curriculum: `notebooks/09_production_deployment/09.1_environment_detection.ipynb`
  (environment / hardware detection) and
  `notebooks/02_numpy_and_pytorch_basics/02.4_pytorch_tensors_intro.ipynb`
  (device placement, CPU vs GPU).
- Related decision: [ADR 001 — tiered parity](001_tiered_parity_not_bit_exact.md).
- External: [Hugging Face `accelerate`](https://huggingface.co/docs/accelerate).
