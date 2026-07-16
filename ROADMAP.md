# Roadmap — neural_data_decoding

> Generated 2026-07-16 from a verified remaining-work assessment (all code
> milestones 0/A/B/C/CC/D, the educational curriculum E, and reference docs F
> are complete). Each item below was checked against the current code. This is
> a living document — prune items as they land and add ADRs for methodology
> decisions.

---

# neural_data_decoding — Next-Steps Roadmap

## Framing

The project is feature-complete against `docs/PLAN.md`: every code milestone (0/A/B/C/CC/D), the 76-notebook curriculum (E), and reference docs (F) are shipped, and single-step forward parity against MATLAB holds at ~1e-9 to 1e-6. What remains is **not new features but proof, plumbing, and polish**: (1) closing the gap between "one forward pass matches MATLAB" and "training converges the same way MATLAB does" — the actual scientific-trust question; (2) a MATLAB-parity long tail where several *shipped* sweep entries silently no-op or fall back instead of matching MATLAB; (3) wiring the port onto real GPUs and giving long cluster runs observability; and (4) hardening tests/CI and refreshing a few now-stale docs. No item below is a rewrite — most are wiring existing, already-tested kernels into the live path.

## Progress log

**2026-07-16 — first ("unblock + quick-win") batch landed:**

- ✅ **Single-GPU wire-up** — `cli.py` no longer hardcodes CPU; a `train --device`
  flag (`auto` → CUDA else CPU; MPS opt-in) plus model/network/class-weight
  device placement. *Caveat: full GPU training must still be validated on real
  CUDA hardware — an MPS run surfaced that some non-model tensors (class weights,
  now moved; possibly loss priors) need device placement.*
- ✅ **`tests/unit/test_cli.py`** — device resolution, `check-existing`, and the
  `train` clobber-abort exit code 2 (previously-untested CLI surface).
- ✅ **pyright is clean + CI-enforced** — fixed the 2 pre-existing errors
  (`test_pca.py`), added `pyright` to dev extras + a `typecheck` CI job.
- ✅ **`GradientClipType='SubNetwork'`** now warns loudly instead of silently
  falling back to Global; `parameter_coverage.py` corrected ✅→◐.
- ✅ **Doc-freshness** — narrative `index.md` banner, ADR 015 (`mike` is wired),
  and `monitoring/__init__.py` docstring corrected.

**Correction to the ruff item below:** enforcing `ruff` in CI is **not** a quick
"just wire it" win — the code is not currently clean (`ruff check` reports ~199
lint violations, `ruff format --check` wants ~47 files reformatted). It needs a
dedicated codebase-wide cleanup pass first (a large, deliberate mechanical diff).
Also: the project uses **`ruff-format`, not `black`** — `black` is a
declared-but-unused dev dependency.

---

## Theme 1 — Scientific validation (the trust capstone)

Single-step parity does not rule out divergence accumulated over training (optimizer state, RNG streams, batch ordering, loss cadence, confidence collapse, curriculum timing). These items prove — or disprove — that the Python port is *scientifically equivalent*, not just numerically close for one step. This theme carries the most weight.

| Item | Effort | Priority | Key dependency |
|---|---|---|---|
| **T3/G6 — Statistical & convergence parity** (5–10 seed runs both stacks; KS-test / paired-bootstrap on per-epoch validation-accuracy curves). The prescribed `test_e2e_milestone_*` / `test_end_to_end_milestone_*` files in `docs/PLAN.md` (lines 334-335, 472, 513, 559) **do not exist on disk**; no multi-seed comparison anywhere. `.github/workflows/ci.yml` has no cron, so the PLAN's "G6 nightly" cadence is also unwired. | x-large | **high** | MATLAB executable + source root; multi-day compute; deterministic cross-stack seeding |
| **End-to-end run on REAL ephys data, compared to MATLAB.** `tests/integration/test_real_data_smoke.py` runs the full CLI on `results/Decision/Decision_Data_0000011.mat` but asserts only `rc==0` + file existence — **zero numeric comparison**. The port has never been shown to reproduce a real decoding *result*. | large | **high** | Fuller Decision dataset (only one session fixture in-repo); MATLAB reference; cluster compute (Milestone D infra ready) |
| **T4/G7 — Round-trip through the MATLAB analysis scripts** (`DATA_cggAllNetworkEncoderResults` / `FIGURE_*` run on Python `CM_Table.mat` without error, same aggregate). Schema-level round-trip *is* verified (`tests/parity/test_matlab_table_writer.py`, `test_cm_table_parity.py`), but the real consumer scripts are never invoked. | medium | medium | MATLAB + the DATA_/FIGURE_ scripts + a produced `CM_Table.mat` |
| **G8 — Visual dashboard parity.** Manual milestone gate (`docs/PLAN.md` 1077, 1079); no fixtures/reference images exist. Effectively subsumed by G7 since figures derive from the same aggregates. | small | low | MATLAB + FIGURE_* + human eyeball; **blocked behind G7** |

> Note: G6 and the real-data comparison **overlap** — a real-data seed comparison satisfies both, so pair them to amortize the MATLAB reference-run cost. HANDOFF.md's gaps list does not currently mention G6.

---

## Theme 2 — MATLAB-parity long tail

Each item is a parameter that the **shipped** SLURM sweep table or a target config selects, but which silently falls back or no-ops instead of matching MATLAB. Risk is *silent scientific divergence in exercised sweep entries*, not crashes. `parameter_coverage.py` mis-marks several of these as ✅/◐.

| Item | Effort | Priority | Key dependency |
|---|---|---|---|
| **14 NotImplementedError normalization recipes** (`data/normalization.py` 189-221) — only `'None'` and the Optimal recipe implemented. Worse: `select_normalization` is **never called** in the loader (dead code). `SLURMChoice 8` (`dispatcher.py` 344-356) references 4 stub recipes → currently silent no-op. Two-part fix: port the recipes *and* wire `select_normalization` into `mat_dataset.py`. | medium | medium | MATLAB `cgg_procNormalize*`/`cgg_selectNormalization.m`; fixture parity |
| **GradientClipType='SubNetwork' falls back to Global** (`training/loop.py` 327-328, 642-643 clip over all params; `cli.py` 615/703 never read `gradient_clip_type`). Shipped in `configs/target_milestone/A_logistic_synthetic.yaml:45`. Fix: group params by subnetwork and clip each — or **at minimum error on SubNetwork** instead of silent fallback. | small | medium | MATLAB SubNetwork grouping semantics |
| **WantSeparateTimeShift not wired into loader.** Kernel `generate_time_shift_samples` exists + is tested (`augmentation.py` 139-203), but `mat_dataset.py` window extraction is deterministic. `SLURMChoice 15` sets `STDTimeShift=100`+`WantSeparateTimeShift=True` (`dispatcher.py` 520-545) → zero time-shift applied. Fix is Python-only threading, no MATLAB needed. | medium | medium | None (kernel ported already) |
| **BottleNeckDepth>1 stacking** (`models/bottleneck.py` has only single-layer; `composite.py:817` builds one). `SLURMChoice 9/10` request depth 2/3/4 (`dispatcher.py` 369-380). The docstring's own "Milestone C adds the full stack" promise was never completed. | medium | low | MATLAB `cgg_selectBottleNeck.m` (block structure + He-init) |
| **CM_Table per-window column is a single-window placeholder** (`cli.py:1389` `window_predictions=[window]`, uses only last-timestep logits at 1349). The writer (`cm_table_format.py` 80-147) is already N-window-capable; only the CLI producer collapses. Aggregate column (headline metric) is correct. | medium | low | None (writer supports N windows) |
| **per-time-point PCA / ApplyPerTimePoint** (`models/layers/pca.py`) — flag exists only in docstring; no caller in any config/sweep. Purely latent. | medium | optional | MATLAB `cgg_PCAEncodingLayer.m` (if ever needed) |
| **LossType_Classifier only supports CrossEntropy** (`training/losses/classification.py`) — L1/L2 unported; no shipped config requests them (`base.yaml:42` offers only CE). Purely theoretical. | medium | optional | MATLAB classifier L1/L2 defs |

---

## Theme 3 — Execution & observability

Today every real run executes **on CPU** and long cluster runs have no live telemetry. These unblock and instrument the compute that Theme 1 depends on.

| Item | Effort | Priority | Key dependency |
|---|---|---|---|
| **Single-GPU CLI wire-up.** Loop kernels are already device-parametric, but `cli.py` hardcodes `torch.device('cpu')` at both entry points (611, 699); no `--device` flag, no `cuda.is_available()` check. **Most impactful gap for real cluster use** — CPU-only means real runs are intolerably slow (ADR 014). | small | **high** | A CUDA/MPS GPU; localized change |
| **W&B experiment logging.** `wandb>=0.16` is declared (`pyproject.toml:45`) with **zero** code in `src/`. The hook already exists: `EpochCallback` on `fit_supervised`/`stage2_epoch_callback` (`lifecycle.py` 115/138/467). A logger is one callback away. | small | medium | wandb key + cluster egress (may need offline/`wandb sync`) |
| **OOM memory probe** (`torch.cuda.mem_get_info`, Critical Note #19 / ADR 013). Not implemented; ships a static device→micro-batch table (`accumulation.py` 42-83). Additive (table stays as fallback floor), isolated behind `get_accumulation_size_for_current_system`. | medium | medium | A real CUDA GPU to test |
| **`training/monitoring/` package** — empty stub whose `__init__.py` docstring falsely advertises a CM_Table writer (which actually lives in `interop/`). Intended home for the probe + W&B logger. **Cheap honesty fix now:** correct the docstring. | small | low | None (container); contents = above two |
| **Multi-GPU / accelerate** (ADR 014). No DDP/`accelerate`; not a dep. Low risk — cohort work already fans out via SLURM array (one device/task). Genuinely deferred. | large | optional | Add `accelerate`; wrap model/optim/dataloaders |

> ADR-014 latent bug to watch: the accumulation table can resolve a GPU-specific micro-batch size even while the model runs on CPU if a GPU is merely *visible*.

---

## Theme 4 — Test hardening & CI

The suite is broad (792 tests, clean in ~2.6s), but the whole gap cluster is the **CLI layer**, and CI never enforces the type/lint invariants the docs claim.

| Item | Effort | Priority | Key dependency |
|---|---|---|---|
| **CLI pre-flight abort-path test** — `cli.py:387-393` returns exit code 2 on an existing checkpoint without `--force`; the helper is tested but the CLI abort is not. Every `cli_main` test passes `--force` + asserts `rc==0`. Guards against silent checkpoint clobber. | trivial | **high** | None (tmp dir + `capsys`) |
| **Wire pyright into CI** — `ci.yml` runs only pytest + nbconvert; **no pyright job**. CLAUDE.md's "project-wide zero errors" is a local-only guarantee. | small | **high** | Add dev-extra install + `pyright` step |
| **Wire ruff + black (or `pre-commit run --all-files`) into CI** — both configured (`pyproject.toml` 85-100) + in `.pre-commit-config.yaml`, but CI never runs them; enforcement is opt-in per developer. `pre-commit run --all-files` also catches notebook outputs sneaking past nbstripout. | small | medium | Add a `lint` job |
| **CLI tests for `check` + `emit-slurm`** (`_cmd_check_existing` 735, `_cmd_sweep_emit_slurm` 253) — only pure helpers tested (`test_sweep_cli_helpers.py`); the argparse dispatch + these two commands have no direct test. `check` is the operator's dry-run safety tool. | small | medium | None |
| **Broader coverage-gap audit** — add `pytest --cov=neural_data_decoding --cov-report=term-missing` as a CI artifact. Headline gaps are the CLI items above; this just makes future gaps visible. | medium | low | `pytest-cov` |

> Consolidate the three CLI test items into one `tests/unit/test_cli.py` covering all three subcommands' dispatch + exit codes — closing the largest untested shipped-feature surface.

---

## Theme 5 — Docs polish & config

Small, mostly-trivial cleanups. Two are now-stale docs that *invert* the doc-freshness rule (they describe shipped behavior as aspirational).

| Item | Effort | Priority | Key dependency |
|---|---|---|---|
| **ADR 015 freshness** — says mike is "unwired"/"aspirational" and the script "does not invoke mike," but `docs.yml` deploy DOES run `mike deploy`/`mike set-default` and a real gh-pages deployment exists (commit 3610000). Trim overstated `mkdocs.yml:8` too. Preserve: `build_docs.sh` still emits two sibling trees and does *not* call mike; the API site + shared-nav points stay open. | trivial | medium | None |
| **Publish the Sphinx API site to gh-pages** (e.g. `/api/`). `docs.yml` builds+verifies the API but the deploy job (71-75) only publishes the narrative site (comment at 76-78 admits this). Readers are told to clone + `bash scripts/build_docs.sh` instead. | small | medium | gh-pages (exists); one decision: flat `/api/` vs per-version |
| **Unified nav / cross-links** narrative↔API. Only bridge is `index.md:38` (a "build it locally" instruction, not a link). Scope to **cheap bidirectional links only** — a merged nav bar (Material vs furo chrome) is large + low-value (ADR 015 defers it). | small | medium | **Gated on API being published** |
| **Narrative `index.md` status banner stale** — line 5 still says "Milestone B complete; C in progress"; body says the VAE path "is underway." All milestones + E/F are complete. | trivial | low | None (one-line edit) |
| **config/ typed-configuration-dataclasses** — deliberate deferral (ADR 008 Alt #3), honestly documented in its new README. Config is a bare ~70-field OmegaConf `DictConfig`; no static field validation. Tolerable — runs end-to-end today. | medium | optional | None (dataclass layer over existing merge) |
| **Author more ADRs at decision time** — 24 exist; no specific decision pending. The API-publishing approach and any Theme-1 methodology are natural next ADRs. | trivial | optional | None |

---

## Recommended sequence

**First (unblock + cheap high-value, ~1–2 days).** Land the trivial/high CI + safety items and the one change that gates everything expensive: **single-GPU CLI wire-up** (`cli.py` 611/699). Without a GPU, every Theme-1 validation run executes on CPU and is infeasible — this small change is the prerequisite for the scientific work, so it comes before it. Alongside it: the **CLI abort-path test**, **pyright in CI**, the **GradientClipType SubNetwork fix** (or at minimum make it error — it's a silently-wrong *shipped* config, `A_logistic_synthetic.yaml:45`), and the two **doc-freshness inversions** (ADR 015, `index.md`).

**Next (the trust capstone — highest weight).** Start **T3/G6 convergence parity on the smallest config first** — Milestone A logistic tracer / Synthetic_Easy, where `docs/PLAN.md:472` targets "within 2σ across 5 seeds" — before attempting the 10-seed C comparison. **Pair this with the real-ephys-data comparison**, since a real-data seed run satisfies both and amortizes the one-time cost of standing up the MATLAB reference. This is the difference between "the port passes one forward step" and "the port is scientifically equivalent"; nothing else on this roadmap substitutes for it. In parallel (Python-only, no MATLAB blocking), wire the **normalization recipes + `select_normalization`** and **WantSeparateTimeShift** into the loader so the sweep table stops silently under-augmenting/no-op'ing, and add the **W&B logger** so those multi-day runs are observable while they execute.

**Later (round-trip closure + long tail + polish).** Once a real `CM_Table.mat` exists from the validation runs, close **T4/G7** (drive the actual `DATA_`/`FIGURE_` scripts) and then **G8** falls out for near-free. Mop up the remaining MATLAB long-tail parity (**BottleNeckDepth>1**, **CM_Table per-window**, and the optional **per-time-point PCA** / **classifier L1/L2** only if a future experiment needs them), the observability container (**memory probe**, **monitoring/** contents), and the deferred/optional docs+config work (**publish API site → cross-links**, **multi-GPU/accelerate**, **typed config dataclasses**).

---

## Quick wins (an afternoon each, non-blocked)

- **CLI pre-flight abort-path test** (trivial) — `tests/unit/test_cli.py`, assert `rc==2` + stderr via `capsys`.
- **Fix `training/monitoring/__init__.py` docstring** (trivial) — stop advertising the CM_Table writer (lives in `interop/`) + two unbuilt pieces.
- **ADR 015 + `mkdocs.yml:8` freshness** (trivial) — state mike IS wired in the deploy job; keep API-unpublished/no-shared-nav intact.
- **Narrative `index.md:5` status banner** (trivial) — bump to all-milestones-complete; drop the "stubs remain" caveat.
- **pyright CI step** (small) — add to the existing `tests` job or a dedicated `typecheck` job.
- **ruff + black CI** (small) — one `pre-commit run --all-files` job reuses the existing config.
- **Single-GPU CLI wire-up** (small) — resolve one `torch.device` at the CLI, `model.to(device)`, thread into both `fit_*` calls. *(Non-blocked; also the top-priority unblocker.)*
- **GradientClipType SubNetwork → error** (small) — minimal fix: raise instead of silent global fallback, if the full per-subnet clip is deferred.
- **W&B logger callback** (small) — implement against `EpochHistory`, register via the existing `EpochCallback` hook.
- **CLI `check` / `emit-slurm` tests** (small) — fold into the same `test_cli.py`.
- **Publish Sphinx API site** (small, non-blocked) — extra CI step copying `docs/build/api` into `/api/` on gh-pages after `mike deploy`.
