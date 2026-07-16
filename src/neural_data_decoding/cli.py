"""Command-line entry point for the neural_data_decoding pipeline.

Examples
--------
After ``pip install -e .``, the package can be invoked as a module::

    python -m neural_data_decoding --help
    python -m neural_data_decoding train --config-name A_logistic_synthetic
    python -m neural_data_decoding check-existing --config-name A_logistic_synthetic

The ``train`` subcommand loads a composed config from
``configs/target_milestone/<name>.yaml`` (with ``base.yaml`` as the default
inherited base), builds the Milestone A logistic-regression pipeline, runs
:func:`~neural_data_decoding.training.lifecycle.fit_supervised`, and writes
the ``CM_Table_Validation.mat`` + ``EncodingParameters.yaml`` outputs to a
deterministic result directory derived from the config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from . import __version__
from .data.dataset import SyntheticTrialDataset, collate_trials
from .data.mat_dataset import MatFileTrialDataset
from .interop import (
    ENCODING_PARAMETERS_FILENAME,
    TEST_CM_TABLE_FILENAME,
    VALIDATION_CM_TABLE_FILENAME,
    build_matlab_run_dirs,
    write_cm_table_mat,
    write_encoding_parameters_yaml,
)
import neural_data_decoding.models  # noqa: F401 — triggers architecture registrations
from .models.bottleneck import LinearBottleneck, PassthroughBottleneck
from .models.classifier import MultiHeadClassifier
from .models.composite import (
    EncoderClassifierComposite,
    build_variational_autoencoder,
    build_variational_composite,
)
from .models.registry import build_classifier, build_encoder
from .sweeps.banner import collect_banner_data, render_banner
from .sweeps.cli_helpers import (
    apply_overrides,
    apply_sweep_index,
    decompose_session_run_idx,
)
from .sweeps.slurm_template import (
    DEFAULT_ARRAY_THROTTLE,
    DEFAULT_CPUS_PER_TASK,
    DEFAULT_MEM,
    DEFAULT_NUM_FOLDS,
    DEFAULT_NUM_SESSIONS,
    DEFAULT_OUTPUT_DIR_REL,
    DEFAULT_TIME,
    SlurmTemplateOptions,
    write_slurm_template,
)
from .training.accumulation import get_accumulation_size_for_current_system
from .training.checkpoint import has_existing_checkpoint
from .training.freezing import (
    build_optimizer_with_module_groups,
    resolve_optimizer_factory,
)
from .training.lifecycle import EpochHistory, fit_supervised, fit_two_stage
from .training.losses.confidence import ConfidenceHistory
from .training.losses.multi_objective import LossPriors
from .training.losses.classification import (
    aggregate_classifier_predictions,
    inverse_frequency_class_weights,
)
from .training.schedules import (
    CurriculumBundle,
    KLBaseAnneal,
    load_curriculum_by_name,
)
from .utils.seeding import set_global_seed


CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "configs"
DEFAULT_OUTPUT_ROOT = CONFIG_ROOT.parent / "results"


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the requested subcommand.

    Parameters
    ----------
    argv
        Optional list of arguments. Defaults to :data:`sys.argv` when ``None``.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``. ``0`` is success.
    """
    parser = argparse.ArgumentParser(
        prog="neural_data_decoding",
        description=(
            "Python port of the MATLAB neural decoding pipeline. "
            "Subcommands: train, check-existing, sweep."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Run a training session.")
    _add_common_args(train_p)
    train_p.add_argument(
        "--force",
        action="store_true",
        help="Allow training even if existing checkpoints are present in the result directory.",
    )
    train_p.add_argument(
        "--device",
        default="auto",
        help=(
            "Compute device: 'auto' (default; prefers CUDA, then Apple MPS, "
            "else CPU), or an explicit device string like 'cuda', 'cuda:0', "
            "'mps', or 'cpu'."
        ),
    )
    train_p.add_argument(
        "--wandb",
        action="store_true",
        help="Stream per-epoch metrics to a Weights & Biases run (off by default).",
    )
    train_p.add_argument(
        "--wandb-project",
        default="neural-data-decoding",
        help="W&B project name (used only with --wandb).",
    )
    train_p.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B run mode (used only with --wandb).",
    )
    train_p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Global RNG seed applied before model build + training, so the "
            "same (config, fold, seed) reproduces the same run (default 0). "
            "Vary it to draw an ensemble of seeds — e.g. --seed 1..5 for a "
            "five-seed convergence study. Data splits are always seeded "
            "separately from the fold index and are unaffected."
        ),
    )

    check_p = sub.add_parser(
        "check-existing",
        help="Resolve the result directory and report whether any checkpoint files would be clobbered.",
    )
    _add_common_args(check_p)

    emit_p = sub.add_parser(
        "sweep-emit-slurm",
        help="Render a SLURM array .slurm file for a given sweep index.",
    )
    _add_sweep_emit_args(emit_p)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "check-existing":
        return _cmd_check_existing(args)
    if args.command == "sweep-emit-slurm":
        return _cmd_sweep_emit_slurm(args)

    parser.print_help()
    return 0


def _add_sweep_emit_args(p: argparse.ArgumentParser) -> None:
    """Attach the flag set for the ``sweep-emit-slurm`` subcommand."""
    p.add_argument(
        "--sweep-index",
        type=int,
        required=True,
        help="Sweep entry index (1-based) to bake into the .slurm script.",
    )
    p.add_argument(
        "--config-name",
        required=True,
        help="Name of a YAML file in configs/target_milestone/ (without .yaml).",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to write the .slurm file (parents are created as needed).",
    )
    p.add_argument(
        "--num-sessions",
        type=int,
        default=DEFAULT_NUM_SESSIONS,
        help=f"Cohort session count (default: {DEFAULT_NUM_SESSIONS}).",
    )
    p.add_argument(
        "--num-folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help=f"K-fold count (default: {DEFAULT_NUM_FOLDS}).",
    )
    p.add_argument(
        "--time",
        type=str,
        default=DEFAULT_TIME,
        help=f"SLURM --time value (default: {DEFAULT_TIME}).",
    )
    p.add_argument(
        "--mem",
        type=str,
        default=DEFAULT_MEM,
        help=f"SLURM --mem value (default: {DEFAULT_MEM}).",
    )
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=DEFAULT_CPUS_PER_TASK,
        help=f"SLURM --cpus-per-task (default: {DEFAULT_CPUS_PER_TASK}).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR_REL,
        help=(
            "SLURM --output target directory inside the job. Default: "
            f"{DEFAULT_OUTPUT_DIR_REL}/ (matches the MATLAB pipeline convention)."
        ),
    )
    p.add_argument(
        "--array-throttle",
        type=int,
        default=DEFAULT_ARRAY_THROTTLE,
        help=(
            f"Concurrent-task throttle for the SLURM array (default: "
            f"{DEFAULT_ARRAY_THROTTLE} — sequential). Becomes the '%%K' "
            "suffix on the --array spec."
        ),
    )
    p.add_argument(
        "--mail-user",
        type=str,
        default=None,
        help=(
            "Override --mail-user. When omitted, the helper auto-detects "
            "the project owner via $USER / git email; for anyone else the "
            "mail line is left out of the script."
        ),
    )
    p.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help=(
            "Optional absolute path to inject as the 'cd ...' line before "
            "venv activation. Use this when the script will be sbatched "
            "from somewhere other than the repo root."
        ),
    )
    p.add_argument(
        "--extra-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra --override flag forwarded to the inner train command. "
            "Repeatable. Use to pin real-data paths (--extra-override "
            "data_dir=/scratch/...)."
        ),
    )


def _cmd_sweep_emit_slurm(args: argparse.Namespace) -> int:
    """Render and write a sweep ``.slurm`` array script."""
    options = SlurmTemplateOptions(
        sweep_index=int(args.sweep_index),
        config_name=str(args.config_name),
        num_sessions=int(args.num_sessions),
        num_folds=int(args.num_folds),
        cpus_per_task=int(args.cpus_per_task),
        time=str(args.time),
        mem=str(args.mem),
        output_dir=str(args.output_dir),
        array_throttle=int(args.array_throttle),
        mail_user=args.mail_user,
        repo_dir=args.repo_dir,
        extra_overrides=tuple(args.extra_override),
    )
    written = write_slurm_template(options, Path(args.output_path))
    print(f"Wrote {written}")
    return 0


def _apply_cfg_flags(cfg: DictConfig, args: argparse.Namespace) -> None:
    """Apply CLI flags to ``cfg`` in the precedence order documented per-flag.

    Order (later wins):

    1. ``--sweep-index`` — bundled override set from the dispatcher table.
    2. ``--session-run-idx`` — sets ``cfg.session_run_idx`` (the dataset
       loader decomposes it once it knows ``num_sessions``); also sets
       ``cfg.fold`` from the decomposition assuming a 25-session cohort
       so single-session smoke runs work without the loader being live.
    3. ``--session`` — sets ``cfg.subset`` to the named session
       (hyphens → underscores).
    4. ``--override KEY=VALUE`` — ad-hoc cfg surgery (escape hatch).
    5. ``--fold`` — final say on fold index (wins over sweep-index
       and session-run-idx).

    The ``cfg.sweep_index`` / ``cfg.session_run_idx`` fields end up in
    the EncodingParameters.yaml so the run can be reproduced.
    """
    if args.sweep_index is not None:
        description, notes = apply_sweep_index(cfg, int(args.sweep_index))
        cfg.sweep_index = int(args.sweep_index)
        cfg.sweep_description = description
        for note in notes:
            print(f"[sweep #{args.sweep_index}] note: {note}", file=sys.stderr)
    if args.session_run_idx is not None:
        cfg.session_run_idx = int(args.session_run_idx)
        # Heuristic fold derivation against the default 25-session cohort
        # so synthetic smoke runs still pick a useful fold; the real-data
        # loader recomputes this against the actual session count.
        decomposition = decompose_session_run_idx(int(args.session_run_idx), 25)
        cfg.fold = decomposition.fold
    if args.session is not None:
        cfg.subset = str(args.session).replace("-", "_")
    if args.override:
        apply_overrides(cfg, list(args.override))
    if args.fold is not None:
        cfg.fold = int(args.fold)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Attach the standard config + sweep flag group."""
    p.add_argument(
        "--config-name",
        required=True,
        help="Name of a YAML file in configs/target_milestone/ (without .yaml).",
    )
    p.add_argument(
        "--fold", type=int, default=None, help="Override fold index (else uses config)."
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Top-level results directory. Default: <neural_data_decoding>/results "
            "(gitignored). Set to <ACCRE_DATA>/... for cluster-equivalent paths."
        ),
    )
    p.add_argument(
        "--sweep-index",
        type=int,
        default=None,
        help=(
            "Apply sweep entry N (1-based) from the SLURMPARAMETERS port. "
            "Bundles cfg overrides; see "
            "neural_data_decoding.sweeps.dispatcher for the full table."
        ),
    )
    p.add_argument(
        "--session-run-idx",
        type=int,
        default=None,
        help=(
            "Flat MATLAB-style index K decomposed into (session_idx, fold) "
            "via session_idx = (K-1) %% NumSessions + 1 and "
            "fold = (K-1) // NumSessions + 1. Preserves MATLAB's "
            "session-inside-fold ordering so the cohort's first-fold accuracy "
            "lands across every session before any second-fold runs start."
        ),
    )
    p.add_argument(
        "--session",
        type=str,
        default=None,
        help=(
            "Single-session mode: set cfg.subset to this session name. "
            "Hyphens are converted to underscores to match the Target.SessionName "
            "format used inside .mat files."
        ),
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Ad-hoc cfg override (repeatable). KEY is a Python cfg field "
            "(snake_case); VALUE is parsed via ast.literal_eval first, "
            "then falls back to string. Wins over --sweep-index. Example: "
            "--override data_width=50 --override hidden_sizes=[500,250]."
        ),
    )


def _resolve_device(name: str) -> torch.device:
    """Resolve a ``--device`` argument to a concrete :class:`torch.device`.

    Parameters
    ----------
    name
        Either ``"auto"`` (prefer CUDA, then Apple MPS, else CPU) or an explicit
        device string such as ``"cuda"``, ``"cuda:0"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    torch.device
        The resolved device. ``"auto"`` never raises; an explicit device that is
        unavailable falls back to CPU with a warning so a run is never silently
        blocked on a missing accelerator.
    """
    if name == "auto":
        # Auto-select CUDA (the cluster target) or fall back to CPU. Apple MPS
        # is deliberately NOT auto-selected: several ops in this pipeline are
        # unsupported on the MPS backend, so it would break rather than
        # accelerate. Mac users can still opt in explicitly with ``--device mps``.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            f"+++ Requested device '{name}' but CUDA is unavailable; falling back to CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        print(
            f"+++ Requested device '{name}' but MPS is unavailable; falling back to CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    return device


def _cmd_train(args: argparse.Namespace) -> int:
    """Implementation of the ``train`` subcommand."""
    # Seed every RNG before anything stochastic (model init, dropout, batch
    # shuffling) so the same (config, fold, seed) reproduces the same run.
    # Data splits carry their own fold-derived seeds and are unaffected.
    set_global_seed(int(args.seed))

    cfg = _load_config(args.config_name)
    _apply_cfg_flags(cfg, args)

    result_dir = _resolve_result_dir(cfg, args.output_root)
    result_dir.mkdir(parents=True, exist_ok=True)

    if has_existing_checkpoint(result_dir) and not args.force:
        print(
            f"ERROR: existing checkpoints found at {result_dir}. "
            "Delete them or re-run with --force.",
            file=sys.stderr,
        )
        return 2

    # Write the resolved EncodingParameters.yaml up-front (stable schema —
    # Critical Note #25). Sweep launchers will overwrite the schema_template
    # but this captures the resolved per-run config either way.
    raw_schema = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw_schema, dict)
    # Drop Hydra-internal keys that shouldn't appear in the MATLAB-facing YAML;
    # also stringify keys (OmegaConf can technically return int/bool keys though
    # our configs never do — the str() narrows the type to dict[str, Any]).
    schema: dict[str, Any] = {
        str(k): v for k, v in raw_schema.items() if k != "defaults"
    }
    write_encoding_parameters_yaml(
        result_dir / ENCODING_PARAMETERS_FILENAME,
        run_config=schema,
        schema_template=schema,
    )

    # Curriculum (Milestone C #5) — load the named regime preset from
    # configs/schedule/ if the config asks for one. Defaults to "None"
    # (no schedules), which keeps every parameter at its base value.
    curriculum = _build_curriculum(cfg)

    # Build the datasets (train/val/test), model, optimizer, then fit.
    # Real-data path activates when cfg.data_dir is set (not the Hydra
    # ??? sentinel); otherwise fall back to the synthetic generator.
    # Only the training dataset receives the load schedule — val/test
    # stay un-augmented so the metrics reflect the model, not the
    # augmentation.
    use_real_data = _real_data_path_active(cfg)
    if use_real_data:
        train_ds, val_ds, test_ds = _build_real_data_split(
            cfg,
            train_load_schedule=(
                curriculum.load if curriculum is not None else None
            ),
        )
    else:
        train_ds, val_ds, test_ds = _build_synthetic_split(
            cfg,
            train_load_schedule=(
                curriculum.load if curriculum is not None else None
            ),
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.mini_batch_size),
        shuffle=True,
        collate_fn=collate_trials,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.mini_batch_size),
        shuffle=False,
        collate_fn=collate_trials,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.mini_batch_size),
        shuffle=False,
        collate_fn=collate_trials,
    )

    # Build the model. Milestone A's Logistic Regression has no encoder
    # — the classifier consumes raw features directly. Milestone B's GRU
    # path composes Encoder → Bottleneck → Classifier so the classifier
    # sees the encoder's hidden representation.
    if use_real_data:
        # Real data: derive (T, A, C) from the dataset itself, then thread
        # them through cfg so the composite-builder's UnflattenPerWindow
        # restores the correct shape after the decoder. The encoder
        # builders multiply T*A*C internally, so ``num_features`` here is
        # C only (channels per area), NOT the flat product.
        assert isinstance(train_ds, MatFileTrialDataset)
        probe_x, _, _ = train_ds[0]
        # probe_x is (W, T, A, C).
        cfg.synthetic_samples_per_window = int(probe_x.shape[1])
        cfg.synthetic_num_areas = int(probe_x.shape[2])
        num_features = int(probe_x.shape[3])
        num_classes_per_dim = list(train_ds.num_classes_per_dim)
    else:
        num_features = int(cfg.synthetic_num_features)
        num_classes_per_dim = list(cfg.synthetic_num_classes_per_dim)

    # Start-of-run banner — mirrors the diagnostic block at
    # cgg_runAutoEncoder.m lines 320-323. Printed to stderr so it
    # never confuses downstream parsers consuming stdout.
    banner_data = collect_banner_data(
        config_name=str(args.config_name),
        cfg=cfg,
        args=args,
        result_dir=result_dir,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        use_real_data=use_real_data,
        num_classes_per_dim=num_classes_per_dim,
    )
    print(render_banner(banner_data), file=sys.stderr, flush=True)
    model = _build_model(
        cfg,
        in_features=num_features,
        num_classes_per_dim=num_classes_per_dim,
    )
    # CC.2 — fit the PCA encoder on training data before the optimizer
    # is built, since PCA holds buffers (not parameters) and the
    # optimizer would otherwise have nothing to do on the frozen
    # transform. The fit method walks the train_loader once.
    _fit_pca_if_present(model, train_loader)
    # Optimizer: when a freeze schedule is active, build per-module param
    # groups so apply_freeze_to_optimizer can scale each network's lr
    # independently (mirrors MATLAB setLearnRateFactor per submodule).
    # Without a freeze schedule, a single group keeps existing behavior.
    # Resolve the compute device and move the model onto it BEFORE building the
    # optimizer (PCA is fit above on CPU data first; ``model.to`` then moves its
    # buffers along with the rest). The loop kernels already move each batch to
    # this device, so this is the only wiring the CLI needs for GPU training.
    device = _resolve_device(args.device)
    model = model.to(device)
    print(f"+++ Training device: {device}", file=sys.stderr, flush=True)
    # GradientClipType='SubNetwork' is not yet implemented (the loop always
    # applies a Global gradient-norm clip). Surface the fallback loudly instead
    # of silently diverging from a config that requests per-subnetwork clipping.
    _clip_type = str(cfg.get("gradient_clip_type", "Global"))
    if _clip_type not in ("Global", ""):
        print(
            f"+++ WARNING: GradientClipType='{_clip_type}' is not implemented; "
            "falling back to Global gradient-norm clipping.",
            file=sys.stderr,
            flush=True,
        )
    optimizer = _build_optimizer(cfg, model, curriculum)

    train_labels = torch.from_numpy(train_ds._labels).long()  # noqa: SLF001
    class_weights = (
        inverse_frequency_class_weights(train_labels, num_classes_per_dim)
        if str(cfg.weighted_loss).lower() == "inverse"
        else None
    )
    # The per-dimension class-weight tensors are passed to the cross-entropy
    # kernel, which requires them on the same device as the logits. No-op on CPU.
    if class_weights is not None:
        class_weights = [w.to(device) for w in class_weights]

    # For variational configs, expose all per-component weights AND enable
    # EMA prior normalization. For Logistic/non-variational, only classification
    # is active, so the simpler weight dict suffices.
    is_variational = bool(cfg.get("is_variational", False))
    confidence_active = is_variational and bool(
        cfg.get("confidence_type") and float(cfg.get("weight_confidence", 0)) != 0,
    )
    mil_mode = (
        is_variational
        and str(cfg.get("multiple_instance_learning_type", "None")) == "MIL"
    )

    # Hardware-aware gradient accumulation (Critical Note #18). Reads
    # cfg.accumulation_information as a {system_name: max_micro_batch}
    # mapping; resolves to the current device's entry. None falls back to
    # no accumulation (single-pass per mini-batch).
    accum_info_raw = cfg.get("accumulation_information", None)
    accumulation_max_size: Optional[int] = None
    if accum_info_raw is not None:
        accum_info: dict[str, int]
        if isinstance(accum_info_raw, list):
            # OmegaConf list-of-dicts form (one item per entry).
            accum_info = {
                str(item.get("SystemName") or item.get("system_name", "")): int(
                    item.get("MaxBatchSize") or item.get("max_batch_size", 0),
                )
                for item in accum_info_raw
            }
            accum_info = {k: v for k, v in accum_info.items() if k and v > 0}
        else:
            # Dict form.
            accum_info = {str(k): int(v) for k, v in dict(accum_info_raw).items()}
        accumulation_max_size = get_accumulation_size_for_current_system(accum_info)
    initial_confidence_history: Optional["ConfidenceHistory"] = None
    if is_variational:
        loss_weights_dict: dict[str, float] = {
            "classification": float(cfg.weight_classification),
            "reconstruction": float(cfg.get("weight_reconstruction", 1.0)),
            "kl": float(cfg.get("weight_kl", 1.0)),
        }
        if confidence_active:
            loss_weights_dict["confidence"] = float(cfg.weight_confidence)
            initial_confidence_history = ConfidenceHistory.initial(dtype=torch.float32)
        initial_priors: Optional[LossPriors] = LossPriors.initial()
    else:
        loss_weights_dict = {"classification": float(cfg.weight_classification)}
        initial_priors = None

    # Whenever an epoch becomes the new best validation metric, write BOTH
    # CM_Tables from the just-optimal model state. The on-disk files always
    # reflect the best-so-far model — inspectable mid-training, available
    # even if training is killed before completion. Matches MATLAB's
    # cgg_trainNetwork.m:636-641 pattern (both saves gated on IsOptimal).
    def _on_optimal(opt_model: torch.nn.Module, entry: EpochHistory) -> None:
        _write_cm_table_for_split(
            opt_model, val_loader, result_dir / VALIDATION_CM_TABLE_FILENAME,
            mil_mode=mil_mode,
        )
        _write_cm_table_for_split(
            opt_model, test_loader, result_dir / TEST_CM_TABLE_FILENAME,
            mil_mode=mil_mode,
        )
        val_acc = entry.val.accuracy if entry.val is not None else float("nan")
        print(
            f"  ↳ New optimal at epoch {entry.epoch} (val acc {val_acc:.3f}); "
            "CM_Table_Validation.mat + CM_Table.mat updated."
        )

    epoch_cb = (
        _make_print_epoch_with_curriculum(curriculum)
        if curriculum is not None
        else _print_epoch
    )

    # Optional Weights & Biases logging (off by default). Compose a W&B logger
    # onto the stdout callback so both run each epoch; finish the run below.
    wandb_run = None
    if args.wandb:
        from .training.monitoring.wandb_logger import WandbEpochLogger, init_wandb_run

        wandb_run = init_wandb_run(
            project=args.wandb_project,
            mode=args.wandb_mode,
            config=cast("Mapping[str, Any]", OmegaConf.to_container(cfg, resolve=True)),
        )
        _wandb_logger = WandbEpochLogger(wandb_run)
        _base_epoch_cb = epoch_cb

        def _epoch_cb_with_wandb(history: Any) -> None:
            _base_epoch_cb(history)
            _wandb_logger(history)

        epoch_cb = _epoch_cb_with_wandb

    # Two-stage dispatch: Stage 1 unsupervised pre-training + Stage 2
    # supervised fine-tuning when the variational config asks for it.
    num_epochs_ae = int(cfg.num_epochs_autoencoder)
    is_variational = bool(cfg.get("is_variational", False))
    if num_epochs_ae > 0 and is_variational:
        history = _dispatch_two_stage(
            cfg=cfg, composite=model, composite_optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            result_dir=result_dir, num_features=num_features,
            num_classes_per_dim=num_classes_per_dim,
            loss_weights_dict=loss_weights_dict, class_weights=class_weights,
            initial_priors=initial_priors, epoch_cb=epoch_cb,
            on_optimal=_on_optimal, curriculum=curriculum, device=device,
        )
    else:
        history = fit_supervised(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=int(cfg.num_epochs_full),
            device=device,
            loss_weights=loss_weights_dict,
            checkpoint_dir=result_dir,
            class_weights_per_dim=class_weights,
            grad_clip_norm=float(cfg.gradient_threshold),
            epoch_callback=epoch_cb,
            on_optimal_callback=_on_optimal,
            loss_priors=initial_priors,
            prior_proportion=float(cfg.get("prior_proportion", 0.9)),
            curriculum=curriculum,
            freeze_base_lr=float(cfg.initial_learning_rate),
            rescale_loss_epoch=int(cfg.get("rescale_loss_epoch", 0)),
            confidence_history=initial_confidence_history,
            mil_mode=mil_mode,
            accumulation_max_size=accumulation_max_size,
            loss_type_decoder=str(cfg.get("loss_type_decoder", "MSE")),
        )

    if wandb_run is not None:
        wandb_run.finish()

    final_val_acc = history[-1].val.accuracy if history and history[-1].val else None
    print(f"\nDone. Final validation accuracy: {final_val_acc}")
    print(f"Results written to: {result_dir}")
    return 0


def _dispatch_two_stage(
    *,
    cfg: DictConfig,
    composite: torch.nn.Module,
    composite_optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    result_dir: Path,
    num_features: int,
    num_classes_per_dim: list[int],
    loss_weights_dict: dict[str, float],
    class_weights: Optional[list[torch.Tensor]],
    initial_priors: Optional[LossPriors],
    epoch_cb: Any,
    on_optimal: Any,
    curriculum: Optional[CurriculumBundle],
    device: torch.device,
) -> list[EpochHistory]:
    """Build Stage 1 autoencoder + optimizer and dispatch to fit_two_stage.

    The Stage 2 composite + its optimizer come pre-built from the caller
    (the same construction the single-stage path uses); we only need to
    add the Stage 1 autoencoder and a separate optimizer for it.
    """
    from .models.composite import VariationalComposite
    assert isinstance(composite, VariationalComposite), \
        "Two-stage dispatch requires a VariationalComposite as the Stage 2 model."

    ae_cfg = {
        "in_features": num_features,
        "samples_per_window": int(cfg.get("synthetic_samples_per_window", 1)),
        "num_areas": int(cfg.get("synthetic_num_areas", 1)),
        "hidden_sizes": list(cfg.hidden_sizes),
        "num_classes_per_dim": num_classes_per_dim,        # unused by AE builder
        "classifier_hidden_size": list(cfg.classifier_hidden_size),  # unused
        "transform": str(cfg.model_name),
        "dropout": float(cfg.dropout),
        "want_normalization": bool(cfg.want_normalization),
        "activation": str(cfg.activation),
        "loss_type_decoder": str(cfg.get("loss_type_decoder", "MSE")),
        "stitching_and_fusion_layer": str(cfg.get("stitching_and_fusion_layer", "")),
    }
    autoencoder = build_variational_autoencoder(ae_cfg)
    # Move both networks onto the resolved device before building the Stage 1
    # optimizer. The composite is typically already moved by the caller; moving
    # it again is a no-op, and keeps this function self-contained.
    autoencoder = autoencoder.to(device)
    composite = composite.to(device)
    stage1_optimizer = resolve_optimizer_factory(
        str(cfg.get("optimizer", "ADAM")),
    )(
        autoencoder.parameters(),
        lr=float(cfg.initial_learning_rate),
        weight_decay=float(cfg.l2_factor),
    )

    print(
        f"+++ Two-stage training: Stage 1 ({int(cfg.num_epochs_autoencoder)} "
        f"unsupervised epochs) → handoff → Stage 2 "
        f"({int(cfg.num_epochs_full)} supervised epochs)"
    )
    _, stage2_history = fit_two_stage(
        autoencoder=autoencoder,
        composite=composite,
        stage1_optimizer=stage1_optimizer,
        stage2_optimizer=composite_optimizer,
        stage1_num_epochs=int(cfg.num_epochs_autoencoder),
        stage2_num_epochs=int(cfg.num_epochs_full),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_weights=loss_weights_dict,
        checkpoint_dir=result_dir,
        class_weights_per_dim=class_weights,
        grad_clip_norm=float(cfg.gradient_threshold),
        stage1_epoch_callback=_print_unsupervised_epoch,
        stage2_epoch_callback=epoch_cb,
        on_optimal_callback=on_optimal,
        loss_priors=initial_priors,
        prior_proportion=float(cfg.get("prior_proportion", 0.9)),
        curriculum=curriculum,
        freeze_base_lr=float(cfg.initial_learning_rate),
        rescale_loss_epoch=int(cfg.get("rescale_loss_epoch", 0)),
        loss_type_decoder=str(cfg.get("loss_type_decoder", "MSE")),
    )
    return stage2_history


def _print_unsupervised_epoch(entry: Any) -> None:
    """Stage 1 epoch printer — recon + KL, no accuracy."""
    val_str = (
        f" val_total={entry.val.total_loss:.4f}"
        f" val_recon={entry.val.reconstruction_loss:.4f}"
        f" val_kl={entry.val.kl_loss:.4f}"
        if entry.val is not None
        else ""
    )
    marker = "  *" if entry.is_best else ""
    print(
        f"[Stage 1] Epoch {entry.epoch:03d}  "
        f"train_total={entry.train.total_loss:.4f}  "
        f"train_recon={entry.train.reconstruction_loss:.4f}  "
        f"train_kl={entry.train.kl_loss:.4f}{val_str}{marker}"
    )


def _cmd_check_existing(args: argparse.Namespace) -> int:
    """Implementation of the ``check-existing`` subcommand."""
    cfg = _load_config(args.config_name)
    _apply_cfg_flags(cfg, args)
    result_dir = _resolve_result_dir(cfg, args.output_root)
    found = has_existing_checkpoint(result_dir)
    payload: dict[str, Any] = {
        "result_dir": str(result_dir),
        "has_existing_checkpoint": found,
    }
    print(json.dumps(payload, indent=2))
    return 1 if found else 0


def _load_config(name: str) -> DictConfig:
    """Compose ``base.yaml`` + ``target_milestone/<name>.yaml`` via OmegaConf."""
    base_path = CONFIG_ROOT / "base.yaml"
    target_path = CONFIG_ROOT / "target_milestone" / f"{name}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target config: {target_path}")
    base = OmegaConf.load(base_path)
    target = OmegaConf.load(target_path)
    # OmegaConf merge: target keys win.
    merged = OmegaConf.merge(base, target)
    assert isinstance(merged, DictConfig)
    return merged


def _build_model(
    cfg: DictConfig,
    *,
    in_features: int,
    num_classes_per_dim: list[int],
) -> torch.nn.Module:
    """Build the trainable model for the active milestone.

    Four branches:

    * ``model_name='Logistic Regression'`` — direct
      :class:`MultiHeadClassifier`, no encoder. Milestone A.
    * ``is_variational=True`` — Stochastic VAE composite (encoder →
      bottleneck(2*latent) → sampling → {decoder, classifier}). Returns
      a :class:`VariationalOutput` from ``forward``. Milestone C.
    * Any other ``model_name`` registered as an encoder — build
      ``Encoder + Bottleneck + Classifier`` composite. Milestone B.

    Parameters
    ----------
    cfg
        Resolved config.
    in_features
        Last-axis size of the input data (channel count).
    num_classes_per_dim
        Output classes per dimension.

    Returns
    -------
    torch.nn.Module
        The composed model. For non-variational paths, ``forward(x)``
        returns a list of per-dim logits. For variational, it returns a
        :class:`~neural_data_decoding.models.composite.VariationalOutput`.
    """
    model_name = str(cfg.model_name)

    # Milestone A short-circuit — no encoder.
    if model_name == "Logistic Regression":
        return MultiHeadClassifier(
            in_features=in_features,
            num_classes_per_dim=num_classes_per_dim,
        )

    # Milestone C — Stochastic VAE composite.
    if bool(cfg.get("is_variational", False)):
        variational_cfg = {
            "in_features": in_features,
            "samples_per_window": int(cfg.get("synthetic_samples_per_window", 1)),
            "num_areas": int(cfg.get("synthetic_num_areas", 1)),
            "hidden_sizes": list(cfg.hidden_sizes),
            "num_classes_per_dim": num_classes_per_dim,
            "classifier_hidden_size": list(cfg.classifier_hidden_size),
            "transform": model_name,                     # 'GRU', 'LSTM', or 'Feedforward'
            "dropout": float(cfg.dropout),
            "want_normalization": bool(cfg.want_normalization),
            "activation": str(cfg.activation),
            "loss_type_decoder": str(cfg.get("loss_type_decoder", "MSE")),
            "classifier_dropout": float(cfg.get("classifier_dropout", 0.5)),
            # Confidence heads (Milestone C #7) — built when confidence_type is non-empty.
            "confidence_type": list(cfg.get("confidence_type", [])),
            # Optional stitching+fusion bridge (Milestone CC #3).
            "stitching_and_fusion_layer": str(cfg.get("stitching_and_fusion_layer", "")),
        }
        return build_variational_composite(variational_cfg)

    # Milestone B+ — composite Encoder + Bottleneck + Classifier. The
    # encoder receives 3-D ``(B, W, T*A*C)`` from the composite's
    # FlattenPerWindow (which collapses the per-window dims), so its
    # ``in_features`` is the flat product, not raw ``C``. Matches the
    # variational path's ``_build_ae_core`` convention.
    samples_per_window = int(cfg.get("synthetic_samples_per_window", 1))
    num_areas = int(cfg.get("synthetic_num_areas", 1))
    flat_in_features = in_features * samples_per_window * num_areas
    encoder_cfg = {
        "in_features": flat_in_features,
        "samples_per_window": samples_per_window,
        "num_areas": num_areas,
        "hidden_sizes": list(cfg.hidden_sizes),
        "dropout": float(cfg.dropout),
        "want_normalization": bool(cfg.want_normalization),
        "activation": str(cfg.activation),
        "stride": int(cfg.get("stride", 2)),
    }
    encoder = build_encoder(model_name, encoder_cfg)
    encoder_out = getattr(encoder, "out_features", in_features)

    # Bottleneck: for the tracer bullet, no extra FC — pass through.
    # Milestone C will add the simple-block stack here when
    # bottleneck_hidden_size is set.
    bottleneck_hidden_size = cfg.get("bottleneck_hidden_size", None)
    bottleneck = (
        LinearBottleneck(
            in_features=encoder_out, hidden_size=int(bottleneck_hidden_size)
        )
        if bottleneck_hidden_size
        else PassthroughBottleneck(in_features=encoder_out)
    )

    classifier_cfg = {
        "in_features": bottleneck.out_features,
        "num_classes_per_dim": num_classes_per_dim,
        "classifier_hidden_size": list(cfg.classifier_hidden_size),
    }
    classifier = build_classifier(str(cfg.classifier_name), classifier_cfg)

    return EncoderClassifierComposite(
        encoder=encoder, bottleneck=bottleneck, classifier=classifier
    )


def _fit_pca_if_present(model: torch.nn.Module, train_loader: DataLoader) -> None:
    """Walk the model and fit any :class:`PCAEncoder` on training data.

    PCA encoders (CC.2) hold frozen components — they need to be fit
    once from the training set before any forward pass. This helper
    locates the PCA encoder (if any) anywhere in the model tree and
    calls ``fit_from_dataloader`` on the training loader.

    No-op when the model has no PCA encoder.
    """
    from .models.layers.pca import PCAEncoder
    for module in model.modules():
        if isinstance(module, PCAEncoder) and not module.is_fitted:
            print("  ↳ Fitting PCA encoder on training data...")
            module.fit_from_dataloader(train_loader)
            print(f"  ↳ PCA fit complete (out_features={module.out_features}).")


def _resolve_result_dir(cfg: DictConfig, output_root: Path) -> Path:
    """Compose the MATLAB-parity classifier ``Fold_{N}`` directory for this run.

    The MATLAB results aggregator ``DATA_cggAllNetworkEncoderResults.m``
    walks the deep folder tree generated by
    ``cgg_generateEncoderSubFolders_v3.m`` — Critical Note #15. The
    Python pipeline mirrors that tree exactly so MATLAB-side discovery
    works against Python output unchanged.
    """
    return build_matlab_run_dirs(base_dir=output_root, cfg=cfg).classifier_fold


def _real_data_path_active(cfg: DictConfig) -> bool:
    """``True`` when ``cfg.data_dir`` is resolved (not the Hydra ``???`` sentinel).

    Catches :class:`omegaconf.errors.MissingMandatoryValue` so a config
    that declares ``data_dir: ???`` without an override falls through to
    the synthetic path naturally, with the dataset constructor raising
    a clear error only when the user actually attempts a real-data run.
    """
    try:
        data_dir = cfg.get("data_dir", None)
    except Exception:  # MissingMandatoryValue subclasses ValueError in newer omegaconf
        return False
    return data_dir not in (None, "", "???")


def _resolve_real_session_filter(cfg: DictConfig) -> str | None:
    """Pick the session filter from ``cfg.subset`` for the real-data dataset.

    Mirrors ``cgg_runAutoEncoder.m`` lines 142-152 semantics:

    * ``cfg.subset = true`` → no filter at this layer (the caller is
      expected to set a session via ``--session`` / ``--session-run-idx``
      ahead of time; if neither was passed the whole directory is used).
    * ``cfg.subset = false`` or ``"All"`` → no filter (every trial).
    * ``cfg.subset = "<SessionName>"`` → that single session.
    """
    subset = cfg.get("subset", True)
    if isinstance(subset, bool):
        return None
    s = str(subset)
    if s == "" or s.lower() == "all":
        return None
    return s


def _build_real_data_split(
    cfg: DictConfig,
    *,
    train_load_schedule: Optional[Any] = None,
) -> tuple[MatFileTrialDataset, MatFileTrialDataset, MatFileTrialDataset]:
    """Build a (train, val, test) triple of :class:`MatFileTrialDataset`.

    The first dataset reads every trial in ``cfg.data_dir`` to discover
    sessions and build the per-dim class mapping. That same mapping is
    reused for val/test so the class indices line up across splits.

    Splits are deterministic by ``trial_id % 5`` (matching the MATLAB
    K-fold cadence for the typical num_folds=5 case): residues 0 and 1
    → val and test, the rest → train. With only one trial (the smoke
    fixture), the single trial appears in **all three** splits so the
    pipeline can exercise its full loop without crashing.
    """
    data_dir = str(cfg.data_dir)
    target_dir = str(cfg.target_dir)
    data_pattern = str(cfg.get("data_pattern", "Decision_Data_*.mat"))
    target_pattern = str(cfg.get("target_pattern", "Target_*.mat"))

    target_type = str(cfg.get("target", "Dimension"))
    feature_dims = list(cfg.get("feature_dimensions", [0, 1, 2, 4]))
    dim_indices = list(cfg.get("dimension_indices", [0, 1, 2, 3]))

    starting_idx = int(cfg.get("starting_idx", 0) or 0)
    ending_raw = cfg.get("ending_idx", None)
    ending_idx = None if ending_raw in (None, "", "All", "all") else int(ending_raw)
    sep_raw = cfg.get("start_end_percent", (None, None))
    if sep_raw is None:
        start_end_percent: tuple[float | None, float | None] = (None, None)
    else:
        sep_list = list(sep_raw)
        start_end_percent = (
            None if sep_list[0] is None else float(sep_list[0]),
            None if (len(sep_list) < 2 or sep_list[1] is None) else float(sep_list[1]),
        )

    fold_seed = int(cfg.fold) * 17
    session_filter = _resolve_real_session_filter(cfg)

    common_kwargs: dict[str, Any] = {
        "data_dir": data_dir,
        "target_dir": target_dir,
        "data_pattern": data_pattern,
        "target_pattern": target_pattern,
        "data_width": int(cfg.data_width),
        "window_stride": int(cfg.window_stride),
        "target_type": target_type,
        "feature_dimensions": feature_dims,
        "dimension_indices": dim_indices,
        "starting_idx": starting_idx,
        "ending_idx": ending_idx,
        "start_end_percent": start_end_percent,
        "session_filter": session_filter,
        "sampling_frequency": float(cfg.get("sampling_frequency", 1000.0)),
        "want_separate_time_shift": bool(cfg.get("want_separate_time_shift", True)),
    }

    # Pass 1: load the full set of trials to discover the class mapping
    # and the trial-id partition. Augmentation is OFF on this pass so
    # the mapping isn't biased by noise.
    discovery_ds = MatFileTrialDataset(**common_kwargs)
    class_mapping = [dict(m) for m in discovery_ds.class_mapping_per_dim]
    trial_ids = list(discovery_ds.trial_ids)

    train_ids, val_ids, test_ids = _partition_trial_ids(
        trial_ids,
        val_fraction=float(cfg.get("real_data_validation_fraction", 0.2)),
        test_fraction=float(cfg.get("real_data_test_fraction", 0.2)),
        seed=fold_seed,
    )

    def _build_with_ids(
        ids: set[int], *, schedule: Optional[Any], aug_seed: int
    ) -> MatFileTrialDataset:
        ds = MatFileTrialDataset(
            **common_kwargs,
            class_mapping_per_dim=class_mapping,
            load_schedule=schedule,
            augmentation_seed=aug_seed,
        )
        # Filter to the chosen trial IDs in-place. We rebuild the
        # private bookkeeping arrays so __len__ / __getitem__ /
        # session_ids stay consistent.
        keep_idx = [i for i, t in enumerate(ds._trials) if t.trial_id in ids]  # noqa: SLF001
        if not keep_idx:
            # Empty split — fall back to the full set so smoke tests on
            # a single-trial fixture still work.
            return ds
        ds._trials = [ds._trials[i] for i in keep_idx]  # noqa: SLF001
        ds._session_names = [ds._session_names[i] for i in keep_idx]  # noqa: SLF001
        ds._session_ids = ds._session_ids[keep_idx]  # noqa: SLF001
        ds._labels = ds._labels[keep_idx]  # noqa: SLF001
        if ds._preloaded is not None:  # noqa: SLF001
            ds._preloaded = [ds._preloaded[i] for i in keep_idx]  # noqa: SLF001
        return ds

    train_ds = _build_with_ids(
        train_ids, schedule=train_load_schedule, aug_seed=fold_seed + 1000
    )
    val_ds = _build_with_ids(val_ids, schedule=None, aug_seed=fold_seed + 2000)
    test_ds = _build_with_ids(test_ids, schedule=None, aug_seed=fold_seed + 3000)
    return train_ds, val_ds, test_ds


def _partition_trial_ids(
    trial_ids: list[int],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[int], set[int], set[int]]:
    """Deterministic 60/20/20-style partition of trial IDs into train/val/test.

    For very small corpora (< 5 trials), each split degenerates to the
    same single-trial set — that's the smoke-test case. For production
    use the dataset will have hundreds of trials so the random shuffle
    behaves like a true partition.
    """
    import numpy as np  # local import to avoid module-level dependency

    if not trial_ids:
        return set(), set(), set()
    rng = np.random.default_rng(seed)
    shuffled = list(trial_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    if n < 3:
        # Degenerate corpus: every split gets all trials.
        full = set(trial_ids)
        return full, full, full
    val_set = set(shuffled[:n_val])
    test_set = set(shuffled[n_val : n_val + n_test])
    train_set = set(shuffled[n_val + n_test :])
    if not train_set:
        # Round-off pathology — guarantee at least one trial in train.
        train_set = {shuffled[-1]}
    return train_set, val_set, test_set


def _build_synthetic_split(
    cfg: DictConfig,
    *,
    train_load_schedule: Optional[Any] = None,
) -> tuple[SyntheticTrialDataset, SyntheticTrialDataset, SyntheticTrialDataset]:
    """Build a (train, val, test) triple of :class:`SyntheticTrialDataset`.

    Three disjoint splits, each with its own seed (so the validation and
    test sets are independent of training AND each other). Matches the
    MATLAB pipeline's distinction:

    * **val** drives the Optimal-snapshot model selection during training
      (each epoch's CM_Table_Validation.mat is written from here).
    * **test** is run once at the end against the loaded Optimal weights;
      the resulting CM_Table.mat is what downstream analysis aggregates
      for final reported results.

    The synthetic dataset's trial-count knob is per-session, so we scale
    it by the respective fraction; both val and test fall back to a
    sensible minimum of 1 trial/session.
    """
    fold_seed = int(cfg.fold) * 17  # deterministic, but distinct across folds
    train_ds = SyntheticTrialDataset(
        num_sessions=int(cfg.synthetic_num_sessions),
        trials_per_session=int(cfg.synthetic_trials_per_session),
        num_samples=int(cfg.synthetic_num_samples),
        num_features=int(cfg.synthetic_num_features),
        num_classes_per_dim=list(cfg.synthetic_num_classes_per_dim),
        samples_per_window=int(cfg.get("synthetic_samples_per_window", 1)),
        num_areas=int(cfg.get("synthetic_num_areas", 1)),
        signal_strength=float(cfg.synthetic_signal_strength),
        seed=fold_seed,
        load_schedule=train_load_schedule,
        augmentation_seed=fold_seed + 1000,
    )
    val_trials = max(
        1,
        int(int(cfg.synthetic_trials_per_session)
            * float(cfg.synthetic_validation_fraction)),
    )
    val_ds = SyntheticTrialDataset(
        num_sessions=int(cfg.synthetic_num_sessions),
        trials_per_session=val_trials,
        num_samples=int(cfg.synthetic_num_samples),
        num_features=int(cfg.synthetic_num_features),
        num_classes_per_dim=list(cfg.synthetic_num_classes_per_dim),
        samples_per_window=int(cfg.get("synthetic_samples_per_window", 1)),
        num_areas=int(cfg.get("synthetic_num_areas", 1)),
        signal_strength=float(cfg.synthetic_signal_strength),
        seed=fold_seed + 1,
    )
    test_fraction = float(cfg.get("synthetic_test_fraction", 0.2))
    test_trials = max(
        1, int(int(cfg.synthetic_trials_per_session) * test_fraction)
    )
    test_ds = SyntheticTrialDataset(
        num_sessions=int(cfg.synthetic_num_sessions),
        trials_per_session=test_trials,
        num_samples=int(cfg.synthetic_num_samples),
        num_features=int(cfg.synthetic_num_features),
        num_classes_per_dim=list(cfg.synthetic_num_classes_per_dim),
        samples_per_window=int(cfg.get("synthetic_samples_per_window", 1)),
        num_areas=int(cfg.get("synthetic_num_areas", 1)),
        signal_strength=float(cfg.synthetic_signal_strength),
        seed=fold_seed + 2,
    )
    return train_ds, val_ds, test_ds


def _build_curriculum(cfg: DictConfig) -> Optional[CurriculumBundle]:
    """Resolve ``cfg.dynamic_parameter_set`` to a :class:`CurriculumBundle`.

    Reads the regime name from the config (e.g.
    ``"Soft Three-Stage Curriculum - Shortened"``) and looks up the matching
    YAML preset in ``configs/schedule/``. Bases for each schedule come
    from the static config fields (``weight_*``, ``std_*``).

    Returns ``None`` only if the regime string is empty or unspecified —
    not for ``"None"``/``"No Dynamic Parameters"`` (those resolve to a
    valid bundle whose schedules just have no waypoints, so the bases
    propagate untouched).
    """
    regime = str(cfg.get("dynamic_parameter_set", "")).strip()
    if not regime:
        return None

    base_loads = {
        "std_channel_offset": float(cfg.get("std_channel_offset", float("nan"))),
        "std_white_noise":    float(cfg.get("std_white_noise",    float("nan"))),
        "std_random_walk":    float(cfg.get("std_random_walk",    float("nan"))),
        "std_time_shift":     float(cfg.get("std_time_shift",     float("nan"))),
    }
    base_weights = {
        "reconstruction":    float(cfg.get("weight_reconstruction", float("nan"))),
        "kl":                float(cfg.get("weight_kl",             float("nan"))),
        "classification":    float(cfg.get("weight_classification", float("nan"))),
        "confidence":        float(cfg.get("weight_confidence",     0.0)),
        "offset_and_scale":  float(cfg.get("weight_offset_and_scale", 0.0)),
    }
    base_freezes = {"encoder": 1.0, "decoder": 1.0, "classifier": 1.0}

    # Legacy KL base anneal (cgg_annealWeight wrapped around the KL weight,
    # applied BEFORE the dynamic schedule's multiply). Active iff the
    # config provides a positive ramp length; otherwise the KL weight goes
    # straight to its base value with no warmup.
    kl_anneal: Optional[KLBaseAnneal] = None
    kl_ramp = int(cfg.get("weight_epoch_ramp", 0))
    if kl_ramp > 0:
        kl_anneal = KLBaseAnneal(
            initial_weight=float(cfg.get("weight_kl", 0.0)),
            delay_epoch=int(cfg.get("weight_delay_epoch", 0)),
            epoch_ramp=kl_ramp,
        )

    bundle = load_curriculum_by_name(
        regime,
        base_loads=base_loads,
        base_weights=base_weights,
        base_freezes=base_freezes,
    )
    bundle.kl_anneal = kl_anneal
    return bundle


def _build_optimizer(
    cfg: DictConfig,
    model: torch.nn.Module,
    curriculum: Optional[CurriculumBundle],
) -> torch.optim.Optimizer:
    """Build the optimizer, using per-module groups when freeze is active.

    The freeze applier requires named param groups. When the curriculum's
    freeze schedule has waypoints for any submodule, build per-module
    groups; otherwise stick with the simpler single-group form.

    The optimizer choice is driven by ``cfg.optimizer`` (case-insensitive,
    ``"ADAM"`` or ``"SGDM"``); see
    :func:`~neural_data_decoding.training.freezing.resolve_optimizer_factory`.
    Defaults to ``"ADAM"`` when the field is absent.
    """
    lr = float(cfg.initial_learning_rate)
    wd = float(cfg.l2_factor)
    optimizer_factory = resolve_optimizer_factory(str(cfg.get("optimizer", "ADAM")))

    needs_groups = curriculum is not None and any(
        len(curriculum.freeze[name].epoch_points) > 0
        for name in curriculum.freeze
    )
    if not needs_groups:
        return optimizer_factory(
            model.parameters(), lr=lr, weight_decay=wd,
        )

    # Per-module groups. Skip any submodule the model doesn't expose.
    module_groups: dict[str, torch.nn.Module] = {}
    for name in ("encoder", "decoder", "classifier"):
        sub = getattr(model, name, None)
        if isinstance(sub, torch.nn.Module):
            module_groups[name] = sub

    if not module_groups:
        # Caller passed a freeze schedule but the model has no exposed
        # submodules to attach groups to (e.g., MultiHeadClassifier).
        # Fall back to single-group; freeze just won't take effect.
        return optimizer_factory(
            model.parameters(), lr=lr, weight_decay=wd,
        )

    return build_optimizer_with_module_groups(
        module_groups, initial_lr=lr, weight_decay=wd,
        optimizer_factory=optimizer_factory,
    )


def _print_epoch(history_entry: Any) -> None:
    """Default epoch callback — single-line stdout update."""
    val_str = (
        f" val_acc={history_entry.val.accuracy:.3f}"
        f" val_loss={history_entry.val.classification_loss:.4f}"
        if history_entry.val is not None
        else ""
    )
    marker = "  *" if history_entry.is_best else ""
    print(
        f"Epoch {history_entry.epoch:03d}  "
        f"train_loss={history_entry.train.classification_loss:.4f}  "
        f"train_acc={history_entry.train.accuracy:.3f}{val_str}{marker}"
    )


def _make_print_epoch_with_curriculum(
    curriculum: CurriculumBundle,
) -> Any:
    """Wrap :func:`_print_epoch` to also dump a one-line curriculum snapshot.

    Useful for the Milestone C #5 smoke test — operators can confirm
    augmentation magnitudes, loss weights, and freeze factors are ticking
    across epochs without digging into checkpoints.
    """
    def _cb(entry: Any) -> None:
        _print_epoch(entry)
        weights = ", ".join(
            f"{n}={curriculum.weight.current(n):.3g}"
            for n in ("classification", "kl", "reconstruction")
            if n in curriculum.weight
        )
        loads = ", ".join(
            f"{n}={curriculum.load.current(n):.3g}"
            for n in ("std_white_noise", "std_channel_offset")
            if n in curriculum.load
        )
        freezes = ", ".join(
            f"{n}={curriculum.freeze.current(n):.3g}"
            for n in ("encoder", "decoder", "classifier")
            if n in curriculum.freeze
        )
        print(f"   ↳ weights[{weights}]  loads[{loads}]  freeze[{freezes}]")

    return _cb


def _write_cm_table_for_split(
    model: torch.nn.Module,
    loader: DataLoader,
    output_path: Path,
    *,
    mil_mode: bool = False,
) -> None:
    """Run ``model`` over ``loader`` and persist a CM_Table at ``output_path``.

    Used both for the validation CM_Table (model state at end of training)
    and the test CM_Table (model state restored from the Optimal snapshot).
    Selects between :class:`VariationalOutput` and a plain list[Tensor] of
    logits transparently so the same writer works for Milestones A, B,
    and C.

    Two kinds of prediction columns are written:

    * **Per-window predictions** — argmax of the per-window logits per dim,
      one ``Window_k`` column for each window on the model's ``W`` axis
      (matching MATLAB's ``Window_1 … Window_K`` layout). A non-sequence
      classifier (Milestone A logistic) has no window axis, so it emits a
      single ``Window_1`` column.
    * **Aggregation prediction** — argmax of the normalized aggregated
      probability distribution per dim (matches MATLAB's
      ``Aggregation_Prediction`` column in
      ``cgg_getClassifierOutputsFromProbabilities.m``). Computed
      regardless of ``mil_mode``: non-MIL averages the per-window softmax
      (uniform prior over windows); MIL marginalizes the joint softmax.
    """
    import numpy as np

    from .models.composite import VariationalOutput

    model.eval()
    # Per-batch prediction chunks, each shaped (B, W, D): argmax class per
    # (trial, window, dimension). Concatenated over batches to (N, W, D), then
    # sliced into one Window_k column per window for the MATLAB writer.
    all_pred_chunks: list[np.ndarray] = []
    all_aggregation_predictions: list[list[int]] = []
    all_targets: list[list[int]] = []
    all_trial_ids: list[int] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"])
            # Variational composite returns VariationalOutput; classifier-only
            # composites return a list[Tensor] of per-dim logits directly.
            logits_per_dim = out.logits if isinstance(out, VariationalOutput) else out
            is_sequence = all(L.ndim == 3 for L in logits_per_dim)

            # Per-window prediction: argmax over the class axis for every window
            # on dim 1 (the W axis). A non-sequence classifier has no window
            # axis, so treat it as a single window.
            per_dim_pred = [
                (logits.argmax(dim=-1) if logits.ndim == 3
                 else logits.argmax(dim=-1).unsqueeze(1))     # (B, W) or (B, 1)
                for logits in logits_per_dim
            ]
            preds = torch.stack(per_dim_pred, dim=-1)         # (B, W, D)
            all_pred_chunks.append(preds.cpu().numpy())

            # Aggregation prediction: per-trial uniform-prior aggregate
            # (or MIL marginal) for sequence outputs; for a non-sequence
            # classifier the aggregate equals the single window prediction.
            if is_sequence:
                aggregated_probs = aggregate_classifier_predictions(
                    list(logits_per_dim), mil_mode=mil_mode,
                )
                per_trial_agg: list[list[int]] = []
                for probs in aggregated_probs:               # (B, K_d) per dim
                    pred = probs.argmax(dim=-1).tolist()
                    if not per_trial_agg:
                        per_trial_agg = [[p] for p in pred]
                    else:
                        for i, p in enumerate(pred):
                            per_trial_agg[i].append(p)
                all_aggregation_predictions.extend(per_trial_agg)
            else:
                all_aggregation_predictions.extend(preds[:, 0, :].tolist())

            all_targets.extend(batch["targets"].tolist())
            all_trial_ids.extend(m["trial_id"] for m in batch["metadata"])

    all_preds = np.concatenate(all_pred_chunks, axis=0)       # (N, W, D)
    num_windows = int(all_preds.shape[1])

    data_numbers = np.array(all_trial_ids, dtype=np.int32) + 1  # MATLAB 1-indexed
    true_values = np.array(all_targets, dtype=np.float64)
    window_predictions = [
        all_preds[:, w, :].astype(np.float64) for w in range(num_windows)
    ]
    aggregation = np.array(all_aggregation_predictions, dtype=np.float64)
    write_cm_table_mat(
        output_path,
        data_numbers=data_numbers,
        true_values=true_values,
        window_predictions=window_predictions,
        aggregation_prediction=aggregation,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
