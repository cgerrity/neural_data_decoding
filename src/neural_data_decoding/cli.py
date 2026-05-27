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
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from . import __version__
from .data.dataset import SyntheticTrialDataset, collate_trials
from .interop import (
    ENCODING_PARAMETERS_FILENAME,
    VALIDATION_CM_TABLE_FILENAME,
    build_result_dir,
    write_cm_table_mat,
    write_encoding_parameters_yaml,
)
from .models.classifier import MultiHeadClassifier
from .training.checkpoint import has_existing_checkpoint
from .training.lifecycle import fit_supervised
from .training.losses.classification import inverse_frequency_class_weights


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

    check_p = sub.add_parser(
        "check-existing",
        help="Resolve the result directory and report whether any checkpoint files would be clobbered.",
    )
    _add_common_args(check_p)

    sub.add_parser(
        "sweep",
        help="(stub — Milestone D) Launch a hyperparameter sweep via submitit or Ray Tune.",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "check-existing":
        return _cmd_check_existing(args)
    if args.command == "sweep":
        print("sweep is not yet implemented (Milestone D).")
        return 1

    parser.print_help()
    return 0


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Attach the ``--config-name`` / ``--fold`` / ``--output-root`` triple."""
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


def _cmd_train(args: argparse.Namespace) -> int:
    """Implementation of the ``train`` subcommand."""
    cfg = _load_config(args.config_name)
    if args.fold is not None:
        cfg.fold = int(args.fold)

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
    schema = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(schema, dict)
    # Drop Hydra-internal keys that shouldn't appear in the MATLAB-facing YAML.
    schema = {k: v for k, v in schema.items() if k != "defaults"}
    write_encoding_parameters_yaml(
        result_dir / ENCODING_PARAMETERS_FILENAME,
        run_config=schema,
        schema_template=schema,
    )

    # Build the synthetic dataset, model, optimizer, then fit.
    train_ds, val_ds = _build_synthetic_split(cfg)
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

    # Logistic Regression for Milestone A: classifier consumes raw features
    # directly with no encoder. in_features = num_features.
    num_features = int(cfg.synthetic_num_features)
    num_classes_per_dim = list(cfg.synthetic_num_classes_per_dim)
    model = MultiHeadClassifier(
        in_features=num_features, num_classes_per_dim=num_classes_per_dim
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.initial_learning_rate),
        weight_decay=float(cfg.l2_factor),
    )

    train_labels = torch.from_numpy(train_ds._labels).long()  # noqa: SLF001
    class_weights = (
        inverse_frequency_class_weights(train_labels, num_classes_per_dim)
        if str(cfg.weighted_loss).lower() == "inverse"
        else None
    )

    history = fit_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=int(cfg.num_epochs_full),
        device=torch.device("cpu"),
        loss_weights={"classification": float(cfg.weight_classification)},
        checkpoint_dir=result_dir,
        class_weights_per_dim=class_weights,
        grad_clip_norm=float(cfg.gradient_threshold),
        epoch_callback=_print_epoch,
    )

    # Write a minimal validation CM_Table on the held-out split.
    _write_validation_cm_table(model, val_ds, val_loader, result_dir)

    final_val_acc = history[-1].val.accuracy if history and history[-1].val else None
    print(f"\nDone. Final validation accuracy: {final_val_acc}")
    print(f"Results written to: {result_dir}")
    return 0


def _cmd_check_existing(args: argparse.Namespace) -> int:
    """Implementation of the ``check-existing`` subcommand."""
    cfg = _load_config(args.config_name)
    if args.fold is not None:
        cfg.fold = int(args.fold)
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


def _resolve_result_dir(cfg: DictConfig, output_root: Path) -> Path:
    """Compose the deterministic result directory for this run."""
    identifying = {
        "weight_classification": float(cfg.weight_classification),
        "initial_learning_rate": float(cfg.initial_learning_rate),
        "mini_batch_size": int(cfg.mini_batch_size),
        "num_epochs_full": int(cfg.num_epochs_full),
        "weighted_loss": str(cfg.weighted_loss),
    }
    return build_result_dir(
        base_dir=output_root,
        epoch=str(cfg.epoch),
        target=str(cfg.target),
        model_name=str(cfg.model_name),
        fold=int(cfg.fold),
        identifying_config=identifying,
    )


def _build_synthetic_split(
    cfg: DictConfig,
) -> tuple[SyntheticTrialDataset, SyntheticTrialDataset]:
    """Build a train/val pair of :class:`SyntheticTrialDataset` for the tracer bullet."""
    fold_seed = int(cfg.fold) * 17  # deterministic, but distinct across folds
    train_ds = SyntheticTrialDataset(
        num_sessions=int(cfg.synthetic_num_sessions),
        trials_per_session=int(cfg.synthetic_trials_per_session),
        num_samples=int(cfg.synthetic_num_samples),
        num_features=int(cfg.synthetic_num_features),
        num_classes_per_dim=list(cfg.synthetic_num_classes_per_dim),
        signal_strength=float(cfg.synthetic_signal_strength),
        seed=fold_seed,
    )
    val_ds = SyntheticTrialDataset(
        num_sessions=int(cfg.synthetic_num_sessions),
        trials_per_session=max(
            1,
            int(
                int(cfg.synthetic_trials_per_session)
                * float(cfg.synthetic_validation_fraction)
            ),
        ),
        num_samples=int(cfg.synthetic_num_samples),
        num_features=int(cfg.synthetic_num_features),
        num_classes_per_dim=list(cfg.synthetic_num_classes_per_dim),
        signal_strength=float(cfg.synthetic_signal_strength),
        seed=fold_seed + 1,
    )
    return train_ds, val_ds


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


def _write_validation_cm_table(
    model: MultiHeadClassifier,
    val_ds: SyntheticTrialDataset,
    val_loader: DataLoader,
    result_dir: Path,
) -> None:
    """Run the trained model on the val split and persist the CM_Table."""
    import numpy as np

    model.eval()
    all_predictions: list[list[int]] = []
    all_targets: list[list[int]] = []
    all_trial_ids: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            logits_per_dim = model(batch["x"])
            per_trial_pred: list[list[int]] = []
            for d, logits in enumerate(logits_per_dim):
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                pred = logits.argmax(dim=-1).tolist()
                if not per_trial_pred:
                    per_trial_pred = [[p] for p in pred]
                else:
                    for i, p in enumerate(pred):
                        per_trial_pred[i].append(p)
            all_predictions.extend(per_trial_pred)
            all_targets.extend(batch["targets"].tolist())
            all_trial_ids.extend(m["trial_id"] for m in batch["metadata"])

    data_numbers = np.array(all_trial_ids, dtype=np.int32) + 1  # MATLAB 1-indexed
    true_values = np.array(all_targets, dtype=np.float64)
    window = np.array(all_predictions, dtype=np.float64)
    write_cm_table_mat(
        result_dir / VALIDATION_CM_TABLE_FILENAME,
        data_numbers=data_numbers,
        true_values=true_values,
        window_predictions=[window],
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
