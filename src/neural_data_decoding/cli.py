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
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from . import __version__
from .data.dataset import SyntheticTrialDataset, collate_trials
from .interop import (
    ENCODING_PARAMETERS_FILENAME,
    TEST_CM_TABLE_FILENAME,
    VALIDATION_CM_TABLE_FILENAME,
    build_result_dir,
    write_cm_table_mat,
    write_encoding_parameters_yaml,
)
import neural_data_decoding.models  # noqa: F401 — triggers architecture registrations
from .models.bottleneck import LinearBottleneck, PassthroughBottleneck
from .models.classifier import MultiHeadClassifier
from .models.composite import EncoderClassifierComposite, build_variational_composite
from .models.registry import build_classifier, build_encoder
from .training.checkpoint import has_existing_checkpoint
from .training.lifecycle import EpochHistory, fit_supervised
from .training.losses.multi_objective import LossPriors
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

    # Build the synthetic datasets (train/val/test), model, optimizer, then fit.
    train_ds, val_ds, test_ds = _build_synthetic_split(cfg)
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
    num_features = int(cfg.synthetic_num_features)
    num_classes_per_dim = list(cfg.synthetic_num_classes_per_dim)
    model = _build_model(
        cfg,
        in_features=num_features,
        num_classes_per_dim=num_classes_per_dim,
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

    # For variational configs, expose all per-component weights AND enable
    # EMA prior normalization. For Logistic/non-variational, only classification
    # is active, so the simpler weight dict suffices.
    is_variational = bool(cfg.get("is_variational", False))
    if is_variational:
        loss_weights_dict: dict[str, float] = {
            "classification": float(cfg.weight_classification),
            "reconstruction": float(cfg.get("weight_reconstruction", 1.0)),
            "kl": float(cfg.get("weight_kl", 1.0)),
        }
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
            opt_model, val_loader, result_dir / VALIDATION_CM_TABLE_FILENAME
        )
        _write_cm_table_for_split(
            opt_model, test_loader, result_dir / TEST_CM_TABLE_FILENAME
        )
        val_acc = entry.val.accuracy if entry.val is not None else float("nan")
        print(
            f"  ↳ New optimal at epoch {entry.epoch} (val acc {val_acc:.3f}); "
            "CM_Table_Validation.mat + CM_Table.mat updated."
        )

    history = fit_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=int(cfg.num_epochs_full),
        device=torch.device("cpu"),
        loss_weights=loss_weights_dict,
        checkpoint_dir=result_dir,
        class_weights_per_dim=class_weights,
        grad_clip_norm=float(cfg.gradient_threshold),
        epoch_callback=_print_epoch,
        on_optimal_callback=_on_optimal,
        loss_priors=initial_priors,
        prior_proportion=float(cfg.get("prior_proportion", 0.9)),
    )

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
            "hidden_sizes": list(cfg.hidden_sizes),
            "num_classes_per_dim": num_classes_per_dim,
            "classifier_hidden_size": list(cfg.classifier_hidden_size),
            "transform": model_name,                     # 'GRU', 'LSTM', or 'Feedforward'
            "dropout": float(cfg.dropout),
            "want_normalization": bool(cfg.want_normalization),
            "activation": str(cfg.activation),
            "loss_type_decoder": str(cfg.get("loss_type_decoder", "MSE")),
            "classifier_dropout": float(cfg.get("classifier_dropout", 0.5)),
        }
        return build_variational_composite(variational_cfg)

    # Milestone B+ — composite Encoder + Bottleneck + Classifier.
    encoder_cfg = {
        "in_features": in_features,
        "hidden_sizes": list(cfg.hidden_sizes),
        "dropout": float(cfg.dropout),
        "want_normalization": bool(cfg.want_normalization),
        "activation": str(cfg.activation),
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
        signal_strength=float(cfg.synthetic_signal_strength),
        seed=fold_seed,
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
        signal_strength=float(cfg.synthetic_signal_strength),
        seed=fold_seed + 2,
    )
    return train_ds, val_ds, test_ds


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


def _write_cm_table_for_split(
    model: torch.nn.Module,
    loader: DataLoader,
    output_path: Path,
) -> None:
    """Run ``model`` over ``loader`` and persist a CM_Table at ``output_path``.

    Used both for the validation CM_Table (model state at end of training)
    and the test CM_Table (model state restored from the Optimal snapshot).
    Selects between :class:`VariationalOutput` and a plain list[Tensor] of
    logits transparently so the same writer works for Milestones A, B,
    and C.
    """
    import numpy as np

    from .models.composite import VariationalOutput

    model.eval()
    all_predictions: list[list[int]] = []
    all_targets: list[list[int]] = []
    all_trial_ids: list[int] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"])
            # Variational composite returns VariationalOutput; classifier-only
            # composites return a list[Tensor] of per-dim logits directly.
            logits_per_dim = out.logits if isinstance(out, VariationalOutput) else out
            per_trial_pred: list[list[int]] = []
            for _d, logits in enumerate(logits_per_dim):
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
        output_path,
        data_numbers=data_numbers,
        true_values=true_values,
        window_predictions=[window],
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
