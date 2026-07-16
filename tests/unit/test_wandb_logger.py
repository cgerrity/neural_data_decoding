"""Weights & Biases epoch-logger tests (run in disabled mode — no network)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

from neural_data_decoding.cli import main  # noqa: E402
from neural_data_decoding.training.lifecycle import EpochHistory, EpochMetrics  # noqa: E402
from neural_data_decoding.training.monitoring.wandb_logger import (  # noqa: E402
    WandbEpochLogger,
    epoch_metrics_dict,
    init_wandb_run,
)


def _history(epoch: int, val_acc: float, is_best: bool, with_val: bool = True) -> EpochHistory:
    """Build a minimal EpochHistory for logger tests."""
    train = EpochMetrics(
        total_loss=1.0, classification_loss=0.5, accuracy=0.4,
        num_iterations=3, num_trials=30,
    )
    val = (
        EpochMetrics(
            total_loss=0.9, classification_loss=0.4, accuracy=val_acc,
            num_iterations=1, num_trials=10,
        )
        if with_val
        else None
    )
    return EpochHistory(epoch=epoch, train=train, val=val, is_best=is_best)


def test_epoch_metrics_dict_includes_train_and_val() -> None:
    """The flattened dict carries train/* and (when present) val/* keys."""
    m = epoch_metrics_dict(_history(1, 0.6, is_best=True))
    assert m["epoch"] == 1
    assert m["is_optimal"] == 1
    assert m["train/accuracy"] == 0.4
    assert m["val/accuracy"] == 0.6


def test_epoch_metrics_dict_omits_val_when_absent() -> None:
    """With no validation split, no ``val/*`` keys are emitted."""
    m = epoch_metrics_dict(_history(2, 0.0, is_best=False, with_val=False))
    assert not any(k.startswith("val/") for k in m)


def test_wandb_logger_disabled_mode_logs_without_error() -> None:
    """The logger runs against a disabled W&B run without touching the network."""
    run = init_wandb_run(project="ndd-test", mode="disabled")
    logger = WandbEpochLogger(run)
    logger(_history(1, 0.5, is_best=True))   # exercises the summary branch
    logger(_history(2, 0.4, is_best=False))
    run.finish()


def test_train_with_wandb_disabled_runs_end_to_end(tmp_path: Path) -> None:
    """`train --wandb --wandb-mode disabled` wires the logger without crashing."""
    rc = main([
        "train",
        "--config-name", "A_logistic_synthetic",
        "--fold", "1",
        "--output-root", str(tmp_path),
        "--wandb",
        "--wandb-mode", "disabled",
    ])
    assert rc == 0
