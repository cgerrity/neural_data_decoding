"""Weights & Biases epoch logger.

W&B is a declared dependency but is **off by default** — the pipeline logs to
stdout unless ``train --wandb`` is passed. This module provides the run
initializer and an :data:`~neural_data_decoding.training.lifecycle.EpochCallback`
that streams per-epoch metrics to a W&B run, composing with the existing stdout
callback rather than replacing it.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, cast

from neural_data_decoding.training.lifecycle import EpochHistory


def epoch_metrics_dict(history: EpochHistory) -> dict[str, Any]:
    """Flatten an :class:`EpochHistory` into a W&B-loggable metrics dict.

    Parameters
    ----------
    history
        One epoch's train + validation metrics and the ``is_best`` flag.

    Returns
    -------
    dict
        ``epoch``, ``is_optimal``, the ``train/*`` metrics, and (when a
        validation split ran) the ``val/*`` metrics.
    """
    metrics: dict[str, Any] = {
        "epoch": history.epoch,
        "is_optimal": int(history.is_best),
        "train/loss": history.train.total_loss,
        "train/classification_loss": history.train.classification_loss,
        "train/accuracy": history.train.accuracy,
    }
    if history.val is not None:
        metrics["val/loss"] = history.val.total_loss
        metrics["val/accuracy"] = history.val.accuracy
    return metrics


def init_wandb_run(
    *,
    project: str,
    mode: str = "online",
    config: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Initialize a Weights & Biases run.

    Parameters
    ----------
    project
        The W&B project name.
    mode
        ``"online"`` (default), ``"offline"`` (log locally, sync later), or
        ``"disabled"`` (no-op — safe for CI and tests).
    config
        Optional run config (the resolved training config) recorded on the run.

    Returns
    -------
    Any
        The W&B run object.
    """
    import wandb

    return wandb.init(
        project=project,
        mode=cast(Any, mode),  # validated by the CLI's --wandb-mode choices
        config=dict(config) if config is not None else None,
        reinit=True,
    )


class WandbEpochLogger:
    """An ``EpochCallback`` that logs each epoch's metrics to a W&B run.

    Parameters
    ----------
    run
        A W&B run object (typically from :func:`init_wandb_run`).
    """

    def __init__(self, run: Any) -> None:
        self._run = run

    def __call__(self, history: EpochHistory) -> None:
        """Log this epoch's metrics; record best-val accuracy in the run summary."""
        self._run.log(epoch_metrics_dict(history))
        if history.is_best and history.val is not None:
            self._run.summary["best_val_accuracy"] = history.val.accuracy
