"""Training engine: lifecycle, loop, losses, schedules, monitoring, checkpointing."""

from neural_data_decoding.training.checkpoint import (
    CURRENT_CHECKPOINT_FILENAME,
    OPTIMAL_CHECKPOINT_FILENAME,
    CheckpointState,
    has_existing_checkpoint,
    load_current_checkpoint,
    load_optimal_checkpoint,
    save_current_checkpoint,
    save_optimal_checkpoint,
)
from neural_data_decoding.training.lifecycle import (
    EpochHistory,
    fit_supervised,
)
from neural_data_decoding.training.loop import (
    EpochMetrics,
    train_one_epoch,
    validate,
)

__all__ = [
    "CURRENT_CHECKPOINT_FILENAME",
    "CheckpointState",
    "EpochHistory",
    "EpochMetrics",
    "OPTIMAL_CHECKPOINT_FILENAME",
    "fit_supervised",
    "has_existing_checkpoint",
    "load_current_checkpoint",
    "load_optimal_checkpoint",
    "save_current_checkpoint",
    "save_optimal_checkpoint",
    "train_one_epoch",
    "validate",
]
