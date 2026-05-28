"""Multi-objective loss orchestration (ELBO + classification + confidence + augmentation)."""

from neural_data_decoding.training.losses.classification import (
    inverse_frequency_class_weights,
    multi_head_cross_entropy,
)
from neural_data_decoding.training.losses.elbo import (
    kl_divergence_loss,
    masked_mse_reconstruction_loss,
    per_channel_reconstruction_loss,
)
from neural_data_decoding.training.losses.multi_objective import (
    LossBreakdown,
    aggregate_total_loss,
)

__all__ = [
    "LossBreakdown",
    "aggregate_total_loss",
    "inverse_frequency_class_weights",
    "kl_divergence_loss",
    "masked_mse_reconstruction_loss",
    "multi_head_cross_entropy",
    "per_channel_reconstruction_loss",
]
