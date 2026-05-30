"""Confidence heads — per-trial and per-task selective-classification sigmoids.

The MATLAB pipeline grafts confidence onto the classifier when
``ConfidenceType`` is non-empty (``cgg_constructClassifierArchitecture.m``
lines 59-63 for ``'Trial'``, ``cgg_addTaskConfidenceToClassifier.m`` for
``'Task'``). The grafted heads emit per-trial / per-task sigmoid scalars
that the loss kernel (``cgg_lossConfidence``, ported in C #5 as
:func:`~neural_data_decoding.training.losses.confidence.apply_confidence_routing`)
consumes for confidence routing.

Pythonic design — composition over graph-walking
-------------------------------------------------

MATLAB's ``cgg_addTaskConfidenceToClassifier`` walks the classifier's
layer graph BACKWARDS to find the FC layer at the head of each branch,
then grafts a sigmoid that taps the layer *just before* that FC. We
don't replicate this graph surgery; instead, the composite asks the
classifier for its penultimate features (via
``DeepLSTMClassifier.forward_with_features``) and the Task confidence
head consumes them in parallel to the classifier's own FC head.
Functionally equivalent — both branches diverge from the same upstream
features — without the regex/graph-walking machinery.

* :class:`TrialConfidenceHead` consumes the latent ``Z`` directly
  (shared with the classifier's input), produces a single per-trial
  scalar per timestep.
* :class:`TaskConfidenceHead` consumes the classifier's penultimate
  features per output dimension, produces one per-dim sigmoid scalar
  per timestep.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class TrialConfidenceHead(nn.Module):
    """Per-trial confidence sigmoid head.

    Maps the latent ``Z`` (or any sequence input of shape
    ``(batch, time, in_features)``) to a single confidence scalar per
    trial per timestep via ``Linear → Sigmoid``. The output shape is
    ``(batch, time, 1)``; the loss kernel takes the last timestep
    (Critical Note #36 ``cgg_getLastSequenceValue``).

    Parameters
    ----------
    in_features
        Size of the input's last axis (the latent dimension).

    Notes
    -----
    The MATLAB equivalent (``cgg_generateLayersForClassifier`` with
    ``ConfidenceType='Trial Confidence'`` and ``NumClasses=1``) builds a
    full LSTM trunk with a 1-output FC at the end. Our simpler
    FC+sigmoid is functionally equivalent for the additive scalar output;
    a deeper trunk can be substituted later if parity to MATLAB
    confidence-trunk weights is required.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be > 0; got {in_features}.")
        self.in_features = in_features
        self.fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute per-trial confidence ``(B, T, 1)``."""
        if z.shape[-1] != self.in_features:
            raise ValueError(
                f"Input last dim ({z.shape[-1]}) does not match "
                f"in_features ({self.in_features})."
            )
        return self.sigmoid(self.fc(z))


class TaskConfidenceHead(nn.Module):
    """Per-output-dimension confidence sigmoid head.

    Consumes the classifier's penultimate features per dimension (each a
    sequence tensor ``(batch, time, hidden_dim)``), produces one
    confidence scalar per dimension per trial per timestep via parallel
    ``Linear(hidden_dim → 1) + Sigmoid`` branches.

    The forward returns a stacked tensor ``(batch, time, num_dims)`` for
    direct consumption by :func:`apply_confidence_routing` (whose
    ``task_confidence`` parameter expects that exact shape).

    Parameters
    ----------
    in_features_per_dim
        Sequence whose ``d``-th entry is the hidden size of the
        classifier's penultimate output for dimension ``d``. Usually all
        the same value (the classifier's final LSTM hidden size).

    Raises
    ------
    ValueError
        On an empty sequence or any non-positive size.
    """

    def __init__(self, in_features_per_dim: Sequence[int]) -> None:
        super().__init__()
        if not in_features_per_dim:
            raise ValueError("in_features_per_dim must be non-empty.")
        if any(n <= 0 for n in in_features_per_dim):
            raise ValueError(
                f"All entries of in_features_per_dim must be > 0; "
                f"got {list(in_features_per_dim)}"
            )
        self.in_features_per_dim = tuple(int(n) for n in in_features_per_dim)
        self.fcs = nn.ModuleList(
            [nn.Linear(int(n), 1) for n in in_features_per_dim]
        )
        self.sigmoid = nn.Sigmoid()
        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
            nn.init.zeros_(fc.bias)

    def forward(
        self, features_per_dim: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-dim confidence stacked along the last axis.

        Parameters
        ----------
        features_per_dim
            One ``(batch, time, hidden_dim_d)`` tensor per output
            dimension, in the same order as ``in_features_per_dim``.

        Returns
        -------
        torch.Tensor
            ``(batch, time, num_dims)`` — sigmoid outputs stacked along
            the last axis. Direct input to
            :func:`apply_confidence_routing` ``task_confidence`` argument.
        """
        if len(features_per_dim) != len(self.fcs):
            raise ValueError(
                f"Expected {len(self.fcs)} per-dim feature tensors; "
                f"got {len(features_per_dim)}."
            )
        per_dim_logits = [
            self.sigmoid(fc(f))  # each (B, T, 1)
            for fc, f in zip(self.fcs, features_per_dim)
        ]
        # Concat along the last axis: (B, T, 1) × num_dims → (B, T, num_dims).
        return torch.cat(per_dim_logits, dim=-1)


__all__ = [
    "TaskConfidenceHead",
    "TrialConfidenceHead",
]
