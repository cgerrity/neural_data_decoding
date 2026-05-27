"""Multi-head classifier heads ported from ``cgg_selectClassifier.m``.

The MATLAB pipeline supports 9 ``ClassifierName`` variants (Logistic, Deep
LSTM ± Dropout, Deep Feedforward ± Dropout, GRU/LSTM stacks, etc.) plus
optional confidence and MIL routing branches. Milestone A wires up only
the Logistic variant — the simplest case, used by ``ModelName='Logistic
Regression'`` — and leaves the rest for Milestones B/C/CC.

Common module: :class:`MultiHeadClassifier`. Each classification
dimension gets its own ``nn.Linear`` head fed from a shared input. The
forward pass returns a *list* of per-dimension logit tensors (the user-
side loss applies softmax + cross-entropy per head).

Critical Note #14 in the migration plan: every ``ClassifierName`` should
be registered by end of Milestone C even if exercised by only a few
targets. Milestone A registers ``'Logistic'``; the others will register
in later milestones.

Examples
--------
>>> import torch
>>> from neural_data_decoding.models.classifier import MultiHeadClassifier
>>> head = MultiHeadClassifier(in_features=8, num_classes_per_dim=[3, 4])
>>> x = torch.zeros(2, 5, 8)  # (batch, time, features)
>>> outputs = head(x)
>>> [o.shape for o in outputs]
[torch.Size([2, 5, 3]), torch.Size([2, 5, 4])]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from .registry import register_classifier


class MultiHeadClassifier(nn.Module):
    """Per-dimension linear classification heads with a shared input.

    The forward pass returns one tensor per output dimension; each tensor
    holds the **logits** (pre-softmax), not probabilities. Applying softmax
    or cross-entropy is the caller's responsibility — this keeps numerical
    stability with PyTorch's fused ``F.cross_entropy`` and matches the
    "registry returns building blocks; loss is composed elsewhere" pattern
    used throughout the pipeline.

    The MATLAB equivalent appends a ``softmaxLayer`` after each
    ``fullyConnectedLayer`` (or ``cgg_softmaxLayer`` when MIL is active).
    In PyTorch we defer that step to the loss kernel for numerical reasons.

    Parameters
    ----------
    in_features
        Size of the last input dimension. For ``ModelName='Logistic
        Regression'`` this is the per-time-step input feature count
        (the encoder is empty).
    num_classes_per_dim
        Number of output classes per classification dimension. For the
        Synthetic_Easy / Dimension target this is typically a 4-element
        list (one per Quaddle dimension).
    dropout
        Optional dropout rate applied to the shared input. Defaults to 0;
        the Logistic variant in MATLAB also uses ``DropoutPercent=0``.

    Attributes
    ----------
    heads : torch.nn.ModuleList
        One ``nn.Linear`` per output dimension, in input order.
    dropout : torch.nn.Module
        ``nn.Dropout(dropout)`` if ``dropout > 0`` else ``nn.Identity``.
    """

    def __init__(
        self,
        in_features: int,
        num_classes_per_dim: Sequence[int],
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be > 0; got {in_features}.")
        if not num_classes_per_dim:
            raise ValueError("num_classes_per_dim must be a non-empty sequence.")
        if any(n <= 0 for n in num_classes_per_dim):
            raise ValueError(
                f"All entries of num_classes_per_dim must be > 0; "
                f"got {list(num_classes_per_dim)}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.in_features = in_features
        self.num_classes_per_dim = tuple(int(n) for n in num_classes_per_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.heads = nn.ModuleList(
            [nn.Linear(in_features, n) for n in self.num_classes_per_dim]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Apply each per-dimension head to the same input tensor.

        Parameters
        ----------
        x
            Input tensor with last dimension equal to ``in_features``. May
            include any number of leading batch/time dimensions; each head
            applies independently to the trailing feature axis.

        Returns
        -------
        list of torch.Tensor
            One logit tensor per classification dimension. Each has the
            same leading shape as ``x`` with its last dimension replaced
            by the corresponding ``num_classes`` value.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input last dim ({x.shape[-1]}) does not match "
                f"in_features ({self.in_features})."
            )
        x = self.dropout(x)
        return [head(x) for head in self.heads]


# ───────────────────────── Builders ─────────────────────────


def build_logistic_classifier(cfg: Mapping[str, Any]) -> MultiHeadClassifier:
    """Construct the multi-head Logistic classifier from a resolved config.

    Mirrors ``cgg_selectClassifier.m::'Logistic'`` plumbed through
    ``cgg_generateLayersForClassifier`` — that branch produces a simple
    chain of (optional dropout → fully-connected → softmax) for each
    output dimension.

    Required config keys
    --------------------
    ``in_features``
        Per-trial / per-time-step feature count entering the classifier.
        For ``ModelName='Logistic Regression'`` this is the input data
        dimensionality (no encoder).
    ``num_classes_per_dim``
        Sequence of per-output-dimension class counts.

    Optional config keys
    --------------------
    ``classifier_dropout``
        Dropout rate. Defaults to ``0.0`` (matches MATLAB's Logistic).

    Parameters
    ----------
    cfg
        Mapping holding the required keys above.

    Returns
    -------
    MultiHeadClassifier
        The constructed classifier.

    Raises
    ------
    KeyError
        If a required key is missing from ``cfg``.
    """
    try:
        in_features = int(cfg["in_features"])
        num_classes_per_dim = list(cfg["num_classes_per_dim"])
    except KeyError as exc:
        raise KeyError(
            f"build_logistic_classifier: missing required cfg key {exc}"
        ) from exc

    dropout = float(cfg.get("classifier_dropout", 0.0))

    return MultiHeadClassifier(
        in_features=in_features,
        num_classes_per_dim=num_classes_per_dim,
        dropout=dropout,
    )


# Side-effect: register on import so importing this module makes 'Logistic'
# discoverable through neural_data_decoding.models.registry.build_classifier.
register_classifier("Logistic")(build_logistic_classifier)


__all__ = [
    "MultiHeadClassifier",
    "build_logistic_classifier",
]
