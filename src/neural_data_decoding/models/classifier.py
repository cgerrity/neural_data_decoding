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


# ───────────────────────── Deep LSTM (per-dim trunks) ─────────────────────────


class DeepLSTMClassifier(nn.Module):
    """Multi-head LSTM classifier — separate LSTM stack per output dimension.

    Mirrors ``cgg_selectClassifier::'Deep LSTM - Dropout 0.5'`` →
    ``cgg_generateLayersForClassifier(NetworkType='LSTM', DropoutPercent=0.5)``.
    For each output dimension ``d`` MATLAB builds an independent layer
    graph::

        [LSTM(H_1, seq) → Dropout]    \\
        [LSTM(H_2, seq) → Dropout]     \\  one stack per dim
        ...                             /
        [LSTM(H_last, seq) → Dropout]  /
        FC(NumClasses[d])
        softmax (or cgg_softmaxLayer for MIL)

    The Python equivalent uses ``nn.ModuleList`` of per-dim stacks. All
    LSTMs keep the time dimension (``batch_first=True``, default
    OutputMode='sequence'); the loss kernel reduces over time via
    :func:`~neural_data_decoding.training.losses.classification.multi_head_cross_entropy`,
    which handles ``(B, T, K)`` logits exactly the way MATLAB's
    ``crossentropy(Y, T)`` does when ``T`` is broadcast across time.

    The classifier returns **logits** (pre-softmax) per Critical Note: the
    softmax → cross-entropy fusion happens in :func:`F.cross_entropy`.

    Parameters
    ----------
    in_features
        Last-axis size of the input tensor (the bottleneck's output dim).
    num_classes_per_dim
        Output classes per classification dimension. The classifier
        produces one logit tensor per dim, each of shape
        ``(batch, time, num_classes_d)``.
    hidden_sizes
        LSTM hidden sizes for the inner stack. Length determines depth.
        Must be non-empty.
    dropout
        Dropout rate inserted after each LSTM in the stack. ``0.5`` for
        ``'Deep LSTM - Dropout 0.5'``; ``0.25`` for the
        ``'Deep LSTM - Dropout 0.25'`` variant.

    Attributes
    ----------
    stacks : torch.nn.ModuleList
        One :class:`_LSTMDimStack` per output dimension.
    """

    def __init__(
        self,
        in_features: int,
        num_classes_per_dim: Sequence[int],
        *,
        hidden_sizes: Sequence[int],
        dropout: float = 0.5,
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
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty for DeepLSTMClassifier.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.in_features = in_features
        self.num_classes_per_dim = tuple(int(n) for n in num_classes_per_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.dropout = dropout

        self.stacks = nn.ModuleList(
            [
                _LSTMDimStack(
                    in_features=in_features,
                    hidden_sizes=self.hidden_sizes,
                    dropout=dropout,
                    num_classes=n,
                )
                for n in self.num_classes_per_dim
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run each per-dim stack on the same input.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, time, in_features)``.

        Returns
        -------
        list of torch.Tensor
            One logit tensor per output dimension, each of shape
            ``(batch, time, num_classes_d)``.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input last dim ({x.shape[-1]}) does not match "
                f"in_features ({self.in_features})."
            )
        return [stack(x) for stack in self.stacks]

    def forward_with_features(
        self, x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Same as :meth:`forward` but also returns per-dim penultimate features.

        Used by :class:`~neural_data_decoding.models.confidence_heads.TaskConfidenceHead`
        which taps the classifier's last LSTM output (just before the FC
        head) to emit a per-dim sigmoid scalar. Returning both keeps the
        existing ``forward`` contract intact for callers that only need
        logits.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, time, in_features)``.

        Returns
        -------
        features_per_dim : list of torch.Tensor
            Penultimate-layer output per dim, each
            ``(batch, time, hidden_size_last)``.
        logits_per_dim : list of torch.Tensor
            Same as :meth:`forward`, each ``(batch, time, num_classes_d)``.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input last dim ({x.shape[-1]}) does not match "
                f"in_features ({self.in_features})."
            )
        features_per_dim: list[torch.Tensor] = []
        logits_per_dim: list[torch.Tensor] = []
        for stack in self.stacks:
            features, logits = stack.forward_with_features(x)
            features_per_dim.append(features)
            logits_per_dim.append(logits)
        return features_per_dim, logits_per_dim


class _LSTMDimStack(nn.Module):
    """Single per-dimension LSTM stack + final FC head."""

    def __init__(
        self,
        *,
        in_features: int,
        hidden_sizes: Sequence[int],
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.lstms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        current = in_features
        for h in hidden_sizes:
            self.lstms.append(
                nn.LSTM(input_size=current, hidden_size=h, batch_first=True)
            )
            self.dropouts.append(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            current = h
        self.penultimate_size = current
        self.head = nn.Linear(current, num_classes)
        # MATLAB applies He init explicitly to FC layers; do the same here
        # for parity with weight-load tests (Critical Note #31).
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)

    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LSTM stack + dropouts; return the penultimate features."""
        for lstm, drop in zip(self.lstms, self.dropouts):
            x = lstm(x)[0]  # (B, T, H) — sequence output
            x = drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each LSTM (sequence mode) + dropout, then the per-timestep head."""
        return self.head(self._compute_features(x))

    def forward_with_features(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(penultimate_features, logits)`` — used by the Task confidence head."""
        features = self._compute_features(x)
        return features, self.head(features)


# ───────────────────────── Builders ─────────────────────────


def build_deep_lstm_classifier(
    cfg: Mapping[str, Any], *, dropout: float
) -> DeepLSTMClassifier:
    """Construct a Deep-LSTM classifier with a fixed dropout rate.

    Used by the two registered MATLAB variants
    (``'Deep LSTM - Dropout 0.5'``, ``'Deep LSTM - Dropout 0.25'``) which
    are identical except for the dropout magnitude.

    Required cfg keys
    -----------------
    ``in_features``
    ``num_classes_per_dim``
    ``classifier_hidden_size`` (Sequence[int])

    Parameters
    ----------
    cfg
        Resolved configuration mapping.
    dropout
        Hard-coded dropout rate (set by the registered builder).

    Returns
    -------
    DeepLSTMClassifier
    """
    try:
        in_features = int(cfg["in_features"])
        num_classes_per_dim = list(cfg["num_classes_per_dim"])
        hidden_sizes = list(cfg["classifier_hidden_size"])
    except KeyError as exc:
        raise KeyError(
            f"build_deep_lstm_classifier: missing required cfg key {exc}"
        ) from exc
    return DeepLSTMClassifier(
        in_features=in_features,
        num_classes_per_dim=num_classes_per_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )


# Side-effect: register on import so importing this module makes the
# classifier strings discoverable through registry.build_classifier.
register_classifier("Logistic")(build_logistic_classifier)
register_classifier("Deep LSTM - Dropout 0.5")(
    lambda cfg: build_deep_lstm_classifier(cfg, dropout=0.5)
)
register_classifier("Deep LSTM - Dropout 0.25")(
    lambda cfg: build_deep_lstm_classifier(cfg, dropout=0.25)
)


__all__ = [
    "DeepLSTMClassifier",
    "MultiHeadClassifier",
    "build_deep_lstm_classifier",
    "build_logistic_classifier",
]
