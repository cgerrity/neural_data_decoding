"""Encoder builders ported from ``cgg_selectEncoder.m`` (Simple branch).

The MATLAB encoder dispatcher splits on ``cfg.IsSimple``:

* **Simple branch** (Milestone B target) — stacks of feedforward / GRU /
  LSTM layers built by ``cgg_constructSimpleCoder`` → ``cgg_generateSimpleBlock``.
* **PCA branch** — frozen PCA layer (Milestone CC).
* **Convolutional branch** — Conv/Resnet stacks (Milestone CC).

This module implements the **Simple branch only**. Each block follows the
MATLAB-specific layer order ``[Transform → Dropout → Norm → Activation]``
(Critical Note #27) — *not* the conventional Transform→Norm→Activation→
Dropout. Preserving this order is load-bearing for parity tests.

For the production "Optimal" config:

* ``ModelName='GRU'`` → ``Transform='GRU'``, no activation, no normalization.
* GRU layers output sequences (``OutputMode='sequence'``), so the encoder
  emits an ``(N, T, hidden_sizes[-1])`` tensor that the bottleneck consumes.

The :class:`SimpleSequenceEncoder` keeps the per-block sub-modules separate
(rather than fusing into a single ``nn.GRU(num_layers=N)``) so:

1. Per-block dropout / norm / activation can be inserted between layers
   without monkeypatching CuDNN-fused stacks.
2. Per-layer freeze schedules (Milestone C) can act on individual blocks.
3. Single-step parity tests can verify activations one block at a time.

Examples
--------
>>> import torch
>>> from neural_data_decoding.models.encoder import SimpleSequenceEncoder
>>> enc = SimpleSequenceEncoder(
...     in_features=8, hidden_sizes=[16, 8], transform="GRU", dropout=0.3
... )
>>> y = enc(torch.zeros(4, 10, 8))  # (batch, time, features)
>>> y.shape
torch.Size([4, 10, 8])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from .registry import register_encoder


_SUPPORTED_TRANSFORMS = ("Feedforward", "GRU", "LSTM")
_SUPPORTED_ACTIVATIONS = ("", "ReLU", "Leaky ReLU", "GeLU", "SoftSign")


class SimpleSequenceEncoder(nn.Module):
    """Stacked Simple-branch encoder.

    Each level produces one block: the chosen transform layer, optional
    dropout, optional layer-norm, optional activation — in **that** order
    (Critical Note #27).

    Parameters
    ----------
    in_features
        Last-axis size of the input tensor (the data's feature/channel count).
    hidden_sizes
        Per-block output sizes. Empty means "no encoder" — the module
        becomes an identity transform.
    transform
        Per-block transform type. One of ``'Feedforward'``, ``'GRU'``,
        ``'LSTM'``. Matches MATLAB's ``cfg.Transform``.
    dropout
        Dropout rate applied **after** each transform layer (Note #27 order).
        ``0.0`` disables dropout.
    want_normalization
        If ``True``, inserts a :class:`torch.nn.LayerNorm` after the dropout
        in every block. The MATLAB default is ``False``; production "Optimal"
        keeps it off (the encoder relies on input normalization in
        ``cgg_selectNormalization``).
    activation
        Per-block activation, applied **last** in the block. ``''`` means
        no activation (the production GRU choice). MATLAB has a known
        naming oddity: ``'SoftSign'`` actually instantiates a softplus
        layer (Critical Note #37) — here we make ``'SoftSign'`` an
        explicit ``nn.Softplus`` so behavior matches MATLAB despite the
        misleading name. Use ``'softplus'`` in new configs.

    Attributes
    ----------
    blocks : torch.nn.ModuleList
        One :class:`_EncoderBlock` per ``hidden_sizes`` entry.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int],
        *,
        transform: str = "GRU",
        dropout: float = 0.0,
        want_normalization: bool = False,
        activation: str = "",
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be > 0; got {in_features}.")
        if transform not in _SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"Unsupported transform {transform!r}. Supported: "
                f"{_SUPPORTED_TRANSFORMS}."
            )
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}. Supported: "
                f"{_SUPPORTED_ACTIVATIONS}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.in_features = in_features
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.transform = transform
        self.dropout = dropout
        self.want_normalization = want_normalization
        self.activation = activation

        self.blocks = nn.ModuleList()
        current_features = in_features
        for h in self.hidden_sizes:
            self.blocks.append(
                _EncoderBlock(
                    in_features=current_features,
                    hidden_size=h,
                    transform=transform,
                    dropout=dropout,
                    want_normalization=want_normalization,
                    activation=activation,
                )
            )
            current_features = h
        self.out_features = current_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stacked blocks left-to-right.

        Parameters
        ----------
        x
            Input tensor shape ``(batch, time, features)``.

        Returns
        -------
        torch.Tensor
            Output shape ``(batch, time, hidden_sizes[-1])`` (or ``x``
            unchanged if ``hidden_sizes`` is empty).
        """
        for block in self.blocks:
            x = block(x)
        return x


class _EncoderBlock(nn.Module):
    """One block of the Simple encoder — Transform → Dropout → Norm → Activation."""

    __constants__ = ["transform"]

    def __init__(
        self,
        *,
        in_features: int,
        hidden_size: int,
        transform: str,
        dropout: float,
        want_normalization: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.transform = transform

        if transform == "Feedforward":
            self.transform_layer: nn.Module = nn.Linear(in_features, hidden_size)
            nn.init.kaiming_normal_(
                self.transform_layer.weight, nonlinearity="relu"
            )  # Critical Note #31 — He init explicit on FC layers.
            nn.init.zeros_(self.transform_layer.bias)
        elif transform == "GRU":
            self.transform_layer = nn.GRU(
                input_size=in_features,
                hidden_size=hidden_size,
                batch_first=True,
            )
        elif transform == "LSTM":
            self.transform_layer = nn.LSTM(
                input_size=in_features,
                hidden_size=hidden_size,
                batch_first=True,
            )
        else:  # pragma: no cover - guarded by __init__
            raise ValueError(f"Unsupported transform {transform!r}.")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = (
            nn.LayerNorm(hidden_size) if want_normalization else nn.Identity()
        )
        self.activation: nn.Module = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Transform → Dropout → Norm → Activation in MATLAB's order."""
        if self.transform == "Feedforward":
            x = self.transform_layer(x)
        else:
            # nn.GRU / nn.LSTM return (output, hidden). We want the per-timestep
            # output sequence — MATLAB's gruLayer(OutputMode='sequence').
            x = self.transform_layer(x)[0]

        x = self.dropout(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


def _make_activation(name: str) -> nn.Module:
    """Map MATLAB activation strings to ``nn.Module`` instances.

    Note #37 — MATLAB's ``'SoftSign'`` actually instantiates a softplus
    layer (a long-standing naming bug). We propagate the same behavior
    so loaded MATLAB checkpoints work, but new configs should use
    ``''`` or a correctly named alternative.
    """
    if name == "":
        return nn.Identity()
    if name == "ReLU":
        return nn.ReLU()
    if name == "Leaky ReLU":
        return nn.LeakyReLU()
    if name == "GeLU":
        return nn.GELU()
    if name == "SoftSign":
        return nn.Softplus()
    raise ValueError(f"Unsupported activation {name!r}.")


# ───────────────────────── Builders + registry entries ─────────────────────────


def build_simple_encoder(cfg: Mapping[str, Any]) -> SimpleSequenceEncoder:
    """Construct a Simple-branch encoder from a resolved config.

    Recognized config keys
    ----------------------
    ``in_features`` (required)
        Last-axis size of the data.
    ``hidden_sizes`` (required)
        Sequence of per-block output sizes.
    ``transform``
        ``'Feedforward'`` | ``'GRU'`` | ``'LSTM'``. Defaults to ``'GRU'``.
    ``dropout``
        Defaults to ``0.5`` (matching MATLAB's ``cgg_generateSimpleBlock``).
    ``want_normalization``
        Defaults to ``False``.
    ``activation``
        Defaults to ``''`` (no activation — matches the GRU production path).

    Returns
    -------
    SimpleSequenceEncoder

    Raises
    ------
    KeyError
        If ``in_features`` or ``hidden_sizes`` is missing.
    """
    try:
        in_features = int(cfg["in_features"])
        hidden_sizes = list(cfg["hidden_sizes"])
    except KeyError as exc:
        raise KeyError(
            f"build_simple_encoder: missing required cfg key {exc}"
        ) from exc
    return SimpleSequenceEncoder(
        in_features=in_features,
        hidden_sizes=hidden_sizes,
        transform=str(cfg.get("transform", "GRU")),
        dropout=float(cfg.get("dropout", 0.5)),
        want_normalization=bool(cfg.get("want_normalization", False)),
        activation=str(cfg.get("activation", "")),
    )


def _make_typed_builder(transform: str):
    """Helper: return a builder that hard-codes the ``transform`` field."""

    def _builder(cfg: Mapping[str, Any]) -> SimpleSequenceEncoder:
        cfg_with_transform = dict(cfg)
        cfg_with_transform["transform"] = transform
        return build_simple_encoder(cfg_with_transform)

    _builder.__name__ = f"build_{transform.lower()}_encoder"
    _builder.__doc__ = (
        f"Construct a Simple-branch encoder with ``transform={transform!r}``."
    )
    return _builder


# Register the three Simple-branch ``ModelName`` strings active in Milestone B.
# Milestone CC will register the conv/PCA variants alongside these.
register_encoder("GRU")(_make_typed_builder("GRU"))
register_encoder("LSTM")(_make_typed_builder("LSTM"))
register_encoder("Feedforward")(_make_typed_builder("Feedforward"))


__all__ = ["SimpleSequenceEncoder", "build_simple_encoder"]
