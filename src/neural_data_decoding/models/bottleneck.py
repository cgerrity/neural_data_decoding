"""Bottleneck builders ported from ``cgg_selectBottleNeck.m`` (Simple branch).

The bottleneck sits between the encoder and the classifier (and, when
present, the decoder). For the Simple branch:

* If ``HiddenSizeBottleNeck`` is **empty**, the bottleneck is a
  ``flattenLayer`` — i.e., the encoder's per-timestep output is flattened
  to a single feature vector per trial.
* Otherwise, the bottleneck is a (depth-N) stack of Simple blocks
  ``[Transform, Dropout, Norm, Activation]`` followed by a
  ``fullyConnectedLayer(HiddenSizeBottleNeck)`` with **He init**
  (Critical Note #31).

For Milestone B the bottleneck consumes the GRU encoder's
``(batch, time, hidden)`` output. The Python equivalent of MATLAB's
flatten-then-FC keeps the time dimension by default — the **classifier**
is the consumer of the sequence (Deep LSTM expects sequence input). When
the bottleneck collapses time (the ``BottleNeckDepth=NaN`` case in
``cgg_constructSimpleCoder``), set ``collapse_time=True``.

For the simplest Milestone B target (no explicit bottleneck stack), the
default :class:`PassthroughBottleneck` returns the encoder output
unchanged. That's the right "transparent" choice when ``hidden_sizes``
is empty.

Examples
--------
>>> import torch
>>> from neural_data_decoding.models.bottleneck import (
...     PassthroughBottleneck, LinearBottleneck,
... )
>>> b = LinearBottleneck(in_features=8, hidden_size=4)
>>> y = b(torch.zeros(2, 5, 8))   # (batch, time, features)
>>> y.shape
torch.Size([2, 5, 4])
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class PassthroughBottleneck(nn.Module):
    """No-op bottleneck — returns the input unchanged.

    Used when the config has no explicit bottleneck stack (the encoder's
    final hidden size feeds directly into the classifier). Mirrors the
    MATLAB code path where ``HiddenSizeBottleNeck=[]`` returns just a
    ``flattenLayer`` — for our sequence-mode encoder the equivalent
    "passthrough" is just the identity (the classifier consumes the
    sequence directly).

    Attributes
    ----------
    in_features : int
        Last-axis size, passed through verbatim. Recorded so downstream
        components (the classifier) can read it without a separate cfg.
    out_features : int
        Equals ``in_features``.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be > 0; got {in_features}.")
        self.in_features = in_features
        self.out_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged."""
        return x


class LinearBottleneck(nn.Module):
    """Apply a single ``Linear(in_features → hidden_size)`` with He init.

    Mirrors the final ``fullyConnectedLayer(HiddenSizeBottleNeck,
    "WeightsInitializer","he")`` that ``cgg_selectBottleNeck`` always
    appends after the simple-block stack. For Milestone B we use this
    standalone (no preceding simple-block stack) — Milestone C adds the
    full per-block stack when ``BottleNeckDepth > 1``.

    The bottleneck preserves the time axis by default (applies the linear
    map per-timestep), since the Deep LSTM classifier downstream expects
    sequence input.

    Parameters
    ----------
    in_features
        Last-axis size of the input tensor.
    hidden_size
        Output last-axis size.

    Attributes
    ----------
    linear : torch.nn.Linear
        The transform layer; ``weight`` is initialized via Kaiming-normal
        with ``nonlinearity='relu'`` to match MATLAB's ``'he'`` init.
    """

    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be > 0; got {in_features}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0; got {hidden_size}.")
        self.in_features = in_features
        self.out_features = hidden_size
        self.linear = nn.Linear(in_features, hidden_size)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer to the trailing feature axis."""
        return self.linear(x)


def build_bottleneck(cfg: Mapping[str, Any]) -> nn.Module:
    """Construct the right bottleneck variant from a resolved config.

    Recognized config keys
    ----------------------
    ``in_features`` (required)
        Last-axis size of the encoder's output (or of raw data when the
        encoder is empty).
    ``bottleneck_hidden_size``
        Output size for the FC-style bottleneck. ``None`` or absent →
        :class:`PassthroughBottleneck`. Otherwise →
        :class:`LinearBottleneck`.

    Returns
    -------
    torch.nn.Module
        Either a :class:`PassthroughBottleneck` or a :class:`LinearBottleneck`.

    Raises
    ------
    KeyError
        If ``in_features`` is missing.
    """
    try:
        in_features = int(cfg["in_features"])
    except KeyError as exc:
        raise KeyError(
            f"build_bottleneck: missing required cfg key {exc}"
        ) from exc

    hidden_size = cfg.get("bottleneck_hidden_size", None)
    if hidden_size is None:
        return PassthroughBottleneck(in_features=in_features)
    return LinearBottleneck(in_features=in_features, hidden_size=int(hidden_size))


__all__ = ["LinearBottleneck", "PassthroughBottleneck", "build_bottleneck"]
