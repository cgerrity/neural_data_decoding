"""Decoder builders — port of ``cgg_selectDecoder.m`` (Simple branch).

The decoder is the reconstruction half of the VAE: it maps the latent code
``Z`` back to the input feature space so the ELBO reconstruction loss can
compare against the original (NaN-preserving) target.

Two variants live here:

* :class:`NoopDecoder` — identity placeholder for classifier-only configs
  (``loss_type_decoder == "None"``, Milestones A/B). No reconstruction term
  fires, so the decoder slot is a pass-through.
* :class:`SimpleSequenceDecoder` — the real Simple-branch decoder
  (Milestone C). A reversed GRU/LSTM/Feedforward stack followed by an output
  ``Linear`` that reconstructs the input feature dimensionality.

MATLAB structure (``cgg_selectDecoder`` Simple branch)
------------------------------------------------------
MATLAB builds ``cgg_constructSimpleCoder([HiddenSizeAutoEncoder,
HiddenSizeBottleNeck], Coder='Decoder')``. With ``Coder='Decoder'`` the
block list is **reversed**, so for encoder hidden ``[H1, H2]`` and latent
``L`` the decoder's transform sequence is ``GRU(L) → GRU(H2) → GRU(H1)``
(i.e. ``reversed([H1, H2, L])``). A final output projection
(``OutputFullyConnected`` → ``fullyConnectedLayer(prod(InputSize))``)
reconstructs the input. :class:`SimpleSequenceDecoder` reproduces that:
the GRU stack uses the reversed sizes, then a ``Linear`` maps to
``output_features``.

Examples
--------
>>> import torch
>>> from neural_data_decoding.models.decoder import SimpleSequenceDecoder
>>> dec = SimpleSequenceDecoder(
...     latent_size=4, hidden_sizes=[4, 8], output_features=6, transform="GRU"
... )
>>> dec(torch.zeros(2, 5, 4)).shape   # (batch, time, output_features)
torch.Size([2, 5, 6])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from neural_data_decoding.models.encoder import SimpleSequenceEncoder


class NoopDecoder(nn.Module):
    """Identity placeholder for the decoder slot.

    Returns its input unchanged. Used in Milestone B configurations where
    ``loss_type_decoder == "None"`` — the reconstruction loss never fires,
    so the decoder pathway is a no-op.

    Attributes
    ----------
    out_features : int | None
        Mirrors the ``in_features`` reported by the upstream bottleneck,
        if known. Set to ``None`` when not specified.
    """

    def __init__(self, out_features: int | None = None) -> None:
        super().__init__()
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged."""
        return x


class SimpleSequenceDecoder(nn.Module):
    """Reversed sequence-transform stack + output projection.

    Maps a latent sequence ``(batch, time, latent_size)`` back to a
    reconstruction ``(batch, time, output_features)``. The transform stack
    mirrors the encoder (MATLAB reverses the block order); a final
    ``Linear`` reconstructs the input feature dimension.

    Parameters
    ----------
    latent_size
        Size of the latent input (the sampling layer's output channel
        count). For the Stochastic VAE this is the latent dimensionality.
    hidden_sizes
        Per-block output sizes of the decoder transform stack, in
        **forward (already-reversed) order** — i.e. for encoder hidden
        ``[H1, H2]`` and latent ``L`` pass ``reversed([H1, H2, L])`` =
        ``[L, H2, H1]``. Use :func:`build_decoder` to compute this from a
        config without thinking about the reversal.
    output_features
        Reconstruction target dimensionality (the encoder's input feature
        count). The final ``Linear`` maps the last hidden size to this.
    transform
        ``'GRU'`` | ``'LSTM'`` | ``'Feedforward'``. Defaults to ``'GRU'``.
    dropout
        Dropout rate within each transform block. Defaults to ``0.0``.
    want_normalization
        Whether to insert per-block layer norm. Defaults to ``False``.
    activation
        Per-block activation string (see
        :class:`~neural_data_decoding.models.encoder.SimpleSequenceEncoder`).
        Defaults to ``''`` (none).

    Attributes
    ----------
    stack : SimpleSequenceEncoder
        The transform stack (reused — a decoder stack is structurally the
        same sequence-transform stack as an encoder).
    output : torch.nn.Linear
        Final projection to ``output_features`` (He-initialized to match
        MATLAB's FC weight init).
    """

    def __init__(
        self,
        latent_size: int,
        hidden_sizes: Sequence[int],
        output_features: int,
        *,
        transform: str = "GRU",
        dropout: float = 0.0,
        want_normalization: bool = False,
        activation: str = "",
    ) -> None:
        super().__init__()
        if latent_size <= 0:
            raise ValueError(f"latent_size must be > 0; got {latent_size}.")
        if output_features <= 0:
            raise ValueError(f"output_features must be > 0; got {output_features}.")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty for the decoder stack.")

        self.latent_size = latent_size
        self.output_features = output_features

        self.stack = SimpleSequenceEncoder(
            in_features=latent_size,
            hidden_sizes=hidden_sizes,
            transform=transform,
            dropout=dropout,
            want_normalization=want_normalization,
            activation=activation,
        )
        self.output = nn.Linear(self.stack.out_features, output_features)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity="relu")
        nn.init.zeros_(self.output.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent sequence into a reconstruction.

        Parameters
        ----------
        z
            Latent tensor of shape ``(batch, time, latent_size)``.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape ``(batch, time, output_features)``.
        """
        return self.output(self.stack(z))


def build_decoder(cfg: Mapping[str, Any]) -> nn.Module:
    """Construct the decoder appropriate to ``cfg``.

    Branches
    --------
    * ``loss_type_decoder == "None"`` → :class:`NoopDecoder`.
    * Otherwise → :class:`SimpleSequenceDecoder`, with the transform-stack
      sizes computed as ``reversed(encoder_hidden_sizes + [latent_size])``
      to mirror MATLAB's reversed decoder block order.

    Recognized config keys
    ----------------------
    ``loss_type_decoder``
        ``"None"`` | ``"MSE"`` | ``"MAE"``. ``"MAE"`` is Milestone CC; for
        now any non-``"None"`` value builds the MSE-style decoder
        (reconstruction loss type is applied at the loss layer, not here).
    ``latent_size`` (required when not ``"None"``)
        Latent dimensionality entering the decoder.
    ``encoder_hidden_sizes`` (required when not ``"None"``)
        The encoder's hidden sizes (NOT reversed — this function reverses).
    ``output_features`` (required when not ``"None"``)
        Reconstruction target dimensionality.
    ``transform`` / ``dropout`` / ``want_normalization`` / ``activation``
        Optional; forwarded to :class:`SimpleSequenceDecoder`.

    Returns
    -------
    torch.nn.Module
        :class:`NoopDecoder` or :class:`SimpleSequenceDecoder`.

    Raises
    ------
    KeyError
        If a required key is missing for the non-``None`` branch.
    """
    loss_type = str(cfg.get("loss_type_decoder", "None"))
    if loss_type == "None":
        return NoopDecoder(out_features=cfg.get("bottleneck_out_features"))

    try:
        latent_size = int(cfg["latent_size"])
        encoder_hidden = list(cfg["encoder_hidden_sizes"])
        output_features = int(cfg["output_features"])
    except KeyError as exc:
        raise KeyError(f"build_decoder: missing required cfg key {exc}") from exc

    # MATLAB decoder sizes = reversed([encoder_hidden..., latent]).
    decoder_hidden = list(reversed([*encoder_hidden, latent_size]))
    return SimpleSequenceDecoder(
        latent_size=latent_size,
        hidden_sizes=decoder_hidden,
        output_features=output_features,
        transform=str(cfg.get("transform", "GRU")),
        dropout=float(cfg.get("dropout", 0.0)),
        want_normalization=bool(cfg.get("want_normalization", False)),
        activation=str(cfg.get("activation", "")),
    )


__all__ = ["NoopDecoder", "SimpleSequenceDecoder", "build_decoder"]
