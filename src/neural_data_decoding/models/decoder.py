"""Decoder builders — no-op stub for Milestone B; full impl in Milestone C.

The MATLAB pipeline's ``cgg_selectDecoder.m`` constructs a symmetric
counterpart to the encoder for the VAE / reconstruction-loss path. For
Milestone B's "classifier-only" configuration (``LossType_Decoder='None'``),
no decoder is exercised: the encoder's output feeds straight into the
bottleneck → classifier. We still need a callable identity here so the
Milestone B CLI doesn't have to special-case the "no decoder" branch —
:class:`NoopDecoder` returns its input unchanged.

Milestone C will add the full Simple-branch / Convolutional decoder
builders alongside the VAE sampling layer. The :func:`build_decoder`
dispatcher is in place now so that future code paths slot in cleanly:
configs with ``loss_type_decoder == "None"`` use the stub, anything else
will raise ``NotImplementedError`` until Milestone C lands.

Examples
--------
>>> import torch
>>> from neural_data_decoding.models.decoder import NoopDecoder, build_decoder
>>> dec = build_decoder({"loss_type_decoder": "None"})
>>> isinstance(dec, NoopDecoder)
True
>>> x = torch.zeros(2, 5, 4)
>>> bool(torch.equal(dec(x), x))
True
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


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


def build_decoder(cfg: Mapping[str, Any]) -> nn.Module:
    """Construct the decoder appropriate to ``cfg``.

    Milestone B branches
    --------------------
    * ``loss_type_decoder == "None"`` → :class:`NoopDecoder`.
    * Any other value → :class:`NotImplementedError` (Milestone C will
      add the Simple / Convolutional decoder branches).

    Parameters
    ----------
    cfg
        Resolved configuration. Reads:

        * ``loss_type_decoder`` (str) — ``"None" | "MSE" | "MAE"``.
        * ``bottleneck_out_features`` (int, optional) — when present,
          recorded on the returned module so downstream code can read
          the input dimensionality without re-deriving it.

    Returns
    -------
    torch.nn.Module
        The decoder placeholder (Milestone B) or the actual decoder
        builder output (Milestone C+).

    Raises
    ------
    NotImplementedError
        For any decoder type other than ``"None"`` until Milestone C.
    """
    loss_type = str(cfg.get("loss_type_decoder", "None"))
    if loss_type == "None":
        return NoopDecoder(
            out_features=cfg.get("bottleneck_out_features")
        )
    raise NotImplementedError(
        f"loss_type_decoder={loss_type!r} requires the full decoder branch, "
        "which is scheduled for Milestone C. Set loss_type_decoder='None' "
        "for the classifier-only Milestone B configurations."
    )


__all__ = ["NoopDecoder", "build_decoder"]
