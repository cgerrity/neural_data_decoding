"""NaN→0 input transform — port of ``cgg_setNaNToValue(x, 0)``.

Channels removed during preprocessing are stored as ``NaN`` in the on-disk
``.mat`` files. The MATLAB encoder's ``sequenceInputLayer`` applies a
normalization function ``cgg_setNaNToValue(x, 0)`` that replaces those
``NaN`` entries with ``0`` *before* the encoder forward pass
(``cgg_constructNetworkArchitecture.m:127-129``). Because the data is
Z-scored upstream, ``0`` is the population mean — a neutral substitute that
doesn't bias the encoder.

Critical Note #38 — two-layered NaN handling
--------------------------------------------
This is layer **(a)** of the two-part NaN strategy:

* **(a) Input path (this module):** replace ``NaN`` with ``0`` so the
  encoder never sees a ``NaN`` (which would propagate through every op).
* **(b) Reconstruction loss:** mask the ``NaN`` positions of the *original*
  target so the decoder isn't penalized at removed-channel positions
  (see :func:`neural_data_decoding.training.losses.elbo.masked_mse_reconstruction_loss`).

Both layers are needed and operate on different tensors: the encoder gets
the NaN-zeroed input; the reconstruction loss gets the NaN-preserving
target. Dropping either silently changes training dynamics.

Examples
--------
>>> import torch
>>> layer = NaNToZero()
>>> x = torch.tensor([[1.0, float("nan"), 3.0]])
>>> layer(x)
tensor([[1., 0., 3.]])
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NaNToZero(nn.Module):
    """Replace ``NaN`` entries with a constant (default ``0``).

    Placed at the head of the encoder so removed-channel ``NaN`` markers
    become a neutral value before any learnable op. Stateless and
    parameter-free.

    Parameters
    ----------
    value
        The replacement value. Defaults to ``0.0`` — correct for Z-scored
        data, where 0 is the population mean.

    Attributes
    ----------
    value : float
        The configured replacement value.
    """

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` with every ``NaN`` replaced by ``self.value``.

        Parameters
        ----------
        x
            Any tensor; ``NaN`` entries are replaced, all others pass
            through unchanged.

        Returns
        -------
        torch.Tensor
            A tensor of the same shape with no ``NaN`` entries. Unlike
            :func:`torch.nan_to_num`, ``±inf`` are left untouched —
            matching ``cgg_setNaNToValue``, which replaces only ``NaN``.
        """
        return torch.where(
            torch.isnan(x),
            torch.full_like(x, self.value),
            x,
        )

    def extra_repr(self) -> str:
        """Show the replacement value in the module's repr."""
        return f"value={self.value}"


__all__ = ["NaNToZero"]
