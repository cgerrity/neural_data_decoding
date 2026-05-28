"""VAE reparameterization (sampling) layer — port of ``cgg_samplingLayer.m``.

The encoder emits a tensor whose channel axis holds the latent statistics
concatenated: the first half is the mean ``mu``, the second half the
log-variance ``logSigmaSq``. This layer splits them, draws the latent code
``Z``, and returns ``(Z, mu, logSigmaSq)`` so the downstream ELBO loss can
read the statistics for the KL term.

Critical Note #35 — **deterministic at inference**
--------------------------------------------------
MATLAB's ``cgg_samplingLayer`` has two code paths:

* ``forward`` (training): draws ``epsilon = randn(...)`` and returns the
  proper reparameterized sample ``Z = epsilon .* sigma + mu``.
* ``predict`` (validation / test): computes ``epsilon = randn(...)`` then
  **zeroes it** (``epsilon = epsilon * 0``), so ``Z = mu`` — deterministic.

This is intentional: a trained model must produce the *same* classification
for the same input at inference time, while still using stochastic gradient
flow during training. This is **not** the textbook VAE implementation (most
PyTorch tutorials sample in both modes), so the ``self.training`` branch
below is deliberate — do not "simplify" it to always sample.

Channel split (odd-channel handling)
------------------------------------
MATLAB uses ``K = floor(C/2)``, ``mu = X(1:K)``, ``logSigmaSq = X(K+1:2K)``
— if the channel count is odd, the trailing channel is dropped. We
reproduce that with explicit slicing (``x[..., :K]`` / ``x[..., K:2K]``)
rather than :func:`torch.chunk`, which would split an odd axis unevenly.

Examples
--------
>>> import torch
>>> layer = SamplingLayer()
>>> x = torch.randn(2, 5, 8)          # (batch, time, 2*latent=8) → latent=4
>>> layer.eval()
SamplingLayer(channel_dim=-1)
>>> z, mu, logvar = layer(x)
>>> z.shape, mu.shape, logvar.shape
(torch.Size([2, 5, 4]), torch.Size([2, 5, 4]), torch.Size([2, 5, 4]))
>>> bool(torch.equal(z, mu))          # deterministic in eval mode
True
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SamplingLayer(nn.Module):
    """Split latent statistics and draw the VAE latent code.

    The input's channel axis must hold ``[mu | logSigmaSq]`` concatenated.
    The output latent dimensionality is ``floor(C / 2)`` where ``C`` is the
    channel-axis size.

    Parameters
    ----------
    channel_dim
        Axis along which ``mu`` and ``logSigmaSq`` are concatenated.
        Defaults to ``-1`` (the trailing feature axis of a ``(batch, time,
        channel)`` tensor). Set explicitly if your tensor uses a different
        layout.

    Attributes
    ----------
    channel_dim : int
        The configured channel axis.
    """

    def __init__(self, channel_dim: int = -1) -> None:
        super().__init__()
        self.channel_dim = channel_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split statistics and return ``(Z, mu, logSigmaSq)``.

        In ``train`` mode draws a reparameterized sample; in ``eval`` mode
        returns ``Z = mu`` deterministically (Critical Note #35).

        Parameters
        ----------
        x
            Input tensor whose ``channel_dim`` axis holds ``[mu |
            logSigmaSq]``. Its size along that axis should be even; an odd
            size drops the trailing channel (matching MATLAB).

        Returns
        -------
        z : torch.Tensor
            The latent code. Same shape as ``mu``.
        mu : torch.Tensor
            The latent mean (first half of the channel axis).
        logvar : torch.Tensor
            The latent log-variance (second half of the channel axis).
        """
        channel_size = x.shape[self.channel_dim]
        latent = channel_size // 2

        mu = x.narrow(self.channel_dim, 0, latent)
        logvar = x.narrow(self.channel_dim, latent, latent)

        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu  # deterministic at inference — Critical Note #35

        return z, mu, logvar

    def extra_repr(self) -> str:
        """Show the channel axis in the module's repr."""
        return f"channel_dim={self.channel_dim}"


__all__ = ["SamplingLayer"]
