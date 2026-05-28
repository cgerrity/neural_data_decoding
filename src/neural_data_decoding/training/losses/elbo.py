"""ELBO loss kernels — port of ``cgg_lossELBO_v2.m`` (MSE reconstruction + KL).

The variational autoencoder's evidence-lower-bound loss has two terms:

* **Reconstruction** — a NaN-masked MSE between the decoder output and the
  original (NaN-preserving) target.
* **KL divergence** — pulls the latent posterior toward a unit Gaussian.

Critical Note #38 — NaN-masked reconstruction (verified empirically)
-------------------------------------------------------------------
Removed channels are ``NaN`` in the target. MATLAB computes
``0.5 * l2loss(Y, T, Mask=~isnan(T))`` so those positions contribute zero.
The **normalization** was the highest-risk ambiguity in the whole port: the
migration plan's note suggested dividing by ``mask.sum()`` (the number of
unmasked elements), but a direct probe of MATLAB's ``l2loss`` shows it
divides by the **batch size** regardless of the mask:

    >>> % Y = [1 2; 3 4; 5 6] (CB), T = 0 except T(2,2)=NaN
    >>> % masked sum-of-squares = 75, batch = 2, mask.sum() = 5
    >>> l2loss(Y, T, Mask=~isnan(T))   % == 37.5  == 75/2  (NOT 75/5)

So the reconstruction loss is ``0.5 * Σ(mask · (Y-T)²) / batch_size``.
Using ``mask.sum()`` would silently scale the reconstruction gradient by a
data-dependent factor — exactly the silent-parity-loss this note warns of.

Two-tensor contract
-------------------
The encoder receives the **NaN-zeroed** input (see
:class:`neural_data_decoding.models.layers.nan_to_zero.NaNToZero`); the
reconstruction loss receives the **NaN-preserving** original target. These
kernels expect the target to still contain ``NaN`` at removed-channel
positions so the mask is derived correctly.

Per-channel reconstruction is telemetry only
--------------------------------------------
``cgg_lossELBO_v2`` also returns a per-channel (per-area) reconstruction
loss, but it is detached (``cgg_extractData``) — logged for monitoring,
never backpropagated (Critical Note #33). :func:`per_channel_reconstruction_loss`
returns detached scalars for that purpose.

Examples
--------
>>> import torch
>>> y = torch.zeros(2, 4, 3)               # (batch, time, channel)
>>> t = torch.ones(2, 4, 3)
>>> t[0, 0, 0] = float("nan")              # one removed-channel position
>>> loss = masked_mse_reconstruction_loss(y, t, batch_dim=0)
>>> float(loss) > 0
True
"""

from __future__ import annotations

import torch


def masked_mse_reconstruction_loss(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    *,
    batch_dim: int = 0,
) -> torch.Tensor:
    """NaN-masked MSE reconstruction loss, normalized by batch size.

    Computes ``0.5 * Σ(mask · (y_pred - y_target)²) / N`` where ``mask =
    ~isnan(y_target)`` and ``N`` is the size of ``batch_dim``. This matches
    ``0.5 * l2loss(Y, T, Mask=~isnan(T))`` in MATLAB (batch-size
    normalization — see module docstring).

    Parameters
    ----------
    y_pred
        Decoder output. Must be finite (no ``NaN``).
    y_target
        The original reconstruction target, **with ``NaN`` preserved** at
        removed-channel positions. The mask is derived from these ``NaN``.
    batch_dim
        Axis to normalize by (the observation / batch axis). Defaults to
        ``0``.

    Returns
    -------
    torch.Tensor
        Scalar (0-D) reconstruction loss, differentiable w.r.t. ``y_pred``.

    Raises
    ------
    ValueError
        If ``y_pred`` and ``y_target`` shapes differ.
    """
    if y_pred.shape != y_target.shape:
        raise ValueError(
            f"y_pred shape {tuple(y_pred.shape)} != y_target shape "
            f"{tuple(y_target.shape)}."
        )

    mask = ~torch.isnan(y_target)
    # NaN * 0 == NaN in IEEE, so zero out masked positions with torch.where
    # rather than multiplying — otherwise the NaN would poison the sum.
    diff = torch.where(mask, y_pred - y_target, torch.zeros_like(y_pred))
    sq_sum = (diff**2).sum()
    batch_size = y_pred.shape[batch_dim]
    return 0.5 * sq_sum / batch_size


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    channel_dim: int = -1,
) -> torch.Tensor:
    """KL divergence of the latent posterior from a unit Gaussian.

    Computes ``mean( -0.5 * Σ_channel (1 + logvar - mu² - exp(logvar)) )``,
    matching ``cgg_lossELBO_v2``: the inner sum is over the latent channel
    axis, then the mean is taken over every remaining axis (batch × time).

    Parameters
    ----------
    mu
        Latent mean (from the sampling layer).
    logvar
        Latent log-variance (from the sampling layer). Same shape as ``mu``.
    channel_dim
        Latent-channel axis to sum over. Defaults to ``-1``.

    Returns
    -------
    torch.Tensor
        Scalar (0-D) KL loss.

    Raises
    ------
    ValueError
        If ``mu`` and ``logvar`` shapes differ.
    """
    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu shape {tuple(mu.shape)} != logvar shape {tuple(logvar.shape)}."
        )
    per_sample = -0.5 * (1 + logvar - mu**2 - torch.exp(logvar)).sum(dim=channel_dim)
    return per_sample.mean()


def per_channel_reconstruction_loss(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    *,
    channel_dim: int,
    batch_dim: int = 0,
) -> list[torch.Tensor]:
    """Per-channel reconstruction losses for telemetry (detached).

    Mirrors the per-area loop in ``cgg_lossELBO_v2``: one masked-MSE value
    per channel slice, each computed the same way as the total
    (batch-size normalized) but sliced to a single channel. The values are
    **detached** — they are logged for monitoring and never backpropagated
    (Critical Note #33).

    Parameters
    ----------
    y_pred
        Decoder output.
    y_target
        NaN-preserving reconstruction target.
    channel_dim
        Axis to iterate over (one loss per index along it).
    batch_dim
        Observation axis used for normalization (passed through to
        :func:`masked_mse_reconstruction_loss`).

    Returns
    -------
    list of torch.Tensor
        One detached 0-D tensor per channel, in channel order.
    """
    num_channels = y_pred.shape[channel_dim]
    losses: list[torch.Tensor] = []
    for c in range(num_channels):
        # narrow keeps the channel axis at size 1, so ndim is unchanged and
        # batch_dim stays a valid index for the slice.
        y_slice = y_pred.narrow(channel_dim, c, 1)
        t_slice = y_target.narrow(channel_dim, c, 1)
        losses.append(
            masked_mse_reconstruction_loss(
                y_slice, t_slice, batch_dim=batch_dim
            ).detach()
        )
    return losses


__all__ = [
    "kl_divergence_loss",
    "masked_mse_reconstruction_loss",
    "per_channel_reconstruction_loss",
]
