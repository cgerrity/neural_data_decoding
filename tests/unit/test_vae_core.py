"""Unit tests for the VAE-core building blocks (sampling, NaN-zero, ELBO)."""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.models.layers.nan_to_zero import NaNToZero
from neural_data_decoding.models.layers.sampling import SamplingLayer
from neural_data_decoding.training.losses.elbo import (
    compute_reconstruction_loss,
    kl_divergence_loss,
    masked_mae_reconstruction_loss,
    masked_mse_reconstruction_loss,
    per_channel_reconstruction_loss,
)


# ───────────────────────── SamplingLayer ─────────────────────────


def test_sampling_splits_channel_in_half() -> None:
    """Output latent dim is floor(C/2)."""
    layer = SamplingLayer()
    layer.eval()
    z, mu, logvar = layer(torch.randn(2, 5, 8))
    assert z.shape == (2, 5, 4)
    assert mu.shape == (2, 5, 4)
    assert logvar.shape == (2, 5, 4)


def test_sampling_odd_channel_drops_trailing() -> None:
    """An odd channel count drops the trailing channel (matches MATLAB floor)."""
    layer = SamplingLayer()
    layer.eval()
    z, mu, logvar = layer(torch.randn(2, 5, 7))  # 7 → latent 3
    assert z.shape[-1] == 3
    assert logvar.shape[-1] == 3


def test_sampling_train_uses_reparameterization() -> None:
    """In train mode Z != mu (a sample is drawn) but mu equals the input half."""
    torch.manual_seed(0)
    layer = SamplingLayer()
    layer.train()
    x = torch.randn(2, 3, 6)
    z, mu, _ = layer(x)
    assert not torch.equal(z, mu)
    # mu is exactly the first half of the channel axis.
    assert torch.equal(mu, x[..., :3])


def test_sampling_custom_channel_dim() -> None:
    """channel_dim can target a non-trailing axis (e.g. CBT layout)."""
    layer = SamplingLayer(channel_dim=0)
    layer.eval()
    z, mu, logvar = layer(torch.randn(8, 2, 4))  # channel axis 0 → latent 4
    assert z.shape == (4, 2, 4)


# ───────────────────────── NaNToZero ─────────────────────────


def test_nan_to_zero_replaces_only_nan() -> None:
    """NaN → 0; finite values (including ±inf) pass through unchanged."""
    layer = NaNToZero()
    x = torch.tensor([1.0, float("nan"), -2.0, float("inf"), float("-inf")])
    out = layer(x)
    assert out[0].item() == 1.0
    assert out[1].item() == 0.0
    assert out[2].item() == -2.0
    assert torch.isinf(out[3]) and out[3] > 0  # +inf preserved
    assert torch.isinf(out[4]) and out[4] < 0  # -inf preserved


def test_nan_to_zero_custom_value() -> None:
    """A non-default fill value is honored."""
    layer = NaNToZero(value=-99.0)
    out = layer(torch.tensor([float("nan"), 1.0]))
    assert out[0].item() == -99.0


def test_nan_to_zero_no_nan_is_identity() -> None:
    """A NaN-free tensor passes through unchanged."""
    layer = NaNToZero()
    x = torch.randn(3, 4)
    assert torch.equal(layer(x), x)


# ───────────────────────── ELBO kernels ─────────────────────────


def test_masked_mse_ignores_nan_positions() -> None:
    """A NaN position in the target contributes zero to the loss."""
    y = torch.tensor([[1.0, 1.0]])  # (batch=1, 2)
    t_no_nan = torch.tensor([[0.0, 0.0]])
    t_one_nan = torch.tensor([[0.0, float("nan")]])
    full = masked_mse_reconstruction_loss(y, t_no_nan, batch_dim=0)
    masked = masked_mse_reconstruction_loss(y, t_one_nan, batch_dim=0)
    # full = 0.5*(1+1)/1 = 1.0 ; masked = 0.5*(1)/1 = 0.5
    assert float(full) == pytest.approx(1.0)
    assert float(masked) == pytest.approx(0.5)


def test_masked_mse_is_finite_with_nan_target() -> None:
    """Masking avoids NaN poisoning (NaN*0 == NaN trap)."""
    y = torch.zeros(2, 3)
    t = torch.full((2, 3), float("nan"))
    t[0, 0] = 1.0  # one finite position
    loss = masked_mse_reconstruction_loss(y, t, batch_dim=0)
    assert torch.isfinite(loss)


def test_masked_mse_normalizes_by_batch_size() -> None:
    """Doubling the batch (same per-trial error) halves the per-element mean."""
    y1 = torch.ones(1, 4)
    t1 = torch.zeros(1, 4)
    y2 = torch.ones(2, 4)
    t2 = torch.zeros(2, 4)
    l1 = masked_mse_reconstruction_loss(y1, t1, batch_dim=0)
    l2 = masked_mse_reconstruction_loss(y2, t2, batch_dim=0)
    # Sum doubles (2x rows) but divisor doubles too → equal.
    assert float(l1) == pytest.approx(float(l2))


def test_masked_mse_shape_mismatch_raises() -> None:
    """Mismatched shapes are a programming error."""
    with pytest.raises(ValueError, match="shape"):
        masked_mse_reconstruction_loss(
            torch.zeros(2, 3), torch.zeros(2, 4), batch_dim=0
        )


# ───────────────────────── ELBO kernels — MAE ─────────────────────────


def test_masked_mae_ignores_nan_positions() -> None:
    """MAE: NaN positions contribute zero (parallel to MSE test)."""
    y = torch.tensor([[1.0, 1.0]])
    t_no_nan = torch.tensor([[0.0, 0.0]])
    t_one_nan = torch.tensor([[0.0, float("nan")]])
    full = masked_mae_reconstruction_loss(y, t_no_nan, batch_dim=0)
    masked = masked_mae_reconstruction_loss(y, t_one_nan, batch_dim=0)
    # full = (1+1)/1 = 2.0 ; masked = 1/1 = 1.0  (NO 0.5 factor — l1loss)
    assert float(full) == pytest.approx(2.0)
    assert float(masked) == pytest.approx(1.0)


def test_masked_mae_no_half_factor_unlike_mse() -> None:
    """For the same input, MAE = 2 * MSE only when |diff| = 1 (sanity); the
    structural point is that there is **no 0.5 factor** on MAE."""
    y = torch.tensor([[2.0, 0.0]])
    t = torch.tensor([[0.0, 0.0]])
    mse = masked_mse_reconstruction_loss(y, t, batch_dim=0)
    mae = masked_mae_reconstruction_loss(y, t, batch_dim=0)
    # MSE: 0.5*(4+0)/1 = 2.0  ; MAE: (2+0)/1 = 2.0
    assert float(mse) == pytest.approx(2.0)
    assert float(mae) == pytest.approx(2.0)
    # Now show the factor is different: a unit-diff makes them divergent.
    y2 = torch.tensor([[1.0, 0.0]])
    mse2 = masked_mse_reconstruction_loss(y2, t, batch_dim=0)  # 0.5*1 = 0.5
    mae2 = masked_mae_reconstruction_loss(y2, t, batch_dim=0)  # 1
    assert float(mse2) == pytest.approx(0.5)
    assert float(mae2) == pytest.approx(1.0)


def test_masked_mae_is_finite_with_nan_target() -> None:
    """MAE masking avoids NaN poisoning."""
    y = torch.zeros(2, 3)
    t = torch.full((2, 3), float("nan"))
    t[0, 0] = 1.0
    loss = masked_mae_reconstruction_loss(y, t, batch_dim=0)
    assert torch.isfinite(loss)


def test_masked_mae_normalizes_by_batch_size() -> None:
    """MAE normalization mirrors MSE (batch-size, not mask.sum)."""
    y1 = torch.ones(1, 4)
    t1 = torch.zeros(1, 4)
    y2 = torch.ones(2, 4)
    t2 = torch.zeros(2, 4)
    l1 = masked_mae_reconstruction_loss(y1, t1, batch_dim=0)
    l2 = masked_mae_reconstruction_loss(y2, t2, batch_dim=0)
    assert float(l1) == pytest.approx(float(l2))


def test_masked_mae_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        masked_mae_reconstruction_loss(
            torch.zeros(2, 3), torch.zeros(2, 4), batch_dim=0
        )


def test_masked_mae_gradient_sign_matches_diff() -> None:
    """dL/dy = sign(y - t) / N for the MAE; verify on a finite case."""
    y = torch.tensor([[2.0, -1.0]], requires_grad=True)
    t = torch.tensor([[0.0, 0.0]])
    loss = masked_mae_reconstruction_loss(y, t, batch_dim=0)
    loss.backward()
    assert y.grad is not None
    # +1/N for the positive diff, -1/N for the negative diff (N=1).
    assert torch.allclose(y.grad, torch.tensor([[1.0, -1.0]]))


# ───────────────────────── compute_reconstruction_loss dispatcher ─────────────────────────


def test_dispatcher_mse_matches_direct_call() -> None:
    """Dispatcher MSE branch == direct call."""
    y = torch.randn(3, 4, 5)
    t = torch.randn(3, 4, 5)
    direct = masked_mse_reconstruction_loss(y, t, batch_dim=0)
    via = compute_reconstruction_loss(y, t, loss_type="MSE", batch_dim=0)
    assert float(direct) == pytest.approx(float(via))


def test_dispatcher_mae_matches_direct_call() -> None:
    """Dispatcher MAE branch == direct call."""
    y = torch.randn(3, 4, 5)
    t = torch.randn(3, 4, 5)
    direct = masked_mae_reconstruction_loss(y, t, batch_dim=0)
    via = compute_reconstruction_loss(y, t, loss_type="MAE", batch_dim=0)
    assert float(direct) == pytest.approx(float(via))


def test_dispatcher_is_case_insensitive() -> None:
    """'mse' / 'Mse' / 'MSE' all route to the MSE kernel."""
    y = torch.randn(2, 3)
    t = torch.randn(2, 3)
    a = compute_reconstruction_loss(y, t, loss_type="MSE")
    b = compute_reconstruction_loss(y, t, loss_type="mse")
    c = compute_reconstruction_loss(y, t, loss_type="Mse")
    assert float(a) == pytest.approx(float(b)) == pytest.approx(float(c))


def test_dispatcher_unknown_loss_type_raises() -> None:
    """A loss_type that is not MSE or MAE is a programming error."""
    y = torch.zeros(2, 3)
    t = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="Unknown loss_type"):
        compute_reconstruction_loss(y, t, loss_type="RMSE")


def test_kl_is_zero_for_standard_normal_posterior() -> None:
    """KL(N(0,1) || N(0,1)) == 0: mu=0, logvar=0 gives zero KL."""
    mu = torch.zeros(4, 2, 3)
    logvar = torch.zeros(4, 2, 3)
    loss = kl_divergence_loss(mu, logvar, channel_dim=0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_kl_is_positive_for_nonstandard_posterior() -> None:
    """A shifted posterior has positive KL."""
    mu = torch.ones(4, 2, 3)
    logvar = torch.zeros(4, 2, 3)
    loss = kl_divergence_loss(mu, logvar, channel_dim=0)
    assert float(loss) > 0


def test_kl_shape_mismatch_raises() -> None:
    """mu and logvar must have the same shape."""
    with pytest.raises(ValueError, match="shape"):
        kl_divergence_loss(torch.zeros(4, 2), torch.zeros(4, 3), channel_dim=0)


def test_per_channel_losses_are_detached() -> None:
    """Per-channel telemetry must not carry gradient (Critical Note #33)."""
    y = torch.zeros(2, 3, 4, requires_grad=True)
    t = torch.ones(2, 3, 4)
    per_channel = per_channel_reconstruction_loss(y, t, channel_dim=1, batch_dim=0)
    assert len(per_channel) == 3
    for v in per_channel:
        assert not v.requires_grad
