"""Unit tests for the Milestone C variational architecture wiring."""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.models.bottleneck import LinearBottleneck
from neural_data_decoding.models.classifier import DeepLSTMClassifier
from neural_data_decoding.models.composite import (
    VariationalComposite,
    VariationalOutput,
    build_variational_composite,
)
from neural_data_decoding.models.decoder import (
    NoopDecoder,
    SimpleSequenceDecoder,
    build_decoder,
)
from neural_data_decoding.models.encoder import SimpleSequenceEncoder
from neural_data_decoding.models.layers.sampling import SamplingLayer


# ───────────────────────── SimpleSequenceDecoder ─────────────────────────


def test_decoder_maps_latent_to_output_features() -> None:
    """(B, T, latent) → (B, T, output_features)."""
    dec = SimpleSequenceDecoder(
        latent_size=4, hidden_sizes=[4, 8], output_features=6, transform="GRU"
    )
    y = dec(torch.zeros(2, 5, 4))
    assert y.shape == (2, 5, 6)


def test_decoder_lstm_transform() -> None:
    """LSTM transform decoder produces the right shape."""
    dec = SimpleSequenceDecoder(
        latent_size=3, hidden_sizes=[3, 6], output_features=5, transform="LSTM"
    )
    assert dec(torch.zeros(2, 7, 3)).shape == (2, 7, 5)


def test_decoder_gradients_flow() -> None:
    """A backward through the decoder updates both the stack and output FC."""
    dec = SimpleSequenceDecoder(
        latent_size=4, hidden_sizes=[4], output_features=3, transform="GRU"
    )
    out = dec(torch.randn(2, 5, 4))
    out.sum().backward()
    assert dec.output.weight.grad is not None
    assert dec.output.weight.grad.abs().sum() > 0
    assert next(dec.stack.parameters()).grad is not None


def test_decoder_rejects_empty_hidden_sizes() -> None:
    """The decoder stack must have at least one layer."""
    with pytest.raises(ValueError, match="hidden_sizes must be non-empty"):
        SimpleSequenceDecoder(
            latent_size=4, hidden_sizes=[], output_features=3
        )


# ───────────────────────── build_decoder ─────────────────────────


def test_build_decoder_none_returns_noop() -> None:
    """loss_type_decoder='None' → NoopDecoder."""
    dec = build_decoder({"loss_type_decoder": "None"})
    assert isinstance(dec, NoopDecoder)


def test_build_decoder_mse_reverses_hidden_sizes() -> None:
    """The decoder stack uses reversed(encoder_hidden + [latent]) (MATLAB order)."""
    dec = build_decoder(
        {
            "loss_type_decoder": "MSE",
            "latent_size": 4,
            "encoder_hidden_sizes": [16, 8],
            "output_features": 10,
            "transform": "GRU",
        }
    )
    assert isinstance(dec, SimpleSequenceDecoder)
    # reversed([16, 8, 4]) = [4, 8, 16] → stack hidden sizes.
    assert dec.stack.hidden_sizes == (4, 8, 16)
    assert dec.output_features == 10


def test_build_decoder_missing_key_raises() -> None:
    """A non-None decoder needs latent_size / encoder_hidden_sizes / output_features."""
    with pytest.raises(KeyError):
        build_decoder({"loss_type_decoder": "MSE", "latent_size": 4})


# ───────────────────────── VariationalComposite (direct) ─────────────────────────


def _make_variational(*, with_decoder: bool) -> VariationalComposite:
    latent = 4
    encoder = SimpleSequenceEncoder(
        in_features=6, hidden_sizes=[8], transform="GRU"
    )
    bottleneck = LinearBottleneck(in_features=8, hidden_size=2 * latent)
    sampling = SamplingLayer(channel_dim=-1)
    decoder = (
        SimpleSequenceDecoder(
            latent_size=latent, hidden_sizes=[latent, 8], output_features=6
        )
        if with_decoder
        else NoopDecoder()
    )
    classifier = DeepLSTMClassifier(
        in_features=latent, num_classes_per_dim=[3, 2], hidden_sizes=[4]
    )
    return VariationalComposite(
        encoder=encoder,
        bottleneck=bottleneck,
        sampling=sampling,
        decoder=decoder,
        classifier=classifier,
    )


def test_variational_forward_returns_all_four_outputs() -> None:
    """forward returns logits (per-dim), reconstruction, mu, logvar."""
    model = _make_variational(with_decoder=True)
    model.eval()
    out = model(torch.randn(2, 5, 6))
    assert isinstance(out, VariationalOutput)
    assert len(out.logits) == 2
    assert out.logits[0].shape == (2, 5, 3)
    assert out.logits[1].shape == (2, 5, 2)
    assert out.reconstruction is not None
    assert out.reconstruction.shape == (2, 5, 6)  # reconstructs input features
    assert out.mu.shape == (2, 5, 4)              # latent
    assert out.logvar.shape == (2, 5, 4)


def test_variational_noop_decoder_yields_none_reconstruction() -> None:
    """With a NoopDecoder, reconstruction is None and has_reconstruction is False."""
    model = _make_variational(with_decoder=False)
    model.eval()
    out = model(torch.randn(2, 5, 6))
    assert out.reconstruction is None
    assert model.has_reconstruction is False


def test_variational_eval_is_deterministic() -> None:
    """Eval mode (Z=mu) gives identical logits across calls (Critical Note #35)."""
    model = _make_variational(with_decoder=True)
    model.eval()
    x = torch.randn(2, 5, 6)
    out1 = model(x)
    out2 = model(x)
    assert torch.equal(out1.logits[0], out2.logits[0])
    assert torch.equal(out1.reconstruction, out2.reconstruction)


def test_variational_train_is_stochastic() -> None:
    """Train mode samples Z, so reconstruction differs across calls."""
    torch.manual_seed(0)
    model = _make_variational(with_decoder=True)
    model.train()
    x = torch.randn(2, 5, 6)
    out1 = model(x)
    out2 = model(x)
    assert not torch.equal(out1.reconstruction, out2.reconstruction)


def test_variational_handles_nan_input() -> None:
    """NaN in the input is zeroed before the encoder (no NaN in outputs)."""
    model = _make_variational(with_decoder=True)
    model.eval()
    x = torch.randn(2, 5, 6)
    x[0, 0, 0] = float("nan")
    out = model(x)
    assert torch.isfinite(out.logits[0]).all()
    assert torch.isfinite(out.reconstruction).all()


def test_variational_gradient_flows_through_all_subnetworks() -> None:
    """A combined (classification + reconstruction) loss reaches every subnetwork."""
    torch.manual_seed(0)
    model = _make_variational(with_decoder=True)
    model.train()
    out = model(torch.randn(2, 5, 6))
    loss = out.logits[0].sum() + out.logits[1].sum() + out.reconstruction.sum()
    loss.backward()

    checks = {
        "encoder": next(model.encoder.parameters()).grad,
        "bottleneck": model.bottleneck.linear.weight.grad,
        "decoder_output": model.decoder.output.weight.grad,
        "classifier_head": model.classifier.stacks[0].head.weight.grad,
    }
    for name, grad in checks.items():
        assert grad is not None, f"{name} got no gradient"
        assert grad.abs().sum() > 0, f"{name} gradient is identically zero"


# ───────────────────────── build_variational_composite ─────────────────────────


def test_build_variational_composite_from_config() -> None:
    """The config-driven builder assembles a working composite."""
    model = build_variational_composite(
        {
            "in_features": 6,
            "hidden_sizes": [8, 4],          # encoder [8], latent 4
            "num_classes_per_dim": [3, 2],
            "classifier_hidden_size": [4],
            "transform": "GRU",
            "loss_type_decoder": "MSE",
        }
    )
    model.eval()
    out = model(torch.randn(2, 5, 6))
    assert out.mu.shape[-1] == 4                 # latent = last hidden size
    assert out.reconstruction.shape == (2, 5, 6)
    # Bottleneck emits 2*latent for the mu|logvar split.
    assert model.bottleneck.out_features == 8


def test_build_variational_composite_none_decoder() -> None:
    """loss_type_decoder='None' wires a NoopDecoder (no reconstruction)."""
    model = build_variational_composite(
        {
            "in_features": 6,
            "hidden_sizes": [8, 4],
            "num_classes_per_dim": [3],
            "classifier_hidden_size": [4],
            "loss_type_decoder": "None",
        }
    )
    assert model.has_reconstruction is False


def test_build_variational_composite_requires_two_hidden_sizes() -> None:
    """Need at least one encoder layer + a trailing latent size."""
    with pytest.raises(ValueError, match="at least 2 entries"):
        build_variational_composite(
            {
                "in_features": 6,
                "hidden_sizes": [4],
                "num_classes_per_dim": [3],
                "classifier_hidden_size": [4],
            }
        )
