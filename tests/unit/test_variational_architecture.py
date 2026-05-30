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


# ───────────────────────── VariationalAutoencoder (Stage 1) ─────────────────────────


def _ae_cfg() -> dict:
    return {
        "in_features": 8,
        "hidden_sizes": [16, 4],
        "num_classes_per_dim": [3],
        "classifier_hidden_size": [4],
        "loss_type_decoder": "MSE",
        "transform": "GRU",
    }


def test_variational_autoencoder_forward_returns_autoencoder_output() -> None:
    """Stage 1 model's forward returns AutoencoderOutput (no logits field)."""
    from neural_data_decoding.models.composite import (
        AutoencoderOutput,
        build_variational_autoencoder,
    )
    ae = build_variational_autoencoder(_ae_cfg())
    out = ae(torch.zeros(2, 5, 8))
    assert isinstance(out, AutoencoderOutput)
    assert out.reconstruction.shape == (2, 5, 8)
    assert out.mu.shape[-1] == 4
    assert out.logvar.shape == out.mu.shape
    assert not hasattr(out, "logits")


def test_variational_autoencoder_has_no_classifier_submodule() -> None:
    """Stage 1 model literally cannot produce logits."""
    from neural_data_decoding.models.composite import build_variational_autoencoder
    ae = build_variational_autoencoder(_ae_cfg())
    assert not hasattr(ae, "classifier")
    # Sanity check: it DOES have encoder/bottleneck/sampling/decoder.
    for attr in ("encoder", "bottleneck", "sampling", "decoder"):
        assert hasattr(ae, attr), f"Stage 1 AE missing expected submodule: {attr}"


def test_variational_autoencoder_rejects_noop_decoder() -> None:
    """Stage 1 with loss_type_decoder='None' is meaningless — raise."""
    from neural_data_decoding.models.composite import build_variational_autoencoder
    cfg = _ae_cfg()
    cfg["loss_type_decoder"] = "None"
    with pytest.raises(ValueError, match="Stage 1.*real decoder"):
        build_variational_autoencoder(cfg)


# ───────────────────────── copy_autoencoder_weights handoff ─────────────────────────


def test_copy_autoencoder_weights_roundtrips_encoder_and_decoder() -> None:
    """Stage 1 → Stage 2 weight handoff: encoder/bottleneck/decoder match exactly."""
    from neural_data_decoding.models.composite import (
        build_variational_autoencoder,
        build_variational_composite,
        copy_autoencoder_weights,
    )
    cfg = _ae_cfg()
    ae = build_variational_autoencoder(cfg)
    full = build_variational_composite(cfg)

    # Perturb the full composite's autoencoder weights so we can verify the copy.
    with torch.no_grad():
        for p in full.encoder.parameters():
            p.add_(0.5)
        for p in full.decoder.parameters():
            p.add_(0.5)

    copy_autoencoder_weights(ae, full)

    # After copy, encoder/bottleneck/decoder match Stage 1 exactly.
    for src_p, dst_p in zip(ae.encoder.parameters(), full.encoder.parameters()):
        assert torch.equal(src_p, dst_p)
    for src_p, dst_p in zip(ae.bottleneck.parameters(), full.bottleneck.parameters()):
        assert torch.equal(src_p, dst_p)
    for src_p, dst_p in zip(ae.decoder.parameters(), full.decoder.parameters()):
        assert torch.equal(src_p, dst_p)


def test_copy_autoencoder_weights_leaves_classifier_alone() -> None:
    """Stage 2's classifier should NOT be touched by the autoencoder weight copy."""
    from neural_data_decoding.models.composite import (
        build_variational_autoencoder,
        build_variational_composite,
        copy_autoencoder_weights,
    )
    cfg = _ae_cfg()
    ae = build_variational_autoencoder(cfg)
    full = build_variational_composite(cfg)

    classifier_snapshot = {
        name: p.detach().clone() for name, p in full.classifier.named_parameters()
    }
    copy_autoencoder_weights(ae, full)

    for name, p in full.classifier.named_parameters():
        assert torch.equal(p, classifier_snapshot[name]), \
            f"Classifier parameter {name} was modified by autoencoder weight copy."


# ───────────────────────── VariationalComposite with confidence heads (Milestone C #7) ─────────────────────────


def _conf_cfg(*, confidence_type: list[str] | str | None) -> dict:
    cfg = {
        "in_features": 8,
        "hidden_sizes": [16, 4],
        "num_classes_per_dim": [3, 4],
        "classifier_hidden_size": [4],
        "loss_type_decoder": "MSE",
        "transform": "GRU",
        "confidence_type": confidence_type,
    }
    return cfg


def test_composite_without_confidence_keeps_existing_contract() -> None:
    """No confidence_type → no heads built; VariationalOutput's confidence fields are None."""
    from neural_data_decoding.models.composite import build_variational_composite
    composite = build_variational_composite(_conf_cfg(confidence_type=None))
    assert composite.trial_confidence_head is None
    assert composite.task_confidence_head is None
    out = composite(torch.zeros(2, 5, 8))
    assert out.trial_confidence is None
    assert out.task_confidence is None


def test_composite_with_trial_confidence_only() -> None:
    """confidence_type=['Trial'] builds only the Trial head."""
    from neural_data_decoding.models.composite import build_variational_composite
    composite = build_variational_composite(_conf_cfg(confidence_type=["Trial"]))
    assert composite.trial_confidence_head is not None
    assert composite.task_confidence_head is None
    out = composite(torch.zeros(2, 5, 8))
    assert out.trial_confidence is not None
    assert out.trial_confidence.shape == (2, 5, 1)
    assert out.task_confidence is None


def test_composite_with_task_confidence_only() -> None:
    """confidence_type=['Task'] builds only the Task head."""
    from neural_data_decoding.models.composite import build_variational_composite
    composite = build_variational_composite(_conf_cfg(confidence_type=["Task"]))
    assert composite.trial_confidence_head is None
    assert composite.task_confidence_head is not None
    out = composite(torch.zeros(2, 5, 8))
    assert out.trial_confidence is None
    assert out.task_confidence is not None
    # num_dims = 2 (matches num_classes_per_dim).
    assert out.task_confidence.shape == (2, 5, 2)


def test_composite_with_both_confidence_types() -> None:
    """confidence_type=['Trial', 'Task'] builds both heads in parallel."""
    from neural_data_decoding.models.composite import build_variational_composite
    composite = build_variational_composite(
        _conf_cfg(confidence_type=["Trial", "Task"]),
    )
    assert composite.trial_confidence_head is not None
    assert composite.task_confidence_head is not None
    out = composite(torch.zeros(2, 5, 8))
    assert out.trial_confidence is not None
    assert out.task_confidence is not None
    # Classification logits unchanged by confidence presence.
    assert len(out.logits) == 2
    assert out.logits[0].shape == (2, 5, 3)


def test_composite_with_string_confidence_type_works() -> None:
    """confidence_type='Trial' (bare string) is accepted (MATLAB sometimes passes single)."""
    from neural_data_decoding.models.composite import build_variational_composite
    composite = build_variational_composite(_conf_cfg(confidence_type="Trial"))
    assert composite.trial_confidence_head is not None
    assert composite.task_confidence_head is None


def test_composite_confidence_type_is_case_insensitive() -> None:
    """'trial' and 'TRIAL' are equivalent to 'Trial' for the config parser."""
    from neural_data_decoding.models.composite import build_variational_composite
    a = build_variational_composite(_conf_cfg(confidence_type=["trial"]))
    b = build_variational_composite(_conf_cfg(confidence_type=["TRIAL"]))
    assert a.trial_confidence_head is not None
    assert b.trial_confidence_head is not None
