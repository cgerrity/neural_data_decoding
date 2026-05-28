"""Unit tests for the Milestone B encoder / bottleneck / classifier surfaces."""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.models.bottleneck import (
    LinearBottleneck,
    PassthroughBottleneck,
    build_bottleneck,
)
from neural_data_decoding.models.decoder import NoopDecoder, build_decoder
from neural_data_decoding.models.classifier import (
    DeepLSTMClassifier,
    MultiHeadClassifier,
)
from neural_data_decoding.models.composite import EncoderClassifierComposite
from neural_data_decoding.models.encoder import (
    SimpleSequenceEncoder,
    build_simple_encoder,
)
from neural_data_decoding.models.registry import (
    build_classifier,
    build_encoder,
    list_classifiers,
    list_encoders,
)


# ───────────────────────── SimpleSequenceEncoder ─────────────────────────


def test_gru_encoder_preserves_time_axis_and_reshapes_features() -> None:
    """A 2-layer GRU encoder maps (B, T, F) → (B, T, hidden_sizes[-1])."""
    enc = SimpleSequenceEncoder(in_features=8, hidden_sizes=[16, 4], transform="GRU")
    y = enc(torch.zeros(3, 5, 8))
    assert y.shape == (3, 5, 4)
    assert enc.out_features == 4


def test_encoder_with_empty_hidden_sizes_is_identity() -> None:
    """No hidden sizes → the encoder returns its input unchanged."""
    enc = SimpleSequenceEncoder(in_features=8, hidden_sizes=[], transform="GRU")
    x = torch.randn(2, 4, 8)
    assert torch.equal(enc(x), x)
    assert enc.out_features == 8


def test_encoder_block_order_matches_matlab_note_27() -> None:
    """Block order is Transform → Dropout → Norm → Activation.

    Verified structurally by checking each block's child attribute order
    against the Critical Note #27 invariant.
    """
    enc = SimpleSequenceEncoder(
        in_features=4,
        hidden_sizes=[8],
        transform="Feedforward",
        dropout=0.5,
        want_normalization=True,
        activation="ReLU",
    )
    block = enc.blocks[0]
    # All four sub-modules are present and in this order.
    expected_attrs = ["transform_layer", "dropout", "norm", "activation"]
    actual = [
        name for name, _ in block.named_children() if name in expected_attrs
    ]
    assert actual == expected_attrs


def test_encoder_lstm_transform_runs() -> None:
    """LSTM transform produces the right shape too."""
    enc = SimpleSequenceEncoder(in_features=5, hidden_sizes=[10, 3], transform="LSTM")
    y = enc(torch.zeros(2, 7, 5))
    assert y.shape == (2, 7, 3)


def test_encoder_feedforward_transform_runs() -> None:
    """Feedforward transform applies Linear per-timestep."""
    enc = SimpleSequenceEncoder(
        in_features=4, hidden_sizes=[6, 2], transform="Feedforward"
    )
    y = enc(torch.zeros(2, 7, 4))
    assert y.shape == (2, 7, 2)


def test_encoder_invalid_transform_rejected() -> None:
    """An unsupported transform value raises clearly."""
    with pytest.raises(ValueError, match="Unsupported transform"):
        SimpleSequenceEncoder(in_features=4, hidden_sizes=[8], transform="Magic")


def test_encoder_builder_reads_cfg() -> None:
    """``build_simple_encoder`` constructs the same module from a cfg dict."""
    enc = build_simple_encoder(
        {
            "in_features": 5,
            "hidden_sizes": [6, 3],
            "transform": "GRU",
            "dropout": 0.3,
        }
    )
    assert isinstance(enc, SimpleSequenceEncoder)
    assert enc.out_features == 3


# ───────────────────────── Bottlenecks ─────────────────────────


def test_passthrough_bottleneck_returns_input_unchanged() -> None:
    """Identity behavior; ``out_features == in_features``."""
    b = PassthroughBottleneck(in_features=12)
    x = torch.randn(3, 5, 12)
    assert torch.equal(b(x), x)
    assert b.out_features == 12


def test_linear_bottleneck_reshapes_last_axis() -> None:
    """Applies the FC per-timestep, reshaping the last axis."""
    b = LinearBottleneck(in_features=12, hidden_size=4)
    y = b(torch.zeros(3, 5, 12))
    assert y.shape == (3, 5, 4)
    assert b.out_features == 4


def test_linear_bottleneck_uses_he_initialization() -> None:
    """He init produces a non-zero variance that scales with fan_in.

    A truly zero-init weight matrix would give zero output for any input;
    He init produces a finite-variance distribution. The test asserts the
    initialised weight has positive variance — sanity check on the call to
    ``nn.init.kaiming_normal_``.
    """
    b = LinearBottleneck(in_features=12, hidden_size=4)
    assert b.linear.weight.var().item() > 0


def test_build_bottleneck_dispatch() -> None:
    """No hidden_size → passthrough; with hidden_size → linear."""
    p = build_bottleneck({"in_features": 8})
    assert isinstance(p, PassthroughBottleneck)
    L = build_bottleneck({"in_features": 8, "bottleneck_hidden_size": 4})
    assert isinstance(L, LinearBottleneck)
    assert L.out_features == 4


# ───────────────────────── DeepLSTMClassifier ─────────────────────────


def test_deep_lstm_returns_one_logit_tensor_per_dim() -> None:
    """One ``(B, T, K_d)`` tensor per output dimension."""
    cls = DeepLSTMClassifier(
        in_features=4,
        num_classes_per_dim=[3, 5],
        hidden_sizes=[8, 8],
        dropout=0.5,
    )
    outs = cls(torch.zeros(2, 6, 4))
    assert len(outs) == 2
    assert outs[0].shape == (2, 6, 3)
    assert outs[1].shape == (2, 6, 5)


def test_deep_lstm_per_dim_trunks_are_distinct() -> None:
    """Each dim has its own LSTM stack — distinct parameter sets."""
    cls = DeepLSTMClassifier(
        in_features=4,
        num_classes_per_dim=[3, 3],
        hidden_sizes=[8],
        dropout=0.0,
    )
    stack0 = list(cls.stacks[0].lstms[0].parameters())
    stack1 = list(cls.stacks[1].lstms[0].parameters())
    # Same shapes (constructed identically) but they are *different* tensors.
    for p0, p1 in zip(stack0, stack1):
        assert p0.shape == p1.shape
        assert p0.data_ptr() != p1.data_ptr()


def test_deep_lstm_rejects_empty_hidden_sizes() -> None:
    """The Deep variant must have at least one LSTM layer."""
    with pytest.raises(ValueError, match="hidden_sizes must be non-empty"):
        DeepLSTMClassifier(
            in_features=4,
            num_classes_per_dim=[3],
            hidden_sizes=[],
            dropout=0.5,
        )


def test_deep_lstm_builder_via_registry_uses_correct_dropout() -> None:
    """Building through the registry stamps in the dropout level from the name."""
    cls_05 = build_classifier(
        "Deep LSTM - Dropout 0.5",
        {
            "in_features": 4,
            "num_classes_per_dim": [3],
            "classifier_hidden_size": [8],
        },
    )
    cls_025 = build_classifier(
        "Deep LSTM - Dropout 0.25",
        {
            "in_features": 4,
            "num_classes_per_dim": [3],
            "classifier_hidden_size": [8],
        },
    )
    assert isinstance(cls_05, DeepLSTMClassifier)
    assert cls_05.dropout == 0.5
    assert cls_025.dropout == 0.25


# ───────────────────────── Registry ─────────────────────────


def test_registry_has_milestone_b_entries() -> None:
    """After importing the models package, the Milestone B keys are present."""
    encs = list_encoders()
    cls = list_classifiers()
    assert "GRU" in encs
    assert "LSTM" in encs
    assert "Feedforward" in encs
    assert "Logistic" in cls
    assert "Deep LSTM - Dropout 0.5" in cls
    assert "Deep LSTM - Dropout 0.25" in cls


def test_registry_dispatches_gru_encoder() -> None:
    """``build_encoder('GRU', cfg)`` produces a SimpleSequenceEncoder with GRU transform."""
    enc = build_encoder(
        "GRU",
        {
            "in_features": 4,
            "hidden_sizes": [8],
            "dropout": 0.0,
        },
    )
    assert isinstance(enc, SimpleSequenceEncoder)
    assert enc.transform == "GRU"


# ───────────────────────── Composite ─────────────────────────


def test_composite_forward_returns_per_dim_logits() -> None:
    """``EncoderClassifierComposite(x)`` matches the classifier-only contract."""
    encoder = SimpleSequenceEncoder(in_features=5, hidden_sizes=[8, 4], transform="GRU")
    bottleneck = PassthroughBottleneck(in_features=4)
    classifier = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3, 2])
    model = EncoderClassifierComposite(
        encoder=encoder, bottleneck=bottleneck, classifier=classifier
    )
    outs = model(torch.zeros(2, 7, 5))
    assert len(outs) == 2
    assert outs[0].shape == (2, 7, 3)
    assert outs[1].shape == (2, 7, 2)


def test_composite_with_linear_bottleneck() -> None:
    """LinearBottleneck reshapes encoder output before classifier."""
    encoder = SimpleSequenceEncoder(in_features=5, hidden_sizes=[8], transform="GRU")
    bottleneck = LinearBottleneck(in_features=8, hidden_size=4)
    classifier = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    model = EncoderClassifierComposite(
        encoder=encoder, bottleneck=bottleneck, classifier=classifier
    )
    outs = model(torch.zeros(2, 7, 5))
    assert outs[0].shape == (2, 7, 3)


# ───────────────────────── Decoder stub ─────────────────────────


def test_noop_decoder_returns_input_unchanged() -> None:
    """The Milestone B stub is an identity transform."""
    dec = NoopDecoder()
    x = torch.randn(2, 5, 4)
    assert torch.equal(dec(x), x)


def test_build_decoder_returns_noop_for_loss_type_none() -> None:
    """``loss_type_decoder='None'`` → :class:`NoopDecoder`."""
    dec = build_decoder({"loss_type_decoder": "None"})
    assert isinstance(dec, NoopDecoder)


def test_build_decoder_rejects_non_none_loss_type_until_milestone_c() -> None:
    """Anything other than ``'None'`` raises until the full decoder lands."""
    with pytest.raises(NotImplementedError, match="Milestone C"):
        build_decoder({"loss_type_decoder": "MSE"})


def test_composite_gradients_flow_through_all_three_subnetworks() -> None:
    """A backward call updates encoder, bottleneck, AND classifier parameters."""
    torch.manual_seed(0)
    encoder = SimpleSequenceEncoder(in_features=4, hidden_sizes=[8], transform="GRU")
    bottleneck = LinearBottleneck(in_features=8, hidden_size=4)
    classifier = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    model = EncoderClassifierComposite(
        encoder=encoder, bottleneck=bottleneck, classifier=classifier
    )

    x = torch.randn(2, 5, 4)
    outs = model(x)
    # Sum over all logits — guaranteed non-zero gradient.
    outs[0].sum().backward()

    grads = {
        "encoder": next(encoder.parameters()).grad,
        "bottleneck": bottleneck.linear.weight.grad,
        "classifier_head": classifier.heads[0].weight.grad,
    }
    for name, g in grads.items():
        assert g is not None, f"{name} did not receive a gradient."
        assert g.abs().sum().item() > 0, f"{name}'s gradient is identically zero."
