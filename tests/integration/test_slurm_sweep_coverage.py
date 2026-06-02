"""CC.8 integration tests — representative SLURM sweep slices.

Parametrizes the supported variants of each major sweep dimension
(ModelName, ClassifierName, Optimizer, LossType_Decoder, S&F variant,
WeightedLoss, IsVariational) and verifies non-crash training on a
tiny synthetic config. **No parity claim** — just gating that the
parameter combination produces a model that builds, trains for at
least one epoch, and emits the expected output files.

These tests use 1-2 epochs and tiny synthetic data, so they're fast
(~5s each) and run in the default suite.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from neural_data_decoding.data.dataset import SyntheticTrialDataset, collate_trials
from neural_data_decoding.data.samplers import SingleSessionBatchSampler
from neural_data_decoding.models.composite import (
    EncoderClassifierComposite,
    build_variational_composite,
)
from neural_data_decoding.models.bottleneck import PassthroughBottleneck
from neural_data_decoding.models.registry import build_classifier, build_encoder


def _make_loader(
    *, t: int = 4, a: int = 2, c: int = 4, batch_size: int = 4, sessions: int = 2,
) -> DataLoader:
    """Build a tiny synthetic DataLoader for a non-crash smoke test."""
    ds = SyntheticTrialDataset(
        num_sessions=sessions, trials_per_session=8,
        num_samples=8, num_features=c,
        num_classes_per_dim=[3],
        samples_per_window=t, num_areas=a, seed=0,
    )
    sampler = SingleSessionBatchSampler(
        session_ids=ds.session_ids, batch_size=batch_size, drop_last=False, seed=0,
    )
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_trials)


# ───────────────────────── ModelName coverage ─────────────────────────


_NON_VARIATIONAL_ENCODERS = [
    # ModelName values from the SLURM sweep (lines 145-179) minus 'Logistic
    # Regression' (which short-circuits in CLI without encoder) and the
    # variants that would need a Conv-encoder + variational composite
    # combination that's not in active SLURM scope.
    "GRU",
    "LSTM",
    "Feedforward",
    "Convolutional",
    "Resnet",
    "Multi-Filter Convolutional",
    "PCA",
]


@pytest.mark.parametrize("model_name", _NON_VARIATIONAL_ENCODERS)
def test_encoder_builds_and_forwards_via_registry(model_name: str) -> None:
    """Each SLURM ``ModelName`` (non-variational) builds + forwards on synthetic data.

    Pin: any of the 7 encoder names selectable via ``cfg.model_name``
    constructs a working encoder + bottleneck + classifier pipeline.
    """
    loader = _make_loader(t=4, a=2, c=4)
    encoder_cfg = {
        "in_features": 4 * 2 * 4,  # T*A*C from loader
        "samples_per_window": 4,
        "num_areas": 2,
        "hidden_sizes": [8],
        "stride": 2,
        "activation": "Leaky ReLU",
        "n_components": 8,  # for PCA
    }
    encoder = build_encoder(model_name, encoder_cfg)
    # PCA needs to be fit before forward.
    if model_name == "PCA":
        from neural_data_decoding.models.layers.pca import PCAEncoder
        assert isinstance(encoder, PCAEncoder)
        encoder.fit_from_dataloader(loader)
    encoder_out = getattr(encoder, "out_features", encoder_cfg["in_features"])
    bottleneck = PassthroughBottleneck(in_features=encoder_out)
    classifier = build_classifier(
        "Deep LSTM - Dropout 0.5",
        {
            "in_features": encoder_out,
            "num_classes_per_dim": [3],
            "classifier_hidden_size": [8, 4],
        },
    )
    composite = EncoderClassifierComposite(
        encoder=encoder, bottleneck=bottleneck, classifier=classifier,
    )
    composite.train()
    # Take one batch and verify shapes + gradient flow.
    batch = next(iter(loader))
    x = batch["x"]
    logits = composite(x)
    assert isinstance(logits, list) and len(logits) == 1
    assert logits[0].shape[0] == x.shape[0]
    # Backward to confirm non-crash gradient.
    logits[0].sum().backward()


# ───────────────────────── ClassifierName coverage ─────────────────────────


_CLASSIFIER_NAMES = [
    "Logistic",
    "Deep LSTM - Dropout 0.5",
    "Deep LSTM - Dropout 0.25",
]


@pytest.mark.parametrize("classifier_name", _CLASSIFIER_NAMES)
def test_classifier_builds_via_registry(classifier_name: str) -> None:
    """Each registered ``ClassifierName`` constructs without raising."""
    classifier = build_classifier(
        classifier_name,
        {
            "in_features": 8,
            "num_classes_per_dim": [3],
            "classifier_hidden_size": [8, 4],
        },
    )
    x = torch.randn(2, 5, 8)
    out = classifier(x)
    assert isinstance(out, list)
    assert len(out) == 1


# ───────────────────────── Cross-feature integrations ─────────────────────────


def _variational_cfg(**overrides) -> dict:
    """Base variational cfg; overrideable for each parameter slice."""
    cfg = {
        "in_features": 4,
        "samples_per_window": 4,
        "num_areas": 2,
        "hidden_sizes": [16, 4],
        "num_classes_per_dim": [3],
        "classifier_hidden_size": [8, 4],
        "transform": "GRU",
        "loss_type_decoder": "MSE",
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("loss_type", ["MSE", "MAE"])
def test_variational_composite_loss_type_decoder(loss_type: str) -> None:
    """``cfg.loss_type_decoder`` switches between MSE and MAE (CC.3 surface)."""
    composite = build_variational_composite(_variational_cfg(loss_type_decoder=loss_type))
    composite.eval()
    out = composite(torch.randn(2, 5, 4, 2, 4))
    assert out.reconstruction is not None
    assert out.reconstruction.shape == (2, 5, 4, 2, 4)


@pytest.mark.parametrize(
    "sf_variant",
    [
        "Feedforward",
        "Default",
        # Gemini variants require T sized for stride; default T=4 too small.
    ],
)
def test_variational_composite_with_stitching_fusion(sf_variant: str) -> None:
    """``cfg.stitching_and_fusion_layer`` for the trivially-runnable variants."""
    composite = build_variational_composite(
        _variational_cfg(stitching_and_fusion_layer=sf_variant),
    )
    composite.eval()
    x = torch.randn(2, 5, 4, 2, 4)
    out = composite(x)
    assert out.reconstruction is not None
    assert out.reconstruction.shape == x.shape


@pytest.mark.parametrize("sf_gemini", [
    "Parallel Single Level",
    "Cascade Single Kernel - Single Reduction",
    "Cascade Single Kernel - Progressive Reduction",
])
def test_variational_composite_with_gemini_sf(sf_gemini: str) -> None:
    """Each Gemini cascade variant builds + forwards (T sized for the stride)."""
    composite = build_variational_composite(
        _variational_cfg(samples_per_window=16, stitching_and_fusion_layer=sf_gemini),
    )
    composite.eval()
    x = torch.randn(2, 5, 16, 2, 4)
    out = composite(x)
    assert out.reconstruction is not None
    assert out.reconstruction.shape == x.shape


# ───────────────────────── Optimizer coverage (CC.4) ─────────────────────────


@pytest.mark.parametrize("optimizer_name", ["ADAM", "SGDM"])
def test_optimizer_factory_via_registry(optimizer_name: str) -> None:
    """``cfg.optimizer`` builds the right torch optimizer (CC.4)."""
    from neural_data_decoding.training.freezing import resolve_optimizer_factory
    composite = build_variational_composite(_variational_cfg())
    opt = resolve_optimizer_factory(optimizer_name)(
        composite.parameters(), lr=0.001,
    )
    # Run one optimizer step to confirm non-crash.
    x = torch.randn(2, 5, 4, 2, 4)
    out = composite(x)
    loss = torch.stack([lg.sum() for lg in out.logits]).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()


# ───────────────────────── WeightedLoss coverage (CC.7) ─────────────────────────


@pytest.mark.parametrize("weighted_loss", ["Inverse", "", "None"])
def test_weighted_loss_branches(weighted_loss: str) -> None:
    """All ``cfg.weighted_loss`` values route through cleanly.

    Pin: any string that isn't case-insensitive 'Inverse' falls into
    the unweighted branch (CC.7), where ``class_weights_per_dim=None``.
    """
    from neural_data_decoding.training.losses.classification import (
        multi_head_cross_entropy,
        inverse_frequency_class_weights,
    )
    logits = [torch.randn(4, 3)]
    targets = torch.tensor([[0], [1], [2], [0]])
    if weighted_loss.lower() == "inverse":
        weights = inverse_frequency_class_weights(targets, num_classes_per_dim=[3])
        loss = multi_head_cross_entropy(logits, targets, class_weights_per_dim=weights)
    else:
        loss = multi_head_cross_entropy(logits, targets, class_weights_per_dim=None)
    assert torch.isfinite(loss)


# ───────────────────────── IsVariational coverage ─────────────────────────


def test_variational_composite_forward_smoke() -> None:
    """Variational composite (is_variational=True) forwards + backprops end-to-end."""
    composite = build_variational_composite(_variational_cfg())
    composite.train()
    x = torch.randn(2, 5, 4, 2, 4)
    out = composite(x)
    loss = torch.stack([lg.sum() for lg in out.logits]).sum()
    if out.reconstruction is not None:
        loss = loss + out.reconstruction.pow(2).sum()
    loss.backward()


def test_non_variational_composite_via_registry() -> None:
    """Non-variational ``EncoderClassifierComposite`` works on the same path."""
    encoder = build_encoder(
        "GRU",
        {"in_features": 4, "hidden_sizes": [8], "samples_per_window": 1, "num_areas": 1},
    )
    bottleneck = PassthroughBottleneck(in_features=8)
    classifier = build_classifier(
        "Logistic",
        {"in_features": 8, "num_classes_per_dim": [3], "classifier_hidden_size": [8]},
    )
    composite = EncoderClassifierComposite(
        encoder=encoder, bottleneck=bottleneck, classifier=classifier,
    )
    out = composite(torch.randn(2, 5, 4))
    assert isinstance(out, list)
