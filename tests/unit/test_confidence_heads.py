"""Unit tests for the confidence head modules (Milestone C #7)."""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.models.classifier import DeepLSTMClassifier
from neural_data_decoding.models.confidence_heads import (
    TaskConfidenceHead,
    TrialConfidenceHead,
)


# ───────────────────────── TrialConfidenceHead ─────────────────────────


def test_trial_confidence_head_output_shape() -> None:
    """``(B, T, latent) → (B, T, 1)`` via FC + sigmoid."""
    head = TrialConfidenceHead(in_features=8)
    z = torch.randn(3, 5, 8)
    out = head(z)
    assert out.shape == (3, 5, 1)


def test_trial_confidence_head_outputs_are_in_unit_interval() -> None:
    """Sigmoid output is in [0, 1] (closed — saturates at extremes)."""
    torch.manual_seed(0)
    head = TrialConfidenceHead(in_features=4)
    z = torch.randn(8, 6, 4) * 10
    out = head(z)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_trial_confidence_head_rejects_wrong_input_dim() -> None:
    head = TrialConfidenceHead(in_features=8)
    with pytest.raises(ValueError, match="in_features"):
        head(torch.zeros(2, 3, 7))


def test_trial_confidence_head_rejects_zero_input_dim() -> None:
    with pytest.raises(ValueError, match="in_features"):
        TrialConfidenceHead(in_features=0)


# ───────────────────────── TaskConfidenceHead ─────────────────────────


def test_task_confidence_head_output_shape() -> None:
    """Per-dim features → stacked ``(B, T, num_dims)`` sigmoid output."""
    head = TaskConfidenceHead(in_features_per_dim=[4, 4])
    feats = [torch.randn(3, 5, 4), torch.randn(3, 5, 4)]
    out = head(feats)
    assert out.shape == (3, 5, 2)


def test_task_confidence_head_supports_variable_per_dim_sizes() -> None:
    """Different penultimate sizes per dim are allowed (general case)."""
    head = TaskConfidenceHead(in_features_per_dim=[4, 8])
    feats = [torch.randn(2, 3, 4), torch.randn(2, 3, 8)]
    out = head(feats)
    assert out.shape == (2, 3, 2)


def test_task_confidence_head_rejects_wrong_feature_count() -> None:
    head = TaskConfidenceHead(in_features_per_dim=[4, 4])
    with pytest.raises(ValueError, match="Expected 2"):
        head([torch.randn(2, 3, 4)])


def test_task_confidence_head_rejects_empty_per_dim_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TaskConfidenceHead(in_features_per_dim=[])


def test_task_confidence_head_outputs_are_in_unit_interval() -> None:
    """All per-dim outputs are sigmoid-bounded [0, 1] (closed)."""
    torch.manual_seed(0)
    head = TaskConfidenceHead(in_features_per_dim=[4, 4, 4])
    feats = [torch.randn(8, 6, 4) * 10 for _ in range(3)]
    out = head(feats)
    assert torch.all(out >= 0) and torch.all(out <= 1)


# ───────────────────────── DeepLSTMClassifier.forward_with_features ─────────────────────────


def test_classifier_forward_with_features_matches_forward_for_logits() -> None:
    """Logits returned by forward and forward_with_features are identical."""
    torch.manual_seed(0)
    clf = DeepLSTMClassifier(
        in_features=4, num_classes_per_dim=[3, 5],
        hidden_sizes=[6], dropout=0.0,
    )
    clf.eval()  # dropout=0 anyway, but eval disables any future stochasticity
    z = torch.randn(2, 5, 4)
    logits_via_forward = clf(z)
    features, logits_via_features = clf.forward_with_features(z)
    for a, b in zip(logits_via_forward, logits_via_features):
        assert torch.allclose(a, b)
    # Features shape: (B, T, last_hidden_size) per dim.
    assert len(features) == 2
    for f in features:
        assert f.shape == (2, 5, 6)


def test_classifier_penultimate_features_are_distinct_per_dim() -> None:
    """Each dim's stack has its own weights, so its penultimate output differs."""
    torch.manual_seed(0)
    clf = DeepLSTMClassifier(
        in_features=4, num_classes_per_dim=[3, 3],
        hidden_sizes=[6], dropout=0.0,
    )
    clf.eval()
    z = torch.randn(2, 5, 4)
    features, _ = clf.forward_with_features(z)
    # The two stacks should have different parameters → different outputs.
    assert not torch.allclose(features[0], features[1])


def test_classifier_penultimate_size_attr_matches_features() -> None:
    """Each stack's penultimate_size attribute matches its forward output dim."""
    clf = DeepLSTMClassifier(
        in_features=4, num_classes_per_dim=[3],
        hidden_sizes=[8, 6], dropout=0.0,
    )
    assert clf.stacks[0].penultimate_size == 6  # last hidden size
    z = torch.zeros(2, 5, 4)
    features, _ = clf.forward_with_features(z)
    assert features[0].shape == (2, 5, 6)
