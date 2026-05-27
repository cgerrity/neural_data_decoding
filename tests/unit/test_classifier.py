"""Tests for :mod:`neural_data_decoding.models.classifier`."""

from __future__ import annotations

import pytest
import torch

from neural_data_decoding.models.classifier import (
    MultiHeadClassifier,
    build_logistic_classifier,
)
from neural_data_decoding.models.registry import build_classifier, list_classifiers


# ───────────────────────── MultiHeadClassifier ─────────────────────────


def test_multihead_constructor_validates_in_features() -> None:
    """Non-positive ``in_features`` is rejected."""
    with pytest.raises(ValueError, match="in_features"):
        MultiHeadClassifier(in_features=0, num_classes_per_dim=[2])


def test_multihead_constructor_validates_num_classes() -> None:
    """Empty or non-positive class counts are rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        MultiHeadClassifier(in_features=4, num_classes_per_dim=[])
    with pytest.raises(ValueError, match="must be > 0"):
        MultiHeadClassifier(in_features=4, num_classes_per_dim=[2, 0, 3])


def test_multihead_constructor_validates_dropout() -> None:
    """Dropout outside ``[0, 1)`` is rejected."""
    with pytest.raises(ValueError, match="dropout"):
        MultiHeadClassifier(in_features=4, num_classes_per_dim=[2], dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        MultiHeadClassifier(in_features=4, num_classes_per_dim=[2], dropout=-0.1)


def test_multihead_forward_returns_per_dim_logits() -> None:
    """Each head produces a logit tensor with the configured class count."""
    head = MultiHeadClassifier(in_features=8, num_classes_per_dim=[3, 4, 5])
    x = torch.zeros(2, 5, 8)  # (batch, time, features)
    outputs = head(x)
    assert len(outputs) == 3
    assert outputs[0].shape == (2, 5, 3)
    assert outputs[1].shape == (2, 5, 4)
    assert outputs[2].shape == (2, 5, 5)


def test_multihead_forward_supports_flat_input() -> None:
    """The classifier works on (batch, features) inputs as well as sequences."""
    head = MultiHeadClassifier(in_features=8, num_classes_per_dim=[3])
    x = torch.zeros(4, 8)
    outputs = head(x)
    assert outputs[0].shape == (4, 3)


def test_multihead_forward_rejects_wrong_feature_dim() -> None:
    """Input whose last dim doesn't match ``in_features`` raises ``ValueError``."""
    head = MultiHeadClassifier(in_features=8, num_classes_per_dim=[3])
    bad = torch.zeros(2, 5, 7)  # last dim should be 8
    with pytest.raises(ValueError, match="in_features"):
        head(bad)


def test_multihead_with_dropout_changes_train_vs_eval() -> None:
    """Dropout layer is active in train mode, inactive in eval mode.

    We don't assert on exact dropout values (stochastic) — only that
    train-mode forward varies across calls while eval-mode is deterministic.
    """
    head = MultiHeadClassifier(in_features=8, num_classes_per_dim=[3], dropout=0.5)
    x = torch.ones(4, 8)
    torch.manual_seed(0)
    head.eval()
    a = head(x)[0]
    b = head(x)[0]
    torch.testing.assert_close(a, b)  # deterministic in eval


# ───────────────────────── build_logistic_classifier ─────────────────────────


def test_build_logistic_classifier_from_cfg() -> None:
    """The builder returns a configured :class:`MultiHeadClassifier`."""
    cfg = {"in_features": 10, "num_classes_per_dim": [2, 4]}
    classifier = build_logistic_classifier(cfg)
    assert isinstance(classifier, MultiHeadClassifier)
    assert classifier.in_features == 10
    assert classifier.num_classes_per_dim == (2, 4)


def test_build_logistic_classifier_honors_dropout_cfg() -> None:
    """``classifier_dropout`` is plumbed through to the module."""
    cfg = {
        "in_features": 10,
        "num_classes_per_dim": [3],
        "classifier_dropout": 0.25,
    }
    classifier = build_logistic_classifier(cfg)
    assert isinstance(classifier.dropout, torch.nn.Dropout)
    assert classifier.dropout.p == 0.25


def test_build_logistic_classifier_missing_required_key_raises() -> None:
    """A missing required key produces a clear ``KeyError``."""
    with pytest.raises(KeyError, match="in_features"):
        build_logistic_classifier({"num_classes_per_dim": [3]})


# ───────────────────────── registry integration ─────────────────────────


def test_logistic_is_registered_under_its_matlab_name() -> None:
    """Importing :mod:`neural_data_decoding.models` registers ``'Logistic'``."""
    # Side-effect import.
    import neural_data_decoding.models  # noqa: F401

    assert "Logistic" in list_classifiers()


def test_registry_dispatch_returns_a_multihead_classifier() -> None:
    """``build_classifier('Logistic', cfg)`` returns a :class:`MultiHeadClassifier`."""
    import neural_data_decoding.models  # noqa: F401

    cfg = {"in_features": 8, "num_classes_per_dim": [3]}
    classifier = build_classifier("Logistic", cfg)
    assert isinstance(classifier, MultiHeadClassifier)
