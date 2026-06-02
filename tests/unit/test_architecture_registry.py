"""Tests for the architecture-spec registry (CC.1)."""

from __future__ import annotations

import pytest

from neural_data_decoding.models.architecture_registry import (
    ArchitectureSpec,
    has_architecture,
    list_architectures,
    resolve_architecture,
)


# ───────────────────────── Registry membership ─────────────────────────


_SLURM_NAMES = [
    "Feedforward",
    "LSTM",
    "Convolutional",
    "Resnet",
    "Multi-Filter Convolutional",
    "Logistic Regression",
    "PCA",
]


@pytest.mark.parametrize("name", _SLURM_NAMES + ["GRU"])
def test_required_architectures_registered(name: str) -> None:
    """Every SLURM-sweep architecture + production GRU is registered."""
    assert has_architecture(name)
    spec = resolve_architecture(name)
    assert isinstance(spec, ArchitectureSpec)


def test_list_architectures_returns_sorted_unique() -> None:
    """``list_architectures`` returns the registered names sorted."""
    names = list_architectures()
    assert names == sorted(set(names))


def test_resolve_unknown_raises_with_listing() -> None:
    """Unknown names raise ValueError that lists known options."""
    with pytest.raises(ValueError, match="Registered"):
        resolve_architecture("Some Unknown Architecture")


def test_has_architecture_negative() -> None:
    assert not has_architecture("Nonexistent Name")


# ───────────────────────── Spec contents ─────────────────────────


def test_logistic_regression_flags() -> None:
    spec = resolve_architecture("Logistic Regression")
    assert spec.is_simple and not spec.is_variational
    assert spec.transform == "Feedforward"
    assert spec.output_fully_connected is False
    assert spec.dropout == 0.0


def test_gru_flags() -> None:
    spec = resolve_architecture("GRU")
    assert spec.is_simple and not spec.is_variational
    assert spec.transform == "GRU"
    assert spec.output_fully_connected is True
    assert spec.filter_sizes is None  # Simple branch — no conv fields.


def test_lstm_flags() -> None:
    spec = resolve_architecture("LSTM")
    assert spec.is_simple
    assert spec.transform == "LSTM"


def test_feedforward_flags() -> None:
    spec = resolve_architecture("Feedforward")
    assert spec.is_simple
    assert spec.transform == "Feedforward"
    assert spec.activation == "ReLU"  # 'Feedforward' uses ReLU per MATLAB.


def test_convolutional_flags() -> None:
    spec = resolve_architecture("Convolutional")
    assert not spec.is_simple
    assert spec.filter_sizes == [[4, 20]]
    assert spec.want_split_areas is False
    assert spec.stride == 2
    assert spec.want_resnet is False
    assert spec.final_activation == "Convolutional"


def test_resnet_differs_from_convolutional_only_in_want_resnet() -> None:
    """The MATLAB 'Resnet' case is 'Convolutional' with WantResnet=true."""
    conv = resolve_architecture("Convolutional")
    resnet = resolve_architecture("Resnet")
    assert resnet.want_resnet is True
    assert conv.want_resnet is False
    # All other fields equal.
    assert resnet.is_simple == conv.is_simple
    assert resnet.filter_sizes == conv.filter_sizes
    assert resnet.filter_size_percent == conv.filter_size_percent
    assert resnet.stride == conv.stride
    assert resnet.down_sample_method == conv.down_sample_method
    assert resnet.up_sample_method == conv.up_sample_method
    assert resnet.activation == conv.activation
    assert resnet.final_activation == conv.final_activation


def test_multi_filter_convolutional_has_multiple_kernel_sizes() -> None:
    spec = resolve_architecture("Multi-Filter Convolutional")
    assert spec.filter_sizes == [3, 5, 7]
    assert spec.filter_size_percent == [0.2, 0.3, 0.4]
    assert spec.want_resnet is False
    assert spec.activation == "Leaky ReLU"


def test_pca_spec_present_for_cc2_planning() -> None:
    """PCA registered as a placeholder so CLI can detect 'not yet built'."""
    spec = resolve_architecture("PCA")
    assert spec.transform == "PCA"
    assert not spec.is_simple  # MATLAB sets IsSimple=false for PCA.


# ───────────────────────── Spec immutability ─────────────────────────


def test_spec_is_frozen() -> None:
    """ArchitectureSpec is a frozen dataclass — fields cannot be mutated."""
    spec = resolve_architecture("GRU")
    with pytest.raises((AttributeError, TypeError)):
        spec.dropout = 0.5  # type: ignore[misc]
