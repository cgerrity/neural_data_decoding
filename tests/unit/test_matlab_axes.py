"""Tests for :mod:`neural_data_decoding.utils.matlab_axes`."""

from __future__ import annotations

import numpy as np
import pytest

from neural_data_decoding.utils.matlab_axes import (
    parse_matlab_format,
    permute_to_matlab,
    permute_to_pytorch,
)


# ───────────────────────── parse_matlab_format ─────────────────────────


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("SSCTB", ["S", "S", "C", "T", "B"]),
        ("CBT", ["C", "B", "T"]),
        ("BCT", ["B", "C", "T"]),
        ("CBTSS", ["C", "B", "T", "S", "S"]),
        ("sscbt", ["S", "S", "C", "B", "T"]),  # case-insensitive
    ],
)
def test_parse_matlab_format_valid(fmt: str, expected: list[str]) -> None:
    """Valid format strings parse into expected per-axis tags."""
    assert parse_matlab_format(fmt) == expected


@pytest.mark.parametrize("fmt", ["", "SSXCT", "ABC", "123"])
def test_parse_matlab_format_invalid(fmt: str) -> None:
    """Invalid format strings raise ``ValueError`` with a helpful message."""
    with pytest.raises(ValueError):
        parse_matlab_format(fmt)


# ───────────────────────── permute_to_pytorch ─────────────────────────


def test_permute_to_pytorch_simple_reorder() -> None:
    """A simple 3-D reorder swaps axes per the format strings."""
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # 'CBT' → channels=2, batch=3, time=4
    y = permute_to_pytorch(x, source_format="CBT", target_format="BCT")
    assert y.shape == (3, 2, 4)
    # spot-check a single element
    assert y[1, 0, 2] == x[0, 1, 2]


def test_permute_to_pytorch_handles_repeated_tags() -> None:
    """Repeated tags (the two ``S``\\ s in ``SSCTB``) preserve their relative order."""
    x = np.zeros((4, 8, 3, 16, 32))  # SSCTB
    y = permute_to_pytorch(x, source_format="SSCTB", target_format="BCTSS")
    assert y.shape == (32, 3, 16, 4, 8)


def test_permute_to_pytorch_is_invertible() -> None:
    """Applying then inverting the permutation returns the original tensor."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8, 3, 16, 32))
    y = permute_to_pytorch(x, source_format="SSCTB", target_format="BCTSS")
    z = permute_to_matlab(y, source_format="BCTSS", target_format="SSCTB")
    np.testing.assert_array_equal(x, z)


def test_permute_mismatched_formats_raise() -> None:
    """Source and target with different tag multisets raise ``ValueError``."""
    x = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="same set of axis tags"):
        permute_to_pytorch(x, source_format="CBT", target_format="BCS")


def test_permute_rejects_unsupported_type() -> None:
    """Calling on a non-array type raises ``TypeError``."""
    with pytest.raises(TypeError):
        permute_to_pytorch([1, 2, 3], source_format="CBT", target_format="BCT")  # type: ignore[arg-type]


# ───────────────────────── torch path ─────────────────────────


def test_permute_handles_torch_tensor() -> None:
    """Permutation works on a :class:`torch.Tensor` and returns a torch view."""
    torch = pytest.importorskip("torch")
    x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    y = permute_to_pytorch(x, source_format="CBT", target_format="BCT")
    assert y.shape == (3, 2, 4)
    assert isinstance(y, torch.Tensor)
    # Verify element correspondence
    assert int(y[1, 0, 2]) == int(x[0, 1, 2])
