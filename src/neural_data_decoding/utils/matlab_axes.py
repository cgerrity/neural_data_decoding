"""Axis-order converters between MATLAB ``dlarray`` and PyTorch tensors.

MATLAB's Deep Learning Toolbox tags each tensor axis with a single letter
(``'S'`` = spatial, ``'C'`` = channel, ``'T'`` = time, ``'B'`` = batch,
``'U'`` = unspecified). The pipeline uses formats like ``'SSCTB'`` (2-D
spatial, channel, time, batch) and ``'CBT'`` (channel, batch, time).

PyTorch has its own conventions: 2-D convolutions expect ``(N, C, H, W)``,
RNNs with ``batch_first=True`` expect ``(N, T, C)``, etc.

This module provides:

* :func:`parse_matlab_format` — turn ``'SSCTB'`` into a list of single-letter
  tags so callers can locate any axis by name.
* :func:`permute_to_pytorch` — permute a tensor from a MATLAB tag order to a
  canonical PyTorch order.
* :func:`permute_to_matlab` — the inverse, for writing tensors back to
  ``.mat`` files.

Conversion is **shape-only**; no copy is made when possible (uses
:meth:`torch.Tensor.permute` or :func:`numpy.transpose`).

Examples
--------
>>> import numpy as np
>>> x = np.zeros((4, 8, 3, 16, 32))            # MATLAB 'SSCTB' layout
>>> y = permute_to_pytorch(x, source_format="SSCTB", target_format="BCTSS")
>>> y.shape
(32, 3, 16, 4, 8)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar, Union, overload

import numpy as np

try:  # PyTorch is a hard runtime dep, but typing-only imports keep this module light.
    import torch

    TensorLike = Union[np.ndarray, "torch.Tensor"]
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TensorLike = np.ndarray  # type: ignore[misc]


T = TypeVar("T", bound=TensorLike)

_VALID_TAGS = frozenset({"S", "C", "T", "B", "U"})


# ───────────────────────── Format parsing ─────────────────────────


def parse_matlab_format(fmt: str) -> list[str]:
    """Split a MATLAB ``dlarray`` format string into per-axis tags.

    Parameters
    ----------
    fmt
        A MATLAB format string such as ``'SSCTB'`` or ``'CBT'``. Each character
        names what the corresponding tensor dimension represents.

    Returns
    -------
    list of str
        One single-letter tag per dimension.

    Raises
    ------
    ValueError
        If ``fmt`` contains a character outside ``{S, C, T, B, U}`` or is empty.

    Examples
    --------
    >>> parse_matlab_format("SSCTB")
    ['S', 'S', 'C', 'T', 'B']
    """
    if not fmt:
        raise ValueError("Format string must be non-empty.")

    fmt = fmt.upper()
    bad = set(fmt) - _VALID_TAGS
    if bad:
        raise ValueError(
            f"Invalid MATLAB format string '{fmt}': unknown tags {sorted(bad)}. "
            f"Valid tags are {sorted(_VALID_TAGS)}."
        )

    return list(fmt)


# ───────────────────────── Permutation ─────────────────────────


def _resolve_permutation(source: str, target: str) -> list[int]:
    """Compute the permutation that maps a source format to a target format.

    Handles repeated tags (e.g. two ``S`` axes in ``'SSCTB'``) by matching them
    in source order to the target's repeated tags in target order.

    Parameters
    ----------
    source
        Source MATLAB format string, e.g. ``'SSCTB'``.
    target
        Target MATLAB format string, e.g. ``'BCTSS'``.

    Returns
    -------
    list of int
        Permutation index list ``perm`` such that
        ``tensor.permute(*perm)`` reorders ``source`` axes into ``target`` order.

    Raises
    ------
    ValueError
        If the source and target tag multisets do not match.
    """
    src_tags = parse_matlab_format(source)
    tgt_tags = parse_matlab_format(target)

    if sorted(src_tags) != sorted(tgt_tags):
        raise ValueError(
            f"Source format '{source}' and target format '{target}' must contain "
            f"the same set of axis tags."
        )

    # Walk the target left-to-right; for each tag, consume the next matching
    # source axis in order. This preserves the relative order of repeated tags.
    remaining: dict[str, list[int]] = {}
    for idx, tag in enumerate(src_tags):
        remaining.setdefault(tag, []).append(idx)

    perm: list[int] = []
    for tag in tgt_tags:
        perm.append(remaining[tag].pop(0))
    return perm


@overload
def permute_to_pytorch(
    tensor: np.ndarray, *, source_format: str, target_format: str
) -> np.ndarray: ...
@overload
def permute_to_pytorch(
    tensor: "torch.Tensor", *, source_format: str, target_format: str
) -> "torch.Tensor": ...


def permute_to_pytorch(
    tensor: TensorLike, *, source_format: str, target_format: str
) -> TensorLike:
    """Permute a tensor from MATLAB axis order into a PyTorch-friendly order.

    Parameters
    ----------
    tensor
        A NumPy array or PyTorch tensor whose axes are currently ordered
        according to ``source_format``.
    source_format
        The MATLAB ``dlarray`` format string the tensor is currently in.
    target_format
        The MATLAB ``dlarray`` format string the tensor should be reordered
        into. Same tag multiset as ``source_format``, just permuted.

    Returns
    -------
    Same type as ``tensor``
        A view (or copy if necessary) of ``tensor`` with axes reordered.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((4, 8, 3, 16, 32))            # 'SSCTB'
    >>> y = permute_to_pytorch(x, source_format="SSCTB", target_format="BCTSS")
    >>> y.shape
    (32, 3, 16, 4, 8)
    """
    perm = _resolve_permutation(source_format, target_format)
    return _apply_perm(tensor, perm)


def permute_to_matlab(
    tensor: TensorLike, *, source_format: str, target_format: str
) -> TensorLike:
    """Permute a tensor back into a MATLAB axis order (for ``.mat`` writes).

    This is symmetrical with :func:`permute_to_pytorch` — the only reason for
    a second name is so call sites read directionally ("converting back to
    MATLAB" reads better than reusing the forward function).

    Parameters
    ----------
    tensor
        A NumPy array or PyTorch tensor whose axes are currently ordered
        according to ``source_format``.
    source_format
        The current MATLAB format string of ``tensor``.
    target_format
        The desired MATLAB format string after the permutation.

    Returns
    -------
    Same type as ``tensor``
        A view (or copy if necessary) of ``tensor`` with axes reordered.
    """
    return permute_to_pytorch(
        tensor, source_format=source_format, target_format=target_format
    )


def _apply_perm(tensor: TensorLike, perm: Iterable[int]) -> TensorLike:
    """Apply a permutation index list to a numpy array or torch tensor."""
    perm_tuple = tuple(perm)
    if isinstance(tensor, np.ndarray):
        return np.transpose(tensor, perm_tuple)
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.permute(*perm_tuple)
    raise TypeError(
        f"permute_to_* requires numpy.ndarray or torch.Tensor; got {type(tensor)!r}."
    )


__all__ = [
    "parse_matlab_format",
    "permute_to_pytorch",
    "permute_to_matlab",
]
