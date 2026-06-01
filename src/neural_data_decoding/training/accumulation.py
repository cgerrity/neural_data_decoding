"""Hardware-aware gradient accumulation utilities (Critical Note #18).

Direct port of MATLAB's ``cgg_getAccumulationSizeForCurrentSystem.m``
plus the accumulation logic from ``cgg_procGradientAggregation.m``.

The motivation: when the per-device VRAM (or CPU RAM) can't fit a full
``mini_batch_size`` forward+backward pass, split the mini-batch into
micro-batches that DO fit, run forward+backward on each, accumulate
gradients in ``.grad`` (PyTorch's default behavior between
``zero_grad()`` calls), then step the optimizer once. The math is
equivalent to a single full-batch pass (up to floating-point); peak
memory is roughly ``micro_size / mini_size`` of the full-batch peak.

User-visible invariant: same number of optimizer steps as
non-accumulation, gradient direction is equivalent, much lower peak
memory. Identical when ``micro_size >= mini_size``.

A config-driven table (``cfg.accumulation_information``) gives the
max micro-batch size per device:

.. code-block:: yaml

    accumulation_information:
      CPU: 100
      "NVIDIA TITAN X (Pascal)": 20
      "NVIDIA RTX A6000": 20

At resolve-time, the helper detects the current device(s) via
``torch.cuda.get_device_name(i)`` (or falls back to ``"CPU"``) and
returns the matching entry — taking ``min`` if multiple GPUs are
available.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Optional

import torch


def get_accumulation_size_for_current_system(
    accumulation_information: Mapping[str, int],
) -> Optional[int]:
    """Resolve the per-device max micro-batch size from the config table.

    Mirrors ``cgg_getAccumulationSizeForCurrentSystem.m``. Looks up the
    detected device names (or ``"CPU"`` if no CUDA is available) in
    ``accumulation_information`` and returns the minimum entry across
    detected devices.

    Parameters
    ----------
    accumulation_information
        Mapping from system name to max micro-batch size. The names
        should match ``torch.cuda.get_device_name(i)`` output for GPUs
        or be literally ``"CPU"`` for CPU-only runs.

    Returns
    -------
    int or None
        Max micro-batch size for the current system. ``None`` when
        ``accumulation_information`` is empty OR none of the detected
        devices have a matching entry — in either case the caller
        should treat as "no accumulation" (single-pass mode).
    """
    if not accumulation_information:
        return None
    if torch.cuda.is_available():
        device_names = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    else:
        device_names = ["CPU"]
    sizes = [
        int(accumulation_information[name])
        for name in device_names
        if name in accumulation_information
    ]
    if not sizes:
        return None
    return min(sizes)


def micro_batch_chunks(
    n_total: int, max_size: Optional[int],
) -> Iterator[tuple[int, int, float]]:
    """Yield ``(start, end, weight)`` triples partitioning ``[0, n_total)``.

    Each chunk has length ``<= max_size``. The ``weight`` is the chunk's
    fraction of the full batch — multiply per-micro-batch losses by this
    weight before ``.backward()`` so the accumulated gradient sums to
    the equivalent full-batch gradient.

    When ``max_size`` is ``None`` or ``>= n_total``, yields a single
    chunk covering the whole batch with weight 1.0 (no-accumulation
    fast path; identical to the no-micro-batching code path).

    Parameters
    ----------
    n_total
        Size of the full mini-batch (number of trials).
    max_size
        Maximum micro-batch size, or ``None`` for no accumulation.

    Yields
    ------
    (start, end, weight)
        Half-open index range into the batch tensors and the fractional
        weight for loss scaling.

    Examples
    --------
    No-accumulation (single chunk):

    >>> list(micro_batch_chunks(8, None))
    [(0, 8, 1.0)]

    Even split:

    >>> list(micro_batch_chunks(8, 4))
    [(0, 4, 0.5), (4, 8, 0.5)]

    Uneven last chunk:

    >>> list(micro_batch_chunks(7, 3))
    [(0, 3, 0.42857142857142855), (3, 6, 0.42857142857142855), (6, 7, 0.14285714285714285)]
    """
    if n_total <= 0:
        return
    if max_size is None or max_size >= n_total:
        yield 0, n_total, 1.0
        return
    for start in range(0, n_total, max_size):
        end = min(start + max_size, n_total)
        yield start, end, (end - start) / n_total


__all__ = [
    "get_accumulation_size_for_current_system",
    "micro_batch_chunks",
]
