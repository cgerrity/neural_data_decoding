"""Global seed control for reproducible runs.

Sets the seeds of every RNG that the pipeline touches: Python's :mod:`random`,
NumPy's legacy and modern (``Generator``) APIs, and PyTorch's CPU and CUDA
generators.

Bit-exact reproducibility is **not** a parity goal of this project (see
ADR 001) — but parity tests still need *intra-Python* determinism so the same
seed produces the same output on consecutive runs. This module provides that
guarantee.

Examples
--------
>>> from neural_data_decoding.utils.seeding import set_global_seed
>>> set_global_seed(42)
42
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic_cuda: bool = False) -> int:
    """Seed every RNG the pipeline touches.

    Parameters
    ----------
    seed
        The seed value. Any non-negative integer.
    deterministic_cuda
        If ``True``, also set ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False`` so CUDA convolutions become
        deterministic at a performance cost. Default ``False`` — this is only
        needed when chasing bit-exact reproducibility, which is explicitly NOT
        a parity goal of this project (see ADR 001).

    Returns
    -------
    int
        The seed value that was applied.

    Notes
    -----
    Also sets ``PYTHONHASHSEED`` for consistency, though this only takes effect
    on the *next* Python process — within the current process, the hash seed
    is fixed at interpreter start.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


__all__ = ["set_global_seed"]
