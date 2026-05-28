"""Bidirectional weight converter — MATLAB ``dlnetwork`` ↔ PyTorch modules.

This module enables Milestone B's T2 forward-pass parity test: load a
MATLAB-trained encoder's weights into a matching Python
:class:`~neural_data_decoding.models.encoder.SimpleSequenceEncoder`, run
the same input through both, and assert the outputs match within fp32
tolerance.

For Milestone B we only need the **MATLAB → PyTorch** direction and only
the **Simple GRU** layer type. Later milestones will extend this to:

* Reverse direction (PyTorch → MATLAB) for handing trained networks to
  the MATLAB analysis pipeline.
* LSTM, Feedforward, BatchNorm, and the VAE sampling layer.

GRU parameter mapping
---------------------
MATLAB's ``gruLayer`` exposes three learnable parameters per layer:

* ``InputWeights`` — shape ``(3*H, I)``, gate order ``[reset, update, candidate]``
* ``RecurrentWeights`` — shape ``(3*H, H)``, same gate order
* ``Bias`` — shape ``(3*H, 1)``, same gate order

PyTorch's :class:`nn.GRU` (with ``batch_first=True``) has four:

* ``weight_ih_l0`` — shape ``(3*H, I)``, gate order ``[r, u, n]``
* ``weight_hh_l0`` — shape ``(3*H, H)``, gate order ``[r, u, n]``
* ``bias_ih_l0`` — shape ``(3*H,)``, gate order ``[r, u, n]``
* ``bias_hh_l0`` — shape ``(3*H,)``, gate order ``[r, u, n]``

Gate orderings are **identical** (both MATLAB and PyTorch use
``[reset, update, candidate]``), so the weight matrices copy across
verbatim. The bias mapping requires care:

* MATLAB has a single ``Bias`` per gate.
* PyTorch has separate ``bias_ih`` and ``bias_hh``; the effective bias
  for each gate is their sum, EXCEPT for the candidate gate where
  PyTorch applies ``r * (W_hn h + b_hn)`` (i.e., ``bias_hh`` for the
  candidate gate is multiplied by the reset gate).

To get an exact match, set ``bias_ih = MATLAB Bias`` and
``bias_hh = 0``. With ``bias_hh = 0``:

* Reset gate: ``sigmoid(W_xr x + 0 + W_hr h + b_r)`` ≡ MATLAB ✓
* Update gate: ``sigmoid(W_xu x + 0 + W_hu h + b_u)`` ≡ MATLAB ✓
* Candidate: ``tanh(W_xn x + b_n + r * (W_hn h + 0))`` ≡ MATLAB ✓

This holds only when MATLAB's ``gruLayer.ResetGateMode ==
'after-multiplication'`` (the default — checked by the fixture
generator). For ``'before-multiplication'`` mode the formula differs
and the converter would need to handle the candidate gate specially.

ResetGateMode parity is verified at fixture-generation time
(``scripts/generate_t2_encoder_fixture.m``) — if MATLAB's default ever
changes, that script's assertion catches it.

Examples
--------
>>> import torch
>>> from neural_data_decoding.models.encoder import SimpleSequenceEncoder
>>> from neural_data_decoding.interop.weight_converter import (
...     load_matlab_gru_encoder_weights,
... )
>>> # Suppose `fixture` is the loaded .mat dict produced by
>>> # generate_t2_encoder_fixture.m.
>>> # encoder = SimpleSequenceEncoder(in_features=3, hidden_sizes=[4, 2],
>>> #                                 transform="GRU")
>>> # load_matlab_gru_encoder_weights(fixture, encoder)  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from neural_data_decoding.models.encoder import SimpleSequenceEncoder


_MATLAB_GRU_LAYER_PREFIX = "gru_Encoder_"  # cgg_generateSimpleBlock convention


def load_matlab_gru_encoder_weights(
    fixture: Mapping[str, Any],
    encoder: SimpleSequenceEncoder,
) -> None:
    """In-place transplant MATLAB gruLayer weights into a Python encoder.

    Walks the encoder's :class:`nn.GRU` blocks in order and copies in the
    matching MATLAB ``InputWeights``, ``RecurrentWeights``, and ``Bias``
    fields from the fixture dict. Each MATLAB ``Bias`` lands on
    ``bias_ih_l0``; ``bias_hh_l0`` is zeroed (see module docstring for
    why this gives bit-exact parity).

    Parameters
    ----------
    fixture
        Mapping produced by ``scipy.io.loadmat`` on the
        ``encoder_t2_gru_simple.mat`` file. Expected to contain a
        ``weights`` sub-struct (loaded as either a ``mat_struct`` or a
        plain dict) with fields ``gru_Encoder_{k}__InputWeights``,
        ``..__RecurrentWeights``, ``..__Bias`` for each layer ``k``.
    encoder
        Target :class:`SimpleSequenceEncoder` to receive the weights.
        Its ``hidden_sizes`` and ``in_features`` must match the
        fixture's. The module is modified **in place**.

    Raises
    ------
    KeyError
        If the fixture is missing a weight tensor for any GRU layer.
    ValueError
        If the encoder's structure (number of blocks, hidden sizes,
        ``transform``) doesn't match the fixture's, or if any GRU's
        learnable shape doesn't match the MATLAB weight shape.
    """
    if encoder.transform != "GRU":
        raise ValueError(
            f"Converter only supports GRU encoders; got transform={encoder.transform!r}."
        )

    weights = _coerce_weights_dict(fixture)

    for layer_idx, block in enumerate(encoder.blocks, start=1):
        gru = block.transform_layer
        if not isinstance(gru, torch.nn.GRU):
            raise ValueError(
                f"Expected nn.GRU at block index {layer_idx - 1}; "
                f"got {type(gru).__name__}."
            )

        prefix = f"{_MATLAB_GRU_LAYER_PREFIX}{layer_idx}__"
        try:
            w_input = np.asarray(weights[prefix + "InputWeights"], dtype=np.float32)
            w_recur = np.asarray(weights[prefix + "RecurrentWeights"], dtype=np.float32)
            bias = np.asarray(weights[prefix + "Bias"], dtype=np.float32).ravel()
        except KeyError as exc:
            raise KeyError(
                f"Fixture is missing weight tensor {exc} for encoder block "
                f"{layer_idx}."
            ) from exc

        expected_3h = 3 * gru.hidden_size
        if w_input.shape != (expected_3h, gru.input_size):
            raise ValueError(
                f"Block {layer_idx} InputWeights shape {w_input.shape} does not "
                f"match nn.GRU's weight_ih_l0 expected shape "
                f"({expected_3h}, {gru.input_size})."
            )
        if w_recur.shape != (expected_3h, gru.hidden_size):
            raise ValueError(
                f"Block {layer_idx} RecurrentWeights shape {w_recur.shape} does "
                f"not match nn.GRU's weight_hh_l0 expected shape "
                f"({expected_3h}, {gru.hidden_size})."
            )
        if bias.shape != (expected_3h,):
            raise ValueError(
                f"Block {layer_idx} Bias shape {bias.shape} does not match "
                f"expected ({expected_3h},)."
            )

        with torch.no_grad():
            gru.weight_ih_l0.copy_(torch.from_numpy(w_input))
            gru.weight_hh_l0.copy_(torch.from_numpy(w_recur))
            gru.bias_ih_l0.copy_(torch.from_numpy(bias))
            gru.bias_hh_l0.zero_()  # critical — see module docstring


def matlab_ctb_to_pytorch_btc(x: np.ndarray) -> np.ndarray:
    """Convert a MATLAB ``'CTB'``-laid-out tensor to PyTorch's ``(B, T, C)``.

    MATLAB's ``sequenceInputLayer`` produces dlarrays formatted as
    ``'CTB'`` (channel-time-batch). PyTorch's ``nn.GRU`` with
    ``batch_first=True`` expects ``(batch, time, channel)``. Permutation:
    ``[0=C, 1=T, 2=B] → [2, 1, 0]``.

    Parameters
    ----------
    x
        Array with shape ``(num_channels, num_timesteps, num_trials)``.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(num_trials, num_timesteps, num_channels)``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D CTB array; got shape {x.shape}.")
    return np.transpose(x, (2, 1, 0))


def matlab_cbt_to_pytorch_btc(x: np.ndarray) -> np.ndarray:
    """Convert a MATLAB ``'CBT'``-laid-out tensor to PyTorch's ``(B, T, C)``.

    The output of a stacked ``gruLayer`` is reported in ``'CBT'`` order
    (channel-batch-time) — different from the input's ``'CTB'``. The
    permutation for output comparison is therefore
    ``[0=C, 1=B, 2=T] → [1, 2, 0]``.

    Parameters
    ----------
    x
        Array with shape ``(num_channels, num_trials, num_timesteps)``.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(num_trials, num_timesteps, num_channels)``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D CBT array; got shape {x.shape}.")
    return np.transpose(x, (1, 2, 0))


# ───────────────────────── Internal helpers ─────────────────────────


def _coerce_weights_dict(fixture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the fixture's ``weights`` field as a plain dict of arrays.

    ``scipy.io.loadmat`` returns nested structs as either ``mat_struct``
    objects (with ``_fieldnames``) or as ``np.void`` records, depending
    on the ``struct_as_record`` setting. This helper accepts either and
    yields a uniform ``dict[str, ndarray]`` so the caller doesn't have
    to handle both representations.
    """
    if "weights" not in fixture:
        raise KeyError(
            "Fixture does not contain a 'weights' key. Did the MATLAB "
            "script save the right struct?"
        )
    raw = fixture["weights"]

    # Common scipy.io.loadmat path with struct_as_record=False, squeeze_me=False:
    # raw is a 1x1 object array wrapping a mat_struct.
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.size != 1:
            raise ValueError(
                f"Unexpected weights structure: object array shape {raw.shape}."
            )
        raw = raw.flat[0]

    # mat_struct case.
    if hasattr(raw, "_fieldnames"):
        return {name: getattr(raw, name) for name in raw._fieldnames}

    # Dict case (e.g. mat73 loader or pre-flattened).
    if isinstance(raw, Mapping):
        return dict(raw)

    raise TypeError(
        f"Could not interpret fixture['weights'] of type {type(raw).__name__}."
    )


__all__ = [
    "load_matlab_gru_encoder_weights",
    "matlab_cbt_to_pytorch_btc",
    "matlab_ctb_to_pytorch_btc",
]
