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
_MATLAB_LSTM_LAYER_PREFIX = "lstm_Encoder_"


# LSTM gate-order note
# --------------------
# Both MATLAB ``lstmLayer`` and PyTorch ``nn.LSTM`` use gate order
# ``[input, forget, cell-candidate, output]`` (i, f, g, o). Per
# MathWorks docs: "The four matrices are concatenated vertically in the
# following order: input gate, forget gate, cell candidate, output gate."
# Per PyTorch docs: "the input weights w_ih_l[k] = (W_ii|W_if|W_ig|W_io)".
# So InputWeights / RecurrentWeights copy straight across.
#
# LSTM has no reset-gate subtlety analogous to GRU's candidate-gate
# multiplication, so the bias mapping is unambiguous: PyTorch's effective
# per-gate bias is ``bias_ih + bias_hh``. Setting ``bias_ih = MATLAB Bias``
# and ``bias_hh = 0`` gives exact parity for all four gates.


def _field_name(layer: str, param: str) -> str:
    """Build the fixture struct field name for a (layer, param) pair.

    Mirrors the MATLAB-side convention in the fixture-generator scripts:
    ``"<layer>__<param>"`` with ``-`` replaced by ``_`` (the only
    field-name-hostile character in our layer names, e.g.
    ``LSTM_Dim_1_Layer-Out`` → ``LSTM_Dim_1_Layer_Out``).
    """
    return f"{layer}__{param}".replace("-", "_")


def _assign_rnn_layer(
    rnn: torch.nn.Module,
    weights: Mapping[str, np.ndarray],
    matlab_layer: str,
    *,
    gates: int,
) -> None:
    """Copy MATLAB RNN learnables into an ``nn.GRU`` / ``nn.LSTM`` layer.

    Parameters
    ----------
    rnn
        Target ``nn.GRU`` (``gates=3``) or ``nn.LSTM`` (``gates=4``).
    weights
        Flat ``{field: ndarray}`` dict from the fixture.
    matlab_layer
        MATLAB layer name (e.g. ``"gru_Encoder_1"``,
        ``"LSTM_Dim_1_Layer-Out"``).
    gates
        3 for GRU, 4 for LSTM. Used for shape validation.

    Raises
    ------
    KeyError
        If any of the three learnables is missing.
    ValueError
        If a learnable shape doesn't match the module's expectation.
    """
    try:
        w_input = np.asarray(
            weights[_field_name(matlab_layer, "InputWeights")], dtype=np.float32
        )
        w_recur = np.asarray(
            weights[_field_name(matlab_layer, "RecurrentWeights")], dtype=np.float32
        )
        bias = np.asarray(
            weights[_field_name(matlab_layer, "Bias")], dtype=np.float32
        ).ravel()
    except KeyError as exc:
        raise KeyError(
            f"Fixture is missing weight tensor {exc} for layer {matlab_layer!r}."
        ) from exc

    expected = gates * rnn.hidden_size
    if w_input.shape != (expected, rnn.input_size):
        raise ValueError(
            f"{matlab_layer} InputWeights shape {w_input.shape} != expected "
            f"({expected}, {rnn.input_size})."
        )
    if w_recur.shape != (expected, rnn.hidden_size):
        raise ValueError(
            f"{matlab_layer} RecurrentWeights shape {w_recur.shape} != expected "
            f"({expected}, {rnn.hidden_size})."
        )
    if bias.shape != (expected,):
        raise ValueError(
            f"{matlab_layer} Bias shape {bias.shape} != expected ({expected},)."
        )

    with torch.no_grad():
        rnn.weight_ih_l0.copy_(torch.from_numpy(w_input))
        rnn.weight_hh_l0.copy_(torch.from_numpy(w_recur))
        rnn.bias_ih_l0.copy_(torch.from_numpy(bias))
        rnn.bias_hh_l0.zero_()  # critical — see module docstring


def _assign_linear_layer(
    linear: torch.nn.Linear,
    weights: Mapping[str, np.ndarray],
    matlab_layer: str,
) -> None:
    """Copy a MATLAB ``fullyConnectedLayer``'s learnables into an ``nn.Linear``.

    MATLAB ``fullyConnectedLayer`` exposes ``Weights`` of shape
    ``(OutputSize, InputSize)`` and ``Bias`` of shape ``(OutputSize, 1)`` —
    the same orientation as PyTorch's ``nn.Linear.weight`` ``(out, in)``
    and ``nn.Linear.bias`` ``(out,)`` — so both copy across verbatim.

    Raises
    ------
    KeyError
        If ``Weights`` or ``Bias`` is missing.
    ValueError
        If a learnable shape doesn't match the module's expectation.
    """
    try:
        weight = np.asarray(
            weights[_field_name(matlab_layer, "Weights")], dtype=np.float32
        )
        bias = np.asarray(
            weights[_field_name(matlab_layer, "Bias")], dtype=np.float32
        ).ravel()
    except KeyError as exc:
        raise KeyError(
            f"Fixture is missing FC learnable {exc} for layer {matlab_layer!r}."
        ) from exc

    if weight.shape != (linear.out_features, linear.in_features):
        raise ValueError(
            f"{matlab_layer} Weights shape {weight.shape} != expected "
            f"({linear.out_features}, {linear.in_features})."
        )
    if bias.shape != (linear.out_features,):
        raise ValueError(
            f"{matlab_layer} Bias shape {bias.shape} != expected "
            f"({linear.out_features},)."
        )

    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(weight))
        linear.bias.copy_(torch.from_numpy(bias))


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
        _assign_rnn_layer(
            gru, weights, f"{_MATLAB_GRU_LAYER_PREFIX}{layer_idx}", gates=3
        )


def load_matlab_lstm_encoder_weights(
    fixture: Mapping[str, Any],
    encoder: SimpleSequenceEncoder,
) -> None:
    """In-place transplant MATLAB lstmLayer weights into a Python encoder.

    Sister function of :func:`load_matlab_gru_encoder_weights` for LSTM
    stacks. Same gate ordering on both sides; ``bias_ih = MATLAB Bias``
    and ``bias_hh = 0`` produces exact per-gate parity. See the module
    docstring for the algebra.

    Parameters
    ----------
    fixture
        Mapping produced by ``scipy.io.loadmat`` on the LSTM fixture
        ``.mat`` file. Expected to contain a ``weights`` sub-struct with
        fields ``lstm_Encoder_{k}__InputWeights``, ``..__RecurrentWeights``,
        ``..__Bias``.
    encoder
        Target :class:`SimpleSequenceEncoder` with ``transform='LSTM'``.
        Modified in place.

    Raises
    ------
    KeyError
        If a weight tensor is missing for any LSTM layer.
    ValueError
        If the encoder's transform isn't ``'LSTM'`` or any LSTM
        learnable shape doesn't match.
    """
    if encoder.transform != "LSTM":
        raise ValueError(
            f"Converter only supports LSTM encoders here; got transform="
            f"{encoder.transform!r}."
        )

    weights = _coerce_weights_dict(fixture)
    for layer_idx, block in enumerate(encoder.blocks, start=1):
        lstm = block.transform_layer
        if not isinstance(lstm, torch.nn.LSTM):
            raise ValueError(
                f"Expected nn.LSTM at block index {layer_idx - 1}; "
                f"got {type(lstm).__name__}."
            )
        _assign_rnn_layer(
            lstm, weights, f"{_MATLAB_LSTM_LAYER_PREFIX}{layer_idx}", gates=4
        )


def load_matlab_composite_weights(
    fixture: Mapping[str, Any],
    composite: Any,
) -> None:
    """Transplant a full Encoder + Bottleneck + Deep-LSTM-Classifier composite.

    Walks an
    :class:`~neural_data_decoding.models.composite.EncoderClassifierComposite`
    and copies every MATLAB learnable into place:

    * **Encoder** — ``gru_Encoder_{k}`` (or ``lstm_Encoder_{k}``) into each
      ``encoder.blocks[k-1].transform_layer`` (GRU ``gates=3`` / LSTM
      ``gates=4``).
    * **Bottleneck** — ``fc_OUT_BottleNeck`` into ``bottleneck.linear``
      when the bottleneck is a
      :class:`~neural_data_decoding.models.bottleneck.LinearBottleneck`.
      A :class:`~neural_data_decoding.models.bottleneck.PassthroughBottleneck`
      has no learnables and is skipped.
    * **Classifier** — for each output dimension ``d`` (1-indexed), the
      per-dim LSTM stack ``LSTM_Dim_{d}_Layer-1`` …
      ``LSTM_Dim_{d}_Layer-Out`` into
      ``classifier.stacks[d-1].lstms[*]`` and the head ``fc_Dim_{d}`` into
      ``classifier.stacks[d-1].head``.

    The MATLAB classifier names the inner LSTM layers ``Layer-1`` …
    ``Layer-{Depth-1}`` and the final one ``Layer-Out``; this function
    reproduces that mapping so a stack of any depth lines up.

    Parameters
    ----------
    fixture
        ``scipy.io.loadmat`` output for the composite fixture, containing a
        ``weights`` sub-struct.
    composite
        Target ``EncoderClassifierComposite``. Modified in place. Typed as
        ``Any`` to avoid importing the composite (and pulling its deps)
        into this interop module.

    Raises
    ------
    KeyError
        If any expected learnable is missing from the fixture.
    ValueError
        If a module's structure doesn't match the fixture's shapes.
    """
    weights = _coerce_weights_dict(fixture)

    # --- Encoder ---
    for layer_idx, block in enumerate(composite.encoder.blocks, start=1):
        rnn = block.transform_layer
        if isinstance(rnn, torch.nn.GRU):
            _assign_rnn_layer(
                rnn, weights, f"{_MATLAB_GRU_LAYER_PREFIX}{layer_idx}", gates=3
            )
        elif isinstance(rnn, torch.nn.LSTM):
            _assign_rnn_layer(
                rnn, weights, f"{_MATLAB_LSTM_LAYER_PREFIX}{layer_idx}", gates=4
            )
        else:
            raise ValueError(
                f"Composite encoder block {layer_idx - 1} has unsupported "
                f"transform {type(rnn).__name__}; expected nn.GRU or nn.LSTM."
            )

    # --- Bottleneck (LinearBottleneck only; Passthrough has no weights) ---
    bottleneck = composite.bottleneck
    if hasattr(bottleneck, "linear") and isinstance(
        bottleneck.linear, torch.nn.Linear
    ):
        _assign_linear_layer(bottleneck.linear, weights, "fc_OUT_BottleNeck")

    # --- Classifier (per-dim LSTM stacks + FC heads) ---
    classifier = composite.classifier
    stacks = getattr(classifier, "stacks", None)
    if stacks is None:
        raise ValueError(
            "load_matlab_composite_weights expects a DeepLSTMClassifier "
            "(with per-dim `stacks`); got a classifier without that attribute."
        )
    for dim_idx, stack in enumerate(stacks, start=1):
        num_lstms = len(stack.lstms)
        for li, lstm in enumerate(stack.lstms):
            if li < num_lstms - 1:
                matlab_layer = f"LSTM_Dim_{dim_idx}_Layer-{li + 1}"
            else:
                matlab_layer = f"LSTM_Dim_{dim_idx}_Layer-Out"
            _assign_rnn_layer(lstm, weights, matlab_layer, gates=4)
        _assign_linear_layer(stack.head, weights, f"fc_Dim_{dim_idx}")


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

    # mat_struct case. After `.flat[0]` the static type is still ndarray, but
    # at runtime the unwrapped element is a scipy.io.mat_struct. Reach for the
    # attribute via getattr to bypass the (over-strict) static check.
    if hasattr(raw, "_fieldnames"):
        fieldnames = getattr(raw, "_fieldnames")
        return {name: getattr(raw, name) for name in fieldnames}

    # Dict case (e.g. mat73 loader or pre-flattened).
    if isinstance(raw, Mapping):
        return dict(raw)

    raise TypeError(
        f"Could not interpret fixture['weights'] of type {type(raw).__name__}."
    )


__all__ = [
    "load_matlab_composite_weights",
    "load_matlab_gru_encoder_weights",
    "load_matlab_lstm_encoder_weights",
    "matlab_cbt_to_pytorch_btc",
    "matlab_ctb_to_pytorch_btc",
]
