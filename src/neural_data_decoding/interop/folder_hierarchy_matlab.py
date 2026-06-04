"""MATLAB-parity long-folder result-directory layout.

Mirrors the deep folder hierarchy produced by
``cgg_generateDecodingFolders.m`` (the outer Aggregate/Epoched/Encoding
chain) and ``cgg_generateEncoderSubFolders_v3.m`` (the 13-level
encoder/classifier subtree). Output written under the returned paths
is discoverable by the MATLAB results aggregator
``DATA_cggAllNetworkEncoderResults.m`` without modification — Critical
Note #15 in the migration plan.

Hierarchy
---------
::

    <base_dir>/
      Aggregate Data/
        Epoched Data/
          {Epoch}/
            Encoding/
              {Target}/
                {ModelName}/
                  {ModelParameters}/        # 'Variational - Stochastic Encoder ~ Dropout - 5.00e-01 ~ ...'
                    {WidthStride}/          # 'Data Width - 100 ~ Window Stride - 50 ~ Time Percent - ...'
                      {Normalization}/      # 'Normalization - Channel - Z-Score ...'
                        {HiddenSize}/       # 'Hidden Size - 1000-500-250'
                          {Learning}/       # 'Initial Learning Rate - 1.00e-03 ~ Gradient Threshold - 1.00e+02 - Global ~ ...'
                            {MiniBatchSize}/    # 'Mini Batch Size - 100 ~ Max Accumulation - 100 ~ Hierarchically Stratified'
                              {DataAugmentation}/   # 'Channel Offset - 3.00e-02 ~ ...'
                                {IsSubset}/         # 'Subset' | 'All Sessions' | '<SessionName>'
                                  {AutoEncoder}/    # 'AutoEncoder - Epochs - 0 ~ Loss Function - MSE ~ ...'
                                    {Loss}/         # 'Weight Reconstruction - 1.00e+02 ~ ...'
                                      {Dynamic}/    # 'Dynamic Set - ... ~ S and F - ...'
                                        Information/          <- 'AutoEncoderInformation'
                                          Fold_{N}/           <- 'AutoEncoderFold'
                                        {Classifier}/         # 'Classifier - Deep LSTM - Dropout 0.5 ~ ...'
                                          Fold_{N}/

The leaf classifier ``Fold_{N}`` directory is the main run output
(``CM_Table.mat``, ``CM_Table_Validation.mat``, ``EncodingParameters.yaml``,
checkpoints). The parallel ``Information/Fold_{N}`` directory holds the
autoencoder pre-training state when ``num_epochs_autoencoder > 0``.

Format conventions
------------------
* Floats render with MATLAB ``%.2e`` ("1.00e-04"). Both sides handle
  the sign symmetrically — Python's ``format(x, '.2e')`` produces the
  same string for non-pathological values.
* Integers render with ``%d``.
* ``NaN`` floats render as the literal string ``None`` (matches the
  MATLAB pipeline's ``'... - None'`` sentinel).
* Hidden-size lists render hyphen-joined: ``[1000, 500, 250]`` →
  ``"1000-500-250"``.
* The within-folder separator is ``' ~ '`` (space-tilde-space, exactly
  as MATLAB).

This module is **pure** — no filesystem access. ``mkdir`` happens in
the caller. The :func:`build_matlab_run_dirs` entry point returns a
:class:`MatlabRunDirs` named tuple with both the classifier fold dir
and the autoencoder fold dir so the caller can create whichever it
needs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _fmt_exp(value: float | int | None) -> str:
    """MATLAB ``%.2e`` rendering, with ``None``/NaN → the literal ``"None"``."""
    if value is None:
        return "None"
    f = float(value)
    if math.isnan(f):
        return "None"
    return f"{f:.2e}"


def _fmt_int_or_str(value: int | str) -> str:
    """Pass an int through ``%d``; leave a string (``'All'``) untouched."""
    if isinstance(value, bool):  # bool is an int subclass — guard explicitly
        return str(value)
    if isinstance(value, int):
        return f"{value:d}"
    return str(value)


def _fmt_hidden_size(sizes: Sequence[int] | int) -> str:
    """Render a hidden-size spec as MATLAB does (``"1000-500-250"``)."""
    if isinstance(sizes, int):
        return f"{sizes:d}"
    sizes_list = list(sizes)
    if len(sizes_list) == 0:
        return "None"
    if len(sizes_list) == 1:
        return f"{int(sizes_list[0]):d}"
    head = f"{int(sizes_list[0]):d}"
    tail = "".join(f"-{int(s):d}" for s in sizes_list[1:])
    return head + tail


def _sanitize_for_path(component: str) -> str:
    """Replace filesystem-hostile chars only.

    MATLAB folder names contain spaces, hyphens, and brackets — all
    legal on macOS / Linux / Windows. We strip only path separators
    and the NUL byte.
    """
    forbidden = "/\\\x00"
    return "".join("-" if c in forbidden else c for c in component).strip()


# ---------------------------------------------------------------------------
# Per-level name builders — each is a private pure function
# ---------------------------------------------------------------------------


def _name_model_parameters(
    is_variational: bool,
    encoder_output_type: str,
    activation: str,
    dropout: float,
    want_normalization: Any,
    bottle_neck_depth: int,
) -> str:
    """Port lines 211-254 — ``ModelParameters`` folder name.

    ``"Variational - Stochastic Encoder ~ Dropout - 5.00e-01 ~ Bottle Neck Depth - 1"``
    is the canonical Optimal-config rendering.
    """
    parts: list[str] = []
    if is_variational:
        variational = "Variational"
        if encoder_output_type == "Stochastic":
            variational += " - Stochastic Encoder"
        parts.append(variational)
    if activation:
        parts.append(f"Activation - {activation}")
    if dropout != 0:
        parts.append(f"Dropout - {_fmt_exp(dropout)}")
    if isinstance(want_normalization, bool):
        if want_normalization:
            parts.append("Normalized")
    else:
        parts.append(f"{want_normalization} Normalized")
    parts.append(f"Bottle Neck Depth - {int(bottle_neck_depth):d}")
    return " ~ ".join(parts)


def _name_width_stride(
    data_width: int | str,
    window_stride: int | str,
    start_end_percent: tuple[float | None, float | None] | None,
) -> str:
    """Port lines 261-281 — ``WidthStride`` folder name."""
    name = (
        f"Data Width - {_fmt_int_or_str(data_width)} ~ "
        f"Window Stride - {_fmt_int_or_str(window_stride)}"
    )
    if start_end_percent is not None and not all(
        v is None or (isinstance(v, float) and math.isnan(v))
        for v in start_end_percent
    ):
        lo = start_end_percent[0]
        hi = start_end_percent[1]
        lo_s = f"{float(lo):.1f}" if lo is not None else "NaN"
        hi_s = f"{float(hi):.1f}" if hi is not None else "NaN"
        name += f" ~ Time Percent - [{lo_s}, {hi_s}]"
    return name


def _name_normalization(normalization: str) -> str:
    """Port line 289 — ``Normalization`` folder name."""
    return f"Normalization - {normalization}"


def _name_hidden_size(hidden_sizes: Sequence[int] | int) -> str:
    """Port lines 296-300 — ``HiddenSize`` folder name."""
    return f"Hidden Size - {_fmt_hidden_size(hidden_sizes)}"


def _name_learning(
    initial_learning_rate: float,
    gradient_threshold: float | None,
    gradient_clip_type: str,
    optimizer: str,
    l2_factor: float,
) -> str:
    """Port lines 308-322 — ``Learning`` folder name."""
    lr = f"Initial Learning Rate - {_fmt_exp(initial_learning_rate)}"
    gt_str = _fmt_exp(gradient_threshold)
    grad = f"Gradient Threshold - {gt_str}"
    if gradient_clip_type == "Global":
        grad += " - Global"
    opt = f"Optimizer - {optimizer}"
    l2 = f"L2 Factor - {_fmt_exp(l2_factor)}"
    return " ~ ".join((lr, grad, opt, l2))


def _name_mini_batch(
    mini_batch_size: int,
    max_worker_mini_batch_size: int,
    want_stratified_partition: Any,
) -> str:
    """Port lines 330-343 — ``MiniBatchSize`` folder name."""
    name = (
        f"Mini Batch Size - {int(mini_batch_size):d} ~ "
        f"Max Accumulation - {int(max_worker_mini_batch_size):d}"
    )
    if isinstance(want_stratified_partition, bool):
        if want_stratified_partition:
            name += " ~ Hierarchically Stratified"
        else:
            name += " ~ Not Stratified"
    else:
        name += f" ~ {want_stratified_partition}"
    return name


def _name_data_augmentation(
    std_channel_offset: float | None,
    std_white_noise: float | None,
    std_random_walk: float | None,
    std_time_shift: float | None,
    want_separate_time_shift: bool,
) -> str:
    """Port lines 351-376 — ``DataAugmentation`` folder name."""
    parts = [
        f"Channel Offset - {_fmt_exp(std_channel_offset)}",
        f"White Noise - {_fmt_exp(std_white_noise)}",
        f"Random Walk - {_fmt_exp(std_random_walk)}",
    ]
    name = " ~ ".join(parts)
    if std_time_shift is not None and not (
        isinstance(std_time_shift, float) and math.isnan(std_time_shift)
    ):
        ts_str = _fmt_exp(std_time_shift)
        if want_separate_time_shift:
            name += f" ~ Separate TimeShift - {ts_str}"
        else:
            name += f" ~ TimeShift - {ts_str}"
    return name


def _name_is_subset(is_subset: bool | str, subset_session_name: bool | str) -> str:
    """Port lines 384-391 — ``IsSubset`` folder name.

    ``IsSubset`` toggles between ``"Subset"`` and ``"All Sessions"``;
    if a ``Subset`` field holds a concrete session string, the string
    wins.
    """
    if not isinstance(subset_session_name, bool):
        return str(subset_session_name)
    return "Subset" if is_subset else "All Sessions"


def _name_autoencoder(
    loss_type_decoder: str,
    num_epochs_autoencoder: int,
    prior_proportion: float,
    rescale_loss_epoch: int,
) -> str:
    """Port lines 399-409 — ``AutoEncoder`` folder name."""
    if loss_type_decoder == "None":
        epochs = "AutoEncoder"
        loss = "None"
    else:
        epochs = f"AutoEncoder - Epochs - {int(num_epochs_autoencoder):d}"
        loss = f"Loss Function - {loss_type_decoder}"
    rescale = (
        f"Prior Proportion - {_fmt_exp(prior_proportion)} ~ "
        f"Rescale Epochs - {int(rescale_loss_epoch):d}"
    )
    return f"{epochs} ~ {loss} ~ {rescale}"


def _name_loss(
    weight_reconstruction: float | None,
    weight_classification: float | None,
    weight_kl: float | None,
    weight_confidence: float | None,
    confidence_type: Iterable[str] | None,
    want_batch_correction: bool,
) -> str:
    """Port lines 416-451 — ``Loss`` folder name."""
    parts = [
        f"Weight Reconstruction - {_fmt_exp(weight_reconstruction)}",
        f"Weight Classification - {_fmt_exp(weight_classification)}",
        f"Weight KL - {_fmt_exp(weight_kl)}",
    ]
    # Confidence label: sort the types after stripping " Confidence" suffix
    # and join with " and ". Empty / "" entries are dropped.
    confidence_list = (
        [c for c in confidence_type if c and str(c).strip()]
        if confidence_type is not None
        else []
    )
    name_confidence = ""
    if confidence_list:
        cleaned = sorted(str(c).replace(" Confidence", "") for c in confidence_list)
        name_confidence = f" {cleaned[0]}"
        for c in cleaned[1:]:
            name_confidence = f"{name_confidence} and {c}"
        if want_batch_correction:
            name_confidence = f" BC{name_confidence}"
    if weight_confidence is None or (
        isinstance(weight_confidence, float) and math.isnan(weight_confidence)
    ):
        parts.append(f"Weight{name_confidence} Confidence - None")
    elif float(weight_confidence) != 0:
        parts.append(
            f"Weight{name_confidence} Confidence - {_fmt_exp(weight_confidence)}"
        )
    return " ~ ".join(parts)


def _name_dynamic(
    dynamic_parameter_set: str,
    stitching_and_fusion_layer: str,
) -> str:
    """Port lines 466-470 — ``Dynamic`` folder name."""
    name = f"Dynamic Set - {dynamic_parameter_set}"
    if stitching_and_fusion_layer and stitching_and_fusion_layer != "":
        name += f" ~ S and F - {stitching_and_fusion_layer}"
    return name


def _name_classifier(
    classifier_name: str,
    classifier_hidden_size: Sequence[int] | int,
    weighted_loss: str,
    multiple_instance_learning_type: str,
) -> str:
    """Port lines 484-508 — ``Classifier`` folder name."""
    parts = [
        f"Classifier - {classifier_name}",
        f"Hidden Size - {_fmt_hidden_size(classifier_hidden_size)}",
        f"Weighted Loss - {weighted_loss}" if weighted_loss else "Weighted Loss - None",
    ]
    name = " ~ ".join(parts)
    if (
        multiple_instance_learning_type
        and multiple_instance_learning_type == "MIL"
    ):
        name += " ~ SCT"
    return name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatlabRunDirs:
    """Per-fold result directories produced by :func:`build_matlab_run_dirs`.

    Attributes
    ----------
    classifier_fold
        Leaf classifier ``Fold_{N}`` directory — receives
        ``CM_Table.mat``, ``CM_Table_Validation.mat``,
        ``EncodingParameters.yaml``, plus state checkpoints.
    autoencoder_fold
        Parallel ``Information/Fold_{N}`` directory — used when
        ``num_epochs_autoencoder > 0`` to hold the Stage 1 (unsupervised)
        autoencoder weights.
    encoding_dir
        Top-level ``Encoding/{Target}`` directory (used by Critical
        Note #15's aggregator discovery).
    """

    classifier_fold: Path
    autoencoder_fold: Path
    encoding_dir: Path


def build_matlab_run_dirs(
    *,
    base_dir: Path,
    cfg: Any,
) -> MatlabRunDirs:
    """Build the full long-folder hierarchy for one (config, fold) run.

    Reads every relevant field from ``cfg`` (using ``cfg.get(key, default)``
    so legacy configs missing a field still produce a path). Path
    components are joined with ``Path.__truediv__`` — caller is
    responsible for ``mkdir(parents=True, exist_ok=True)`` on whichever
    leaf they want to write to.

    Parameters
    ----------
    base_dir
        Top-level results root. The hierarchy begins with
        ``Aggregate Data / Epoched Data / ...`` under this directory.
    cfg
        Resolved config (``omegaconf.DictConfig`` or plain ``dict`` with
        the same key set). The fold index is read from
        ``cfg.fold`` (1-based).

    Returns
    -------
    :class:`MatlabRunDirs`
        Triple of (classifier fold, autoencoder fold, encoding dir)
        paths.

    Raises
    ------
    ValueError
        If ``cfg.fold < 1`` or any required string field is empty.
    """
    fold = int(_cfg_get(cfg, "fold", 1))
    if fold < 1:
        raise ValueError(f"fold must be >= 1 (MATLAB 1-indexed); got {fold}.")

    epoch = str(_cfg_get(cfg, "epoch", ""))
    target = str(_cfg_get(cfg, "target", ""))
    model_name = str(_cfg_get(cfg, "model_name", ""))
    for label, value in (("epoch", epoch), ("target", target), ("model_name", model_name)):
        if not value.strip():
            raise ValueError(f"{label} must be a non-empty string.")

    model_params = _name_model_parameters(
        is_variational=bool(_cfg_get(cfg, "is_variational", False)),
        encoder_output_type=str(_cfg_get(cfg, "encoder_output_type", "")),
        activation=str(_cfg_get(cfg, "activation", "")),
        dropout=float(_cfg_get(cfg, "dropout", 0.0)),
        want_normalization=_cfg_get(cfg, "want_normalization", False),
        bottle_neck_depth=int(_cfg_get(cfg, "bottle_neck_depth", 1)),
    )
    width_stride = _name_width_stride(
        data_width=_cfg_get(cfg, "data_width", "All"),
        window_stride=_cfg_get(cfg, "window_stride", "All"),
        start_end_percent=_coerce_pair(_cfg_get(cfg, "start_end_percent", None)),
    )
    normalization = _name_normalization(
        str(_cfg_get(cfg, "normalization", "None"))
    )
    hidden_size = _name_hidden_size(list(_cfg_get(cfg, "hidden_sizes", [])))
    learning = _name_learning(
        initial_learning_rate=float(_cfg_get(cfg, "initial_learning_rate", 0.0)),
        gradient_threshold=_coerce_optional_float(
            _cfg_get(cfg, "gradient_threshold", None)
        ),
        gradient_clip_type=str(_cfg_get(cfg, "gradient_clip_type", "")),
        optimizer=str(_cfg_get(cfg, "optimizer", "ADAM")),
        l2_factor=float(_cfg_get(cfg, "l2_factor", 0.0)),
    )
    mini_batch = _name_mini_batch(
        mini_batch_size=int(_cfg_get(cfg, "mini_batch_size", 1)),
        max_worker_mini_batch_size=int(
            _cfg_get(cfg, "max_worker_mini_batch_size", 1)
        ),
        want_stratified_partition=_cfg_get(cfg, "want_stratified_partition", False),
    )
    data_augmentation = _name_data_augmentation(
        std_channel_offset=_coerce_optional_float(_cfg_get(cfg, "std_channel_offset", None)),
        std_white_noise=_coerce_optional_float(_cfg_get(cfg, "std_white_noise", None)),
        std_random_walk=_coerce_optional_float(_cfg_get(cfg, "std_random_walk", None)),
        std_time_shift=_coerce_optional_float(_cfg_get(cfg, "std_time_shift", None)),
        want_separate_time_shift=bool(_cfg_get(cfg, "want_separate_time_shift", False)),
    )
    # IsSubset matches MATLAB cfg_Encoder.wantSubset semantics, but the
    # Python `subset` field stores the resolved session name directly.
    subset_raw = _cfg_get(cfg, "subset", True)
    is_subset = _name_is_subset(
        is_subset=bool(subset_raw) if isinstance(subset_raw, bool) else True,
        subset_session_name=subset_raw,
    )
    autoencoder = _name_autoencoder(
        loss_type_decoder=str(_cfg_get(cfg, "loss_type_decoder", "MSE")),
        num_epochs_autoencoder=int(_cfg_get(cfg, "num_epochs_autoencoder", 0)),
        prior_proportion=float(_cfg_get(cfg, "prior_proportion", 0.9)),
        rescale_loss_epoch=int(_cfg_get(cfg, "rescale_loss_epoch", 0)),
    )
    loss = _name_loss(
        weight_reconstruction=_coerce_optional_float(_cfg_get(cfg, "weight_reconstruction", None)),
        weight_classification=_coerce_optional_float(_cfg_get(cfg, "weight_classification", None)),
        weight_kl=_coerce_optional_float(_cfg_get(cfg, "weight_kl", None)),
        weight_confidence=_coerce_optional_float(_cfg_get(cfg, "weight_confidence", None)),
        confidence_type=_cfg_get(cfg, "confidence_type", None),
        want_batch_correction=bool(_cfg_get(cfg, "want_batch_correction", False)),
    )
    dynamic = _name_dynamic(
        dynamic_parameter_set=str(_cfg_get(cfg, "dynamic_parameter_set", "None")),
        stitching_and_fusion_layer=str(_cfg_get(cfg, "stitching_and_fusion_layer", "")),
    )
    classifier = _name_classifier(
        classifier_name=str(_cfg_get(cfg, "classifier_name", "")),
        classifier_hidden_size=list(_cfg_get(cfg, "classifier_hidden_size", [])),
        weighted_loss=str(_cfg_get(cfg, "weighted_loss", "")),
        multiple_instance_learning_type=str(
            _cfg_get(cfg, "multiple_instance_learning_type", "")
        ),
    )

    encoding_dir = (
        Path(base_dir)
        / "Aggregate Data"
        / "Epoched Data"
        / _sanitize_for_path(epoch)
        / "Encoding"
        / _sanitize_for_path(target)
    )
    common_subtree = (
        encoding_dir
        / _sanitize_for_path(model_name)
        / _sanitize_for_path(model_params)
        / _sanitize_for_path(width_stride)
        / _sanitize_for_path(normalization)
        / _sanitize_for_path(hidden_size)
        / _sanitize_for_path(learning)
        / _sanitize_for_path(mini_batch)
        / _sanitize_for_path(data_augmentation)
        / _sanitize_for_path(is_subset)
        / _sanitize_for_path(autoencoder)
        / _sanitize_for_path(loss)
        / _sanitize_for_path(dynamic)
    )
    classifier_fold = (
        common_subtree
        / _sanitize_for_path(classifier)
        / f"Fold_{fold:d}"
    )
    autoencoder_fold = (
        common_subtree
        / "Information"
        / f"Fold_{fold:d}"
    )
    return MatlabRunDirs(
        classifier_fold=classifier_fold,
        autoencoder_fold=autoencoder_fold,
        encoding_dir=encoding_dir,
    )


# ---------------------------------------------------------------------------
# Cfg accessors (omegaconf / dict tolerant)
# ---------------------------------------------------------------------------


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """``cfg.get(key, default)`` with DictConfig and plain-dict support."""
    try:
        if hasattr(cfg, "get"):
            value = cfg.get(key, default)
        else:
            value = getattr(cfg, key, default)
    except Exception:
        return default
    return value


def _coerce_optional_float(value: Any) -> float | None:
    """Coerce to ``float`` when possible; ``None`` for sentinel/missing values."""
    if value is None:
        return None
    if isinstance(value, str):
        if value == "" or value.lower() == "none":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_pair(value: Any) -> tuple[float | None, float | None] | None:
    """Coerce ``cfg.start_end_percent`` into a ``(lo, hi)`` float pair."""
    if value is None:
        return None
    try:
        lst = list(value)
    except TypeError:
        return None
    if len(lst) < 2:
        return None
    return (_coerce_optional_float(lst[0]), _coerce_optional_float(lst[1]))


__all__ = [
    "MatlabRunDirs",
    "build_matlab_run_dirs",
]
