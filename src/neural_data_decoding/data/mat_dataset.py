"""Real-data :class:`MatFileTrialDataset` — one .mat per trial, paired Target.

Ports the load + window + target-select semantics of MATLAB
``cgg_loadDataArray.m`` and ``cgg_loadTargetArray.m`` (called from
``cgg_procAutoEncoder.m``) into a PyTorch :class:`~torch.utils.data.Dataset`.

Layout on disk
--------------
Two sibling directories per epoch:

* ``data_dir/Decision_Data_NNNNNNN.mat`` — variable ``Data`` of shape
  ``(C=NumChannels, TT=NumSamples, A=NumProbes)``, ``float64``. NaN at
  removed-channel positions.
* ``target_dir/Target_NNNNNNN.mat`` — struct ``Target`` with ~46 fields
  (``SelectedObjectDimVals``, ``CorrectTrial``, ``Dimensionality``,
  ``Gain``, ``Loss``, ``SessionName`` etc.).

Pairing is by the zero-padded numeric suffix.

Windowing
---------
Mirrors ``cgg_loadDataArray.m`` (line 14 ``[C, TT, A] = size(Data)``,
line 183 ``StartPoint_IDX = Start:Stride:FinalStart``). Per MATLAB
1-indexing the last possible start is ``NumSamples + 1 - DataWidth``;
the Python conversion uses 0-indexed inclusive end so the start grid
is ``range(start, final_start + 1, stride)``.

The output trial tensor is ``(W, T, A, C)`` — same axis order as
:class:`~neural_data_decoding.data.dataset.SyntheticTrialDataset`, so
the composites and samplers work unchanged. NaN positions are
**preserved** (Critical Note #38); the encoder input is responsible
for zeroing them.

Target dispatch
---------------
Mirrors ``cgg_loadTargetArray.m``. The supported ``target_type`` values
are ``Dimension`` (default — multi-dim object features), ``CorrectTrial``
(alias ``Outcome`` / ``Trial Outcome``; binary), ``Dimensionality``,
``Gain``, ``Loss``, and ``DataNumber``. **Always returns shape
``(num_dims,)`` even when ``num_dims == 1``** — never squeezes. The
MATLAB CM-table aggregator used a naked ``squeeze()`` and broke for
1-D-target configs (e.g. ``CorrectTrial``); this dataset removes the
ambiguity at the source.

Class-index mapping
-------------------
The raw target values are integer feature labels (e.g. ``0, 5, 8`` for
quaddle feature values). The classifier expects dense ``[0, k-1]``
class indices per output dim. The dataset builds a per-dim
``raw_value → class_idx`` mapping at construction by scanning every
trial's target, sorted ascending. The mapping is exposed as
:attr:`class_mapping_per_dim` (list of ``dict[int, int]``) and the
inverse per-dim class count as :attr:`num_classes_per_dim` — feed
this directly into the model builder.

A pre-built mapping can be passed via ``class_mapping_per_dim`` to
hold the class space constant across train/val/test splits.

See ``docs/MILESTONE_D_PLAN.md`` for the broader plan and
``cgg_procAutoEncoder.m`` line 235-247 for the source-side
target-function dispatch.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from neural_data_decoding.data.augmentation import (
    additive_augmentation_signal,
    generate_time_shift_samples,
)
from neural_data_decoding.data.mat_files import load_mat
from neural_data_decoding.training.schedules import Schedule


# Quaddle feature dimensions used by ``cgg_loadTargetArray.m`` line 14
# (``FeatureDimensions = [1,2,3,5]``). MATLAB is 1-indexed; the Python
# equivalent is [0, 1, 2, 4].
_FEATURE_DIMENSIONS_DEFAULT: tuple[int, ...] = (0, 1, 2, 4)

# Default ``Dimension`` index list for the 'Dimension' target — MATLAB
# ``Dimension = 1:4`` (``cgg_procAutoEncoder.m`` line 24). After picking
# ``FeatureDimensions`` (length 4), all four are returned.
_DIMENSION_INDICES_DEFAULT: tuple[int, ...] = (0, 1, 2, 3)

_TRIAL_ID_PATTERN = re.compile(r"_(\d+)\.mat$", re.IGNORECASE)


# ----------------------------------------------------------------------
# Struct-format normalization
# ----------------------------------------------------------------------


def _normalize_target_struct(raw: Any) -> dict[str, Any]:
    """Normalize a ``Target`` payload to a ``dict`` regardless of backend.

    The two loaders return different shapes for MATLAB struct values:

    * :mod:`mat73` (v7.3 / HDF5) returns a plain ``dict``.
    * :mod:`scipy.io` (pre-v7.3, with ``struct_as_record=False``) returns
      a ``numpy.ndarray`` (often a ``(1, 1)`` wrapper) of
      :class:`scipy.io.matlab._mio5_params.mat_struct` whose fields are
      attributes (queryable via ``_fieldnames``).

    Both forms are coerced to a flat ``dict[str, Any]`` here so the
    dispatch code can treat them identically.
    """
    if isinstance(raw, dict):
        return raw

    # scipy returns ndarray-wrapped structs even for scalars; unwrap.
    if isinstance(raw, np.ndarray):
        if raw.size != 1:
            raise ValueError(
                f"Expected a scalar Target struct, got ndarray of shape {raw.shape}"
            )
        raw = raw.item()

    fieldnames = getattr(raw, "_fieldnames", None)
    if fieldnames is None:
        raise TypeError(
            f"Cannot normalize Target payload of type {type(raw).__name__}"
        )
    return {name: getattr(raw, name) for name in fieldnames}


def _unbox_matlab_string(val: Any) -> str:
    """Coerce a MATLAB string field (possibly array-wrapped) to a Python ``str``."""
    if isinstance(val, str):
        return val
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return str(val.item())
        # Char array sometimes round-trips as a 1-D ndarray of strings.
        return str(val.tolist())
    return str(val)


# ----------------------------------------------------------------------
# Trial discovery + pairing
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TrialPaths:
    """One trial's (data, target) file pair plus the parsed suffix index."""

    trial_id: int
    data_path: Path
    target_path: Path


def _parse_trial_id(path: Path) -> int | None:
    """Return the trailing ``_NNNNNNN`` integer suffix of ``path``, or None."""
    m = _TRIAL_ID_PATTERN.search(path.name)
    if m is None:
        return None
    return int(m.group(1))


def _discover_trials(
    data_dir: Path,
    target_dir: Path,
    *,
    data_pattern: str,
    target_pattern: str,
) -> list[_TrialPaths]:
    """Pair data files with target files by trailing ``_NNNNNNN`` index.

    The MATLAB production layout puts Decision_Data_*.mat and Target_*.mat
    in separate directories, but smoke fixtures (e.g. ``results/Decision``)
    co-locate them. The per-side glob patterns let both layouts work
    without ambiguity.

    Raises
    ------
    FileNotFoundError
        If either directory does not exist.
    ValueError
        If any data file lacks a paired target file (or vice versa).
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"target_dir does not exist: {target_dir}")

    data_files = {
        tid: p
        for p in sorted(data_dir.glob(data_pattern))
        if (tid := _parse_trial_id(p)) is not None
    }
    target_files = {
        tid: p
        for p in sorted(target_dir.glob(target_pattern))
        if (tid := _parse_trial_id(p)) is not None
    }

    missing_targets = sorted(data_files.keys() - target_files.keys())
    missing_data = sorted(target_files.keys() - data_files.keys())
    if missing_targets or missing_data:
        msg_parts: list[str] = []
        if missing_targets:
            msg_parts.append(
                f"data files without matching target ({len(missing_targets)}): "
                f"{missing_targets[:5]}"
            )
        if missing_data:
            msg_parts.append(
                f"target files without matching data ({len(missing_data)}): "
                f"{missing_data[:5]}"
            )
        raise ValueError(
            f"Unpaired trial files between {data_dir} and {target_dir}: "
            + "; ".join(msg_parts)
        )

    return [
        _TrialPaths(trial_id=tid, data_path=data_files[tid], target_path=target_files[tid])
        for tid in sorted(data_files.keys())
    ]


# ----------------------------------------------------------------------
# Target dispatch
# ----------------------------------------------------------------------


def _target_field(target: dict[str, Any], name: str) -> Any:
    """Return ``target[name]`` raising a useful error if it is absent."""
    if name not in target:
        raise KeyError(
            f"Target struct missing field {name!r}; available fields: "
            f"{sorted(target.keys())[:10]}..."
        )
    return target[name]


def _select_dimension_target(
    target: dict[str, Any],
    feature_dimensions: Sequence[int],
    dimension_indices: Sequence[int],
) -> np.ndarray:
    """Port the ``case 'Dimension'`` branch of ``cgg_loadTargetArray.m``.

    Returns a length-``len(dimension_indices)`` int vector of object
    feature values for the trial.
    """
    vals = np.asarray(_target_field(target, "SelectedObjectDimVals"))
    if vals.ndim != 1:
        # mat73 sometimes returns nested singleton wrappers; flatten.
        vals = vals.reshape(-1)
    feature_dims = np.asarray(feature_dimensions, dtype=np.int64)
    dim_idx = np.asarray(dimension_indices, dtype=np.int64)
    if feature_dims.max(initial=-1) >= vals.size:
        raise IndexError(
            f"feature_dimensions={feature_dimensions} out of range for "
            f"SelectedObjectDimVals of length {vals.size}"
        )
    selected = vals[feature_dims]  # shape (len(feature_dims),)
    if dim_idx.max(initial=-1) >= selected.size:
        raise IndexError(
            f"dimension={dimension_indices} out of range for selected "
            f"feature-dim vector of length {selected.size}"
        )
    return np.asarray(selected[dim_idx], dtype=np.int64)


def _select_correct_trial(target: dict[str, Any]) -> np.ndarray:
    """Port the ``case 'CorrectTrial'`` branch — returns ``(1,)`` int label.

    MATLAB stores this as either the string ``'True'``/``'False'`` (older
    pipelines) or a logical (mat73 returns ``bool``); we coerce both
    forms to ``int64`` 0/1.
    """
    raw = _target_field(target, "CorrectTrial")
    if isinstance(raw, (bool, np.bool_)):
        val = int(raw)
    elif isinstance(raw, str):
        val = int(raw.strip().lower() == "true")
    elif isinstance(raw, (int, float, np.integer, np.floating)):
        val = int(raw)
    elif isinstance(raw, np.ndarray) and raw.size == 1:
        item = raw.item()
        val = int(item) if not isinstance(item, str) else int(item.strip().lower() == "true")
    else:
        raise TypeError(
            f"Unsupported CorrectTrial value type {type(raw).__name__}: {raw!r}"
        )
    return np.asarray([val], dtype=np.int64)


def _select_scalar_int(target: dict[str, Any], field: str) -> np.ndarray:
    """Read ``target[field]`` and return a ``(1,)`` int64 array."""
    raw = _target_field(target, field)
    if isinstance(raw, np.ndarray) and raw.size == 1:
        raw = raw.item()
    return np.asarray([int(raw)], dtype=np.int64)


def _select_data_number(trial_id: int) -> np.ndarray:
    """Port ``case 'DataNumber'`` — the trailing index parsed from filename."""
    return np.asarray([int(trial_id)], dtype=np.int64)


def _load_target_vector(
    target: dict[str, Any],
    *,
    trial_id: int,
    target_type: str,
    feature_dimensions: Sequence[int],
    dimension_indices: Sequence[int],
) -> np.ndarray:
    """Dispatch on ``target_type`` and return a ``(num_dims,)`` int label vector."""
    if target_type == "Dimension":
        return _select_dimension_target(target, feature_dimensions, dimension_indices)
    if target_type in {"CorrectTrial", "Outcome", "Trial Outcome"}:
        return _select_correct_trial(target)
    if target_type == "Dimensionality":
        return _select_scalar_int(target, "Dimensionality")
    if target_type == "Gain":
        return _select_scalar_int(target, "Gain")
    if target_type == "Loss":
        return _select_scalar_int(target, "Loss")
    if target_type == "DataNumber":
        return _select_data_number(trial_id)
    raise ValueError(
        f"Unsupported target_type={target_type!r}. Supported: "
        "Dimension, CorrectTrial/Outcome/Trial Outcome, Dimensionality, "
        "Gain, Loss, DataNumber."
    )


# ----------------------------------------------------------------------
# Window-start index grid (port of cgg_loadDataArray.m lines 130-184)
# ----------------------------------------------------------------------


def _compute_window_starts(
    num_samples: int,
    data_width: int,
    *,
    starting_idx: int,
    ending_idx: int | None,
    window_stride: int,
    start_end_percent: tuple[float | None, float | None],
) -> np.ndarray:
    """Return the 0-indexed window-start array for one trial.

    Mirrors the MATLAB logic at lines 130-184:

    1. ``final_possible_start = num_samples - data_width`` (0-indexed
       inclusive — the MATLAB ``NumSamples + 1 - DataWidth`` becomes
       ``NumSamples - DataWidth`` once we drop the 1-based offset).
    2. ``ending_idx`` (the **last allowed start**, not the last
       sample) gets clamped to ``final_possible_start``; ``None``
       means use the full range.
    3. ``start_end_percent`` overrides the absolute starts when its
       entries are in ``[0, 1]``.
    4. Window starts: ``arange(start, final_start + 1, stride)``.

    When ``data_width >= num_samples`` only one window is emitted, at
    ``start = 0``.
    """
    if data_width <= 0:
        raise ValueError(f"data_width must be > 0, got {data_width}")
    if window_stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {window_stride}")

    if data_width >= num_samples:
        return np.zeros(1, dtype=np.int64)

    final_possible_start = num_samples - data_width
    if ending_idx is None:
        final_start = final_possible_start
    else:
        final_start = min(ending_idx, final_possible_start)
    start = max(0, int(starting_idx))

    pct_lo, pct_hi = start_end_percent
    if pct_lo is not None and 0.0 <= pct_lo <= 1.0:
        start = int(round(num_samples * pct_lo))
    if pct_hi is not None and 0.0 <= pct_hi <= 1.0:
        final_start = int(round(num_samples * pct_hi - data_width))
        final_start = min(final_start, final_possible_start)

    start = max(0, start)
    if final_start < start:
        # Empty window grid — would produce a 0-window trial.
        return np.empty(0, dtype=np.int64)

    return np.arange(start, final_start + 1, window_stride, dtype=np.int64)


def _window_trial(
    data: np.ndarray,
    starts: np.ndarray,
    data_width: int,
    time_shift_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Window a single ``(C, TT, A)`` trial into ``(W, T, A, C)``.

    The Python axis order matches :class:`SyntheticTrialDataset`: per-
    window data is ``(T, A, C)`` and the leading ``W`` is the recurrent
    axis. MATLAB's ``(C, T, A, W)`` layout is permuted via ``moveaxis``.

    Parameters
    ----------
    data
        A single trial, shape ``(C, TT, A)`` — channels, samples, areas.
    starts
        Per-window start sample indices (0-indexed), shape ``(W,)``.
    data_width
        Samples per window (the ``T`` axis).
    time_shift_idx
        Optional per-``(channel, area, window)`` integer sample offsets,
        shape ``(C, A, W)`` (as returned by
        :func:`~neural_data_decoding.data.augmentation.generate_time_shift_samples`).
        When given, each window's sample range is shifted independently per
        ``(channel, area)`` and **clipped to** ``[0, TT - 1]`` before the
        gather — mirroring MATLAB ``cgg_getDataFromRange`` (which clips OOB
        indices to the signal edge). ``None`` (default) takes the fast
        contiguous-slice path with no shift.

    Returns
    -------
    numpy.ndarray
        The windowed trial, shape ``(W, T, A, C)``.
    """
    num_channels, num_samples, num_areas = data.shape
    num_windows = int(starts.size)
    # Build (C, T, A, W) first — mirrors MATLAB's `this_Data`.
    windowed = np.empty(
        (num_channels, data_width, num_areas, num_windows), dtype=data.dtype
    )
    if time_shift_idx is None:
        for w, s in enumerate(starts):
            windowed[:, :, :, w] = data[:, int(s) : int(s) + data_width, :]
    else:
        # Shifted gather (MATLAB cgg_getDataFromRange): each (channel, area)
        # reads from a range offset by its own shift and clipped to the signal.
        c_idx = np.arange(num_channels)[:, None, None]          # (C, 1, 1)
        a_idx = np.arange(num_areas)[None, None, :]             # (1, 1, A)
        for w, s in enumerate(starts):
            base = np.arange(int(s), int(s) + data_width)       # (T,)
            shift_ca = time_shift_idx[:, :, w]                  # (C, A)
            sidx = base[None, :, None] + shift_ca[:, None, :]   # (C, T, A)
            np.clip(sidx, 0, num_samples - 1, out=sidx)
            windowed[:, :, :, w] = data[c_idx, sidx, a_idx]     # (C, T, A)
    # Permute to (W, T, A, C).
    return np.transpose(windowed, (3, 1, 2, 0))


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------


class MatFileTrialDataset(Dataset):
    """One-trial-per-file ``.mat`` dataset for real Decision-epoch data.

    Wraps the disk layout produced by the MATLAB processing pipeline
    (``Decision_Data_NNNNNNN.mat`` + ``Target_NNNNNNN.mat``) in the
    same shape contract as :class:`SyntheticTrialDataset` so the existing
    composites, samplers, and trainers consume it unchanged.

    Parameters
    ----------
    data_dir
        Directory containing ``Decision_Data_*.mat`` files (one per
        trial, variable ``Data`` shape ``(C, TT, A)``).
    target_dir
        Directory containing ``Target_*.mat`` files (struct ``Target``).
    data_pattern, target_pattern
        Glob patterns used inside ``data_dir`` / ``target_dir`` to
        discover trial files. Defaults match the MATLAB pipeline output
        (``Decision_Data_*.mat`` / ``Target_*.mat``). Override when
        the two file types share a directory (smoke fixtures) or to
        load a different epoch.
    data_width
        Samples per window (MATLAB ``DataWidth``; the ``T`` axis).
    window_stride
        Stride between window starts (MATLAB ``WindowStride``). Set to
        ``data_width`` for non-overlapping windows.
    target_type
        Which target to dispatch. ``'Dimension'`` (default) returns the
        per-feature-dim object label vector; ``'CorrectTrial'`` /
        ``'Outcome'`` / ``'Trial Outcome'`` returns a binary ``(1,)``
        label; ``'Dimensionality'``, ``'Gain'``, ``'Loss'``,
        ``'DataNumber'`` return single-scalar ``(1,)`` labels.
    feature_dimensions
        Object feature dimensions to pull from ``SelectedObjectDimVals``
        (0-indexed). Default ``(0, 1, 2, 4)`` matches MATLAB
        ``FeatureDimensions = [1, 2, 3, 5]``.
    dimension_indices
        Which of the ``feature_dimensions`` entries become output dims
        (0-indexed into the post-selection vector). Default ``(0,1,2,3)``
        matches MATLAB ``Dimension = 1:4``.
    starting_idx, ending_idx
        First / last allowed **window start** (0-indexed; the ``T``
        samples that follow each start are the window). ``None`` for
        either means use the full range.
    start_end_percent
        Alternative bounds expressed as ``(lo, hi)`` fractions of
        ``NumSamples``. When set and in ``[0, 1]``, they override
        the absolute indices (MATLAB ``StartEndPercent``).
    session_filter
        Optional session-name allow-list. Trials whose
        ``Target.SessionName`` is not in this list are excluded at
        construction time. Pass a single string or any iterable of
        strings.
    class_mapping_per_dim
        Optional pre-built ``[raw_value → class_idx]`` dict per output
        dim (length must equal target arity). When ``None``, the
        mapping is built from the data at construction time by sorting
        each dim's unique raw labels ascending.
    load_schedule
        Optional :class:`~neural_data_decoding.training.schedules.Schedule`
        whose ``std_channel_offset`` / ``std_white_noise`` /
        ``std_random_walk`` entries drive per-trial additive augmentation,
        and whose ``std_time_shift`` entry (when present and non-NaN)
        drives per-window time-shift augmentation — all read **live** at
        every ``__getitem__`` call (Critical Note #8).
    augmentation_seed
        Independent RNG seed for the augmentation stream (both the additive
        and time-shift draws consume this generator).
    sampling_frequency
        Data sampling rate in Hz (MATLAB ``SamplingFrequency``). Used to
        convert the ms-based ``std_time_shift`` into a sample-count offset.
        Default ``1000.0`` (the Decision-epoch rate).
    want_separate_time_shift
        When ``True`` (MATLAB ``WantSeparateTimeShift``), each
        ``(channel, area, window)`` cell draws an independent time shift;
        when ``False``, one shift is broadcast across all cells of a trial.
    preload
        When ``True``, all trials are loaded and windowed during
        construction (faster training, more memory); note this bakes a
        single time-shift draw per trial rather than re-drawing live. When
        ``False`` (default), trials are loaded on every ``__getitem__``
        call — matches MATLAB ``fileDatastore`` behavior and keeps
        time-shift live.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        target_dir: str | Path,
        data_width: int,
        window_stride: int,
        data_pattern: str = "Decision_Data_*.mat",
        target_pattern: str = "Target_*.mat",
        target_type: str = "Dimension",
        feature_dimensions: Sequence[int] = _FEATURE_DIMENSIONS_DEFAULT,
        dimension_indices: Sequence[int] = _DIMENSION_INDICES_DEFAULT,
        starting_idx: int = 0,
        ending_idx: int | None = None,
        start_end_percent: tuple[float | None, float | None] = (None, None),
        session_filter: str | Iterable[str] | None = None,
        class_mapping_per_dim: list[dict[int, int]] | None = None,
        load_schedule: Schedule | None = None,
        augmentation_seed: int = 0,
        sampling_frequency: float = 1000.0,
        want_separate_time_shift: bool = True,
        preload: bool = False,
    ) -> None:
        data_dir_p = Path(data_dir)
        target_dir_p = Path(target_dir)
        trials = _discover_trials(
            data_dir_p,
            target_dir_p,
            data_pattern=data_pattern,
            target_pattern=target_pattern,
        )
        if not trials:
            raise ValueError(
                f"No paired (Decision_Data_*.mat, Target_*.mat) trials found in "
                f"{data_dir_p} and {target_dir_p}"
            )

        # Apply session filter early so we never load excluded trials' data.
        session_names = _read_session_names(trials)
        if session_filter is not None:
            allow = {session_filter} if isinstance(session_filter, str) else set(session_filter)
            kept = [
                (t, n) for t, n in zip(trials, session_names, strict=True) if n in allow
            ]
            if not kept:
                raise ValueError(
                    f"session_filter={session_filter!r} matched no trials. "
                    f"Available sessions: {sorted(set(session_names))[:10]}..."
                )
            trials = [t for t, _ in kept]
            session_names = [n for _, n in kept]

        self._trials = trials
        self._session_names = session_names

        # Stable session id assignment: alphabetical, 0-indexed.
        unique_sessions = sorted(set(session_names))
        self._session_name_to_id = {n: i for i, n in enumerate(unique_sessions)}
        self._session_ids = np.asarray(
            [self._session_name_to_id[n] for n in session_names], dtype=np.int64
        )

        self.data_width = int(data_width)
        self.window_stride = int(window_stride)
        self.target_type = target_type
        self.feature_dimensions = tuple(feature_dimensions)
        self.dimension_indices = tuple(dimension_indices)
        self.starting_idx = int(starting_idx)
        self.ending_idx = ending_idx if ending_idx is None else int(ending_idx)
        self.start_end_percent = start_end_percent
        self.load_schedule = load_schedule
        self.sampling_frequency = float(sampling_frequency)
        self.want_separate_time_shift = bool(want_separate_time_shift)
        self._aug_rng = np.random.default_rng(augmentation_seed)

        # Load all targets up front — cheap and needed for class mapping.
        raw_targets = [
            _load_target_vector(
                _normalize_target_struct(load_mat(t.target_path)["Target"]),
                trial_id=t.trial_id,
                target_type=self.target_type,
                feature_dimensions=self.feature_dimensions,
                dimension_indices=self.dimension_indices,
            )
            for t in trials
        ]
        targets_arr = np.stack(raw_targets, axis=0)  # (N, num_dims)
        num_dims = int(targets_arr.shape[1])

        # Class mapping per dim
        if class_mapping_per_dim is None:
            self.class_mapping_per_dim = [
                {int(v): i for i, v in enumerate(sorted(set(targets_arr[:, d].tolist())))}
                for d in range(num_dims)
            ]
        else:
            if len(class_mapping_per_dim) != num_dims:
                raise ValueError(
                    f"class_mapping_per_dim length {len(class_mapping_per_dim)} "
                    f"does not match target arity {num_dims}"
                )
            self.class_mapping_per_dim = [dict(m) for m in class_mapping_per_dim]

        # Apply the per-dim mapping to convert raw → dense class indices
        labels = np.empty_like(targets_arr)
        for d in range(num_dims):
            mapping = self.class_mapping_per_dim[d]
            for n in range(targets_arr.shape[0]):
                raw = int(targets_arr[n, d])
                if raw not in mapping:
                    raise KeyError(
                        f"Raw target value {raw} (trial {trials[n].trial_id}, dim "
                        f"{d}) is not in the provided class_mapping_per_dim[{d}]. "
                        f"Known values: {sorted(mapping.keys())[:10]}..."
                    )
                labels[n, d] = mapping[raw]
        self._labels = labels.astype(np.int64)
        self.num_classes_per_dim = [len(m) for m in self.class_mapping_per_dim]

        # Eager preload (optional)
        self._preloaded: list[np.ndarray] | None = None
        if preload:
            self._preloaded = [self._load_windowed(idx) for idx in range(len(trials))]

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def session_ids(self) -> np.ndarray:
        """Per-trial integer session id (consumed by SingleSessionBatchSampler)."""
        return self._session_ids

    @property
    def session_names(self) -> list[str]:
        """Per-trial session-name strings (in trial order)."""
        return list(self._session_names)

    @property
    def num_dimensions(self) -> int:
        """Number of classification output dimensions."""
        return len(self.num_classes_per_dim)

    @property
    def trial_ids(self) -> list[int]:
        """Per-trial filename suffix index (in trial order)."""
        return [t.trial_id for t in self._trials]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of trials kept after ``session_filter``."""
        return len(self._trials)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return ``(features, targets, metadata)`` for trial ``idx``.

        Features are ``(W, T, A, C)`` ``float32``; targets are the
        per-output-dim class-index vector ``(num_dimensions,)``
        ``int64`` (never squeezed). Metadata carries ``session_id``,
        ``session_name``, and ``trial_id`` (filename suffix).
        """
        if idx < 0 or idx >= len(self._trials):
            raise IndexError(f"Trial index {idx} out of range [0, {len(self._trials)}).")

        if self._preloaded is not None:
            features_np = self._preloaded[idx]
        else:
            features_np = self._load_windowed(idx)

        if self.load_schedule is not None:
            features_np = self._apply_augmentation(features_np)

        features = torch.from_numpy(np.ascontiguousarray(features_np)).float()
        targets = torch.from_numpy(self._labels[idx]).long()
        metadata = {
            "session_id": int(self._session_ids[idx]),
            "session_name": self._session_names[idx],
            "trial_id": int(self._trials[idx].trial_id),
        }
        return features, targets, metadata

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_windowed(self, idx: int) -> np.ndarray:
        """Load + window trial ``idx``. Returns float32 ``(W, T, A, C)``."""
        trial = self._trials[idx]
        data = np.asarray(load_mat(trial.data_path)["Data"])
        if data.ndim != 3:
            raise ValueError(
                f"{trial.data_path}: expected 3-D Data (C, TT, A), got shape {data.shape}"
            )
        num_samples = int(data.shape[1])
        starts = _compute_window_starts(
            num_samples=num_samples,
            data_width=self.data_width,
            starting_idx=self.starting_idx,
            ending_idx=self.ending_idx,
            window_stride=self.window_stride,
            start_end_percent=self.start_end_percent,
        )
        if starts.size == 0:
            raise ValueError(
                f"Window grid is empty for trial {trial.trial_id} "
                f"(num_samples={num_samples}, data_width={self.data_width}, "
                f"window_stride={self.window_stride}, starting_idx={self.starting_idx}, "
                f"ending_idx={self.ending_idx})."
            )
        time_shift_idx = self._time_shift_idx(data.shape[0], data.shape[2], int(starts.size))
        windowed = _window_trial(
            data, starts, self.data_width, time_shift_idx=time_shift_idx
        )
        return windowed.astype(np.float32, copy=False)

    def _time_shift_idx(
        self, num_channels: int, num_areas: int, num_windows: int
    ) -> np.ndarray | None:
        """Draw per-window time-shift offsets, or ``None`` when inactive.

        Time-shift is active only when a :attr:`load_schedule` is attached and
        carries a non-NaN ``std_time_shift`` (so the discovery pass and the
        un-augmented validation/test splits — both built without a schedule —
        take the fast unshifted windowing path). The magnitude is read live
        from the schedule, mirroring the additive augmentation.

        Parameters
        ----------
        num_channels, num_areas, num_windows
            Trial dimensions the shift array is drawn over.

        Returns
        -------
        numpy.ndarray or None
            Integer sample offsets, shape ``(C, A, W)``, or ``None`` when
            time-shift is disabled for this dataset.
        """
        sched = self.load_schedule
        if sched is None or "std_time_shift" not in sched:
            return None
        std = sched.current("std_time_shift")
        if std is None or math.isnan(float(std)):
            return None
        return generate_time_shift_samples(
            num_channels=num_channels,
            num_probes=num_areas,
            num_windows=num_windows,
            std_time_shift=float(std),
            sampling_frequency=self.sampling_frequency,
            want_separate=self.want_separate_time_shift,
            rng=self._aug_rng,
        )

    def _apply_augmentation(self, features_np: np.ndarray) -> np.ndarray:
        """Add a live-read augmentation tensor to a trial's windowed features.

        Reads the augmentation magnitudes from :attr:`load_schedule` at
        call time, never snapshots them (Critical Note #8). The signal
        helper expects ``(C, TT, A)`` so we flatten ``(W, T)`` →
        effective sample axis, generate, then unflatten.
        """
        assert self.load_schedule is not None  # noqa: S101 — guarded by caller
        sched = self.load_schedule

        def _current_or_none(name: str) -> float | None:
            return sched.current(name) if name in sched else None

        w, t, a, c = features_np.shape
        n_samp = w * t
        signal = additive_augmentation_signal(
            shape=(c, n_samp, a),
            std_channel_offset=_current_or_none("std_channel_offset"),
            std_white_noise=_current_or_none("std_white_noise"),
            std_random_walk=_current_or_none("std_random_walk"),
            rng=self._aug_rng,
        )
        # (C, W*T, A) → (W, T, A, C)
        signal = np.transpose(signal.reshape(c, w, t, a), (1, 2, 3, 0))
        return features_np + signal.astype(features_np.dtype, copy=False)


# ----------------------------------------------------------------------
# Session-name probe (used before the full target dispatch runs)
# ----------------------------------------------------------------------


def _read_session_names(trials: list[_TrialPaths]) -> list[str]:
    """Extract ``Target.SessionName`` for every trial (sorted by trial id)."""
    names: list[str] = []
    for t in trials:
        target_struct = _normalize_target_struct(load_mat(t.target_path)["Target"])
        if "SessionName" not in target_struct:
            raise KeyError(
                f"{t.target_path}: Target struct missing SessionName field."
            )
        names.append(_unbox_matlab_string(target_struct["SessionName"]))
    return names


__all__ = [
    "MatFileTrialDataset",
]
