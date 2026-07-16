"""Tests for :mod:`neural_data_decoding.data.mat_dataset`.

Two flavors of fixture cover the dataset:

* A **real** sample trial pair (``results/Decision/Decision_Data_0000011.mat``
  + ``Target_0000011.mat``) is read directly from the repo when present.
  The data shape ``(58, 3001, 6)`` and ``SelectedObjectDimVals`` =
  ``[0, 0, 8, 5, 3]`` were verified via MATLAB MCP and are pinned as
  ground truth in :func:`test_real_fixture_windowing_matches_raw_indexing`
  and :func:`test_real_fixture_dimension_target_matches_matlab`.

* **Synthetic** fixtures generated on the fly via ``scipy.io.savemat``
  give multi-trial coverage for session filtering, class-mapping
  construction, NaN preservation, and the 1-D ``CorrectTrial`` target
  (which is where the MATLAB CM-table squeeze bug originated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io as sio
import torch

from neural_data_decoding.data.mat_dataset import (
    MatFileTrialDataset,
    _compute_window_starts,
    _window_trial,
)
from neural_data_decoding.data.mat_files import load_mat


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_FIXTURE_DIR = REPO_ROOT / "results" / "Decision"
REAL_DATA_PATH = REAL_FIXTURE_DIR / "Decision_Data_0000011.mat"
REAL_TARGET_PATH = REAL_FIXTURE_DIR / "Target_0000011.mat"


# ----------------------------------------------------------------------
# Window-start computation
# ----------------------------------------------------------------------


def test_window_starts_full_range_with_stride() -> None:
    """``starting_idx=0``, ``ending_idx=None``, stride 50, width 100 on N=3001 → 59 windows."""
    starts = _compute_window_starts(
        num_samples=3001,
        data_width=100,
        starting_idx=0,
        ending_idx=None,
        window_stride=50,
        start_end_percent=(None, None),
    )
    assert starts.tolist()[:3] == [0, 50, 100]
    assert int(starts[-1]) == 2900
    assert starts.size == 59


def test_window_starts_width_equals_samples_yields_single_window() -> None:
    """When ``data_width >= num_samples`` only one window is returned."""
    starts = _compute_window_starts(
        num_samples=100,
        data_width=100,
        starting_idx=0,
        ending_idx=None,
        window_stride=10,
        start_end_percent=(None, None),
    )
    assert starts.tolist() == [0]


def test_window_starts_ending_idx_clamps_to_max() -> None:
    """An ``ending_idx`` past the maximum start gets clamped silently."""
    starts = _compute_window_starts(
        num_samples=200,
        data_width=50,
        starting_idx=0,
        ending_idx=10_000,
        window_stride=50,
        start_end_percent=(None, None),
    )
    # Final possible start = 200 - 50 = 150 → starts 0, 50, 100, 150.
    assert starts.tolist() == [0, 50, 100, 150]


def test_window_starts_start_end_percent_overrides_absolute_indices() -> None:
    """``start_end_percent`` overrides the absolute start/end indices."""
    starts = _compute_window_starts(
        num_samples=1000,
        data_width=100,
        starting_idx=0,
        ending_idx=None,
        window_stride=100,
        start_end_percent=(0.2, 0.5),
    )
    # start = round(1000 * 0.2) = 200
    # final_start = round(1000 * 0.5 - 100) = 400
    # → starts 200, 300, 400.
    assert starts.tolist() == [200, 300, 400]


def test_window_starts_rejects_nonpositive_params() -> None:
    """Zero or negative ``data_width`` / ``window_stride`` raise."""
    with pytest.raises(ValueError, match="data_width"):
        _compute_window_starts(
            num_samples=100,
            data_width=0,
            starting_idx=0,
            ending_idx=None,
            window_stride=1,
            start_end_percent=(None, None),
        )
    with pytest.raises(ValueError, match="window_stride"):
        _compute_window_starts(
            num_samples=100,
            data_width=10,
            starting_idx=0,
            ending_idx=None,
            window_stride=0,
            start_end_percent=(None, None),
        )


# ----------------------------------------------------------------------
# Window extraction
# ----------------------------------------------------------------------


def test_window_trial_pulls_correct_slices_and_transposes() -> None:
    """The ``(C, TT, A)`` → ``(W, T, A, C)`` permutation preserves values."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 50, 2)).astype(np.float64)  # (C=4, TT=50, A=2)
    starts = np.array([0, 10, 20], dtype=np.int64)
    out = _window_trial(data, starts, data_width=10)  # (W=3, T=10, A=2, C=4)
    assert out.shape == (3, 10, 2, 4)
    # Spot check: out[w, t, a, c] should equal data[c, starts[w] + t, a].
    for w_idx, s in enumerate(starts):
        for t_idx in range(10):
            for a_idx in range(2):
                for c_idx in range(4):
                    assert out[w_idx, t_idx, a_idx, c_idx] == data[c_idx, s + t_idx, a_idx]


# ----------------------------------------------------------------------
# Time-shift windowing (MATLAB cgg_getDataFromRange parity)
# ----------------------------------------------------------------------


def _ramp_data(num_channels: int, num_samples: int, num_areas: int) -> np.ndarray:
    """Trial where ``data[c, t, a] == t`` — the value reveals the sample pulled."""
    ramp = np.arange(num_samples, dtype=np.float64)
    return np.broadcast_to(
        ramp[None, :, None], (num_channels, num_samples, num_areas)
    ).copy()


def test_window_trial_zero_shift_equals_no_shift() -> None:
    """An all-zero shift array reproduces the unshifted windowing exactly."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 40, 2)).astype(np.float64)
    starts = np.array([0, 8, 16], dtype=np.int64)
    zero_shift = np.zeros((3, 2, 3), dtype=np.int64)  # (C, A, W)
    plain = _window_trial(data, starts, data_width=10)
    shifted = _window_trial(data, starts, data_width=10, time_shift_idx=zero_shift)
    assert np.array_equal(plain, shifted)


def test_window_trial_uniform_shift_moves_every_window() -> None:
    """A single shift shared across cells offsets each window's sample range."""
    data = _ramp_data(2, 30, 2)  # data[c, t, a] == t
    starts = np.array([5, 12], dtype=np.int64)
    delta = 3
    shift = np.full((2, 2, 2), delta, dtype=np.int64)  # (C, A, W)
    out = _window_trial(data, starts, data_width=6, time_shift_idx=shift)
    # out[w, t, a, c] == clip(starts[w] + t + delta, 0, 29); interior -> no clip.
    for w_idx, s in enumerate(starts):
        for t_idx in range(6):
            assert np.all(out[w_idx, t_idx] == s + t_idx + delta)


def test_window_trial_clips_shift_to_signal_edge() -> None:
    """Shifts past the signal end clamp to the last sample (MATLAB clip)."""
    data = _ramp_data(1, 20, 1)  # values 0..19
    starts = np.array([15], dtype=np.int64)
    shift = np.full((1, 1, 1), 100, dtype=np.int64)  # far past the end
    out = _window_trial(data, starts, data_width=6, time_shift_idx=shift)
    assert np.all(out == 19)  # every position clamped to the final sample
    # And a large negative shift clamps to sample 0.
    neg = np.full((1, 1, 1), -100, dtype=np.int64)
    out_neg = _window_trial(data, starts, data_width=6, time_shift_idx=neg)
    assert np.all(out_neg == 0)


def test_window_trial_separate_shift_is_per_cell() -> None:
    """With want_separate, each (channel, area) pulls from its own offset."""
    data = _ramp_data(2, 30, 2)  # data[c, t, a] == t
    starts = np.array([10], dtype=np.int64)
    # Distinct shift per (channel, area) for the single window.
    shift = np.array([[[1], [2]], [[3], [4]]], dtype=np.int64)  # (C=2, A=2, W=1)
    out = _window_trial(data, starts, data_width=4, time_shift_idx=shift)  # (1,4,2,2)
    for t_idx in range(4):
        assert out[0, t_idx, 0, 0] == 10 + t_idx + 1  # (c=0, a=0)
        assert out[0, t_idx, 1, 0] == 10 + t_idx + 2  # (a=1, c=0)
        assert out[0, t_idx, 0, 1] == 10 + t_idx + 3  # (c=1, a=0)
        assert out[0, t_idx, 1, 1] == 10 + t_idx + 4  # (c=1, a=1)


def test_window_trial_time_shift_from_generator_is_deterministic() -> None:
    """generate_time_shift_samples + _window_trial reproduces under a fixed seed."""
    from neural_data_decoding.data.augmentation import generate_time_shift_samples

    data = _ramp_data(2, 60, 2)
    starts = np.array([10, 20, 30], dtype=np.int64)

    def draw() -> np.ndarray:
        rng = np.random.default_rng(7)
        shift = generate_time_shift_samples(
            num_channels=2, num_probes=2, num_windows=3,
            std_time_shift=100.0, sampling_frequency=1000.0,
            want_separate=True, rng=rng,
        )
        return _window_trial(data, starts, data_width=8, time_shift_idx=shift)

    assert np.array_equal(draw(), draw())


# ----------------------------------------------------------------------
# Synthetic-fixture multi-trial coverage
# ----------------------------------------------------------------------


def _save_synthetic_trial(
    data_dir: Path,
    target_dir: Path,
    trial_id: int,
    *,
    session_name: str,
    dim_vals: list[int],
    correct_trial: bool = True,
    dimensionality: int = 3,
    gain: int = 5,
    loss: int = 0,
    num_channels: int = 4,
    num_samples: int = 200,
    num_areas: int = 2,
    seed: int | None = None,
) -> None:
    """Write a paired Decision_Data + Target .mat to disk via scipy savemat."""
    rng = np.random.default_rng(seed if seed is not None else trial_id)
    data = rng.standard_normal((num_channels, num_samples, num_areas))
    # Drop one channel of area 0 (introduces NaN) — mirrors the real fixture.
    data[0, :, 0] = np.nan
    sio.savemat(
        str(data_dir / f"Decision_Data_{trial_id:07d}.mat"),
        {"Data": data},
    )
    target: dict[str, Any] = {
        "SelectedObjectDimVals": np.asarray(dim_vals, dtype=np.float64),
        "CorrectTrial": "True" if correct_trial else "False",
        "Dimensionality": float(dimensionality),
        "Gain": float(gain),
        "Loss": float(loss),
        "SessionName": session_name,
    }
    sio.savemat(str(target_dir / f"Target_{trial_id:07d}.mat"), {"Target": target})


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> Path:
    """Build a 6-trial corpus across two sessions with varied targets."""
    data_dir = tmp_path / "data"
    target_dir = tmp_path / "target"
    data_dir.mkdir()
    target_dir.mkdir()
    # session A: 4 trials with feature-dim values that vary across trials
    for i, dim_vals in enumerate(
        [[0, 1, 2, 3, 4], [5, 1, 7, 3, 4], [0, 2, 2, 3, 9], [5, 2, 7, 3, 9]]
    ):
        _save_synthetic_trial(
            data_dir,
            target_dir,
            trial_id=i + 1,
            session_name="SessA_001_01",
            dim_vals=dim_vals,
            correct_trial=(i % 2 == 0),
        )
    # session B: 2 trials
    for i, dim_vals in enumerate([[0, 1, 2, 3, 4], [5, 2, 7, 3, 9]]):
        _save_synthetic_trial(
            data_dir,
            target_dir,
            trial_id=10 + i + 1,
            session_name="SessB_002_01",
            dim_vals=dim_vals,
            correct_trial=False,
        )
    return tmp_path


def test_synthetic_corpus_full_load(synthetic_corpus: Path) -> None:
    """Default config loads all 6 trials, builds class maps, returns 4-dim targets."""
    ds = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
    )
    assert len(ds) == 6
    assert ds.trial_ids == [1, 2, 3, 4, 11, 12]
    assert ds.session_names == [
        "SessA_001_01",
        "SessA_001_01",
        "SessA_001_01",
        "SessA_001_01",
        "SessB_002_01",
        "SessB_002_01",
    ]
    # FeatureDimensions defaults [0,1,2,4] then Dimension defaults [0,1,2,3]:
    # → 4 output dims.
    assert ds.num_dimensions == 4
    # Per-dim unique raw values across the 6 trials:
    # dim 0 (index 0 of SelectedObjectDimVals): {0, 5} → 2 classes
    # dim 1 (index 1): {1, 2} → 2 classes
    # dim 2 (index 2): {2, 7} → 2 classes
    # dim 3 (index 4 of SelectedObjectDimVals): {4, 9} → 2 classes
    assert ds.num_classes_per_dim == [2, 2, 2, 2]


def test_synthetic_corpus_session_filter_drops_other_sessions(
    synthetic_corpus: Path,
) -> None:
    """``session_filter`` excludes non-matching trials at construction."""
    ds = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
        session_filter="SessB_002_01",
    )
    assert len(ds) == 2
    assert all(n == "SessB_002_01" for n in ds.session_names)
    assert ds.session_ids.tolist() == [0, 0]


def test_synthetic_corpus_session_filter_can_be_list(synthetic_corpus: Path) -> None:
    """``session_filter`` accepts any iterable of session names."""
    ds = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
        session_filter=["SessA_001_01", "SessB_002_01"],
    )
    assert len(ds) == 6


def test_synthetic_corpus_unknown_session_filter_raises(
    synthetic_corpus: Path,
) -> None:
    """A ``session_filter`` matching no trials raises ``ValueError``."""
    with pytest.raises(ValueError, match="matched no trials"):
        MatFileTrialDataset(
            data_dir=synthetic_corpus / "data",
            target_dir=synthetic_corpus / "target",
            data_width=50,
            window_stride=25,
            session_filter="MissingSession",
        )


def test_synthetic_corpus_getitem_shapes(synthetic_corpus: Path) -> None:
    """``__getitem__`` returns ``(W, T, A, C)`` float32 + ``(num_dims,)`` int64."""
    ds = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
    )
    x, y, meta = ds[0]
    # NumSamples=200, width=50, stride=25 → starts 0,25,...,150 → 7 windows.
    assert x.shape == (7, 50, 2, 4)
    assert x.dtype == torch.float32
    assert y.shape == (4,)
    assert y.dtype == torch.int64
    assert set(meta) == {"session_id", "session_name", "trial_id"}
    assert meta["trial_id"] == 1


def test_synthetic_corpus_preserves_nan(synthetic_corpus: Path) -> None:
    """NaN positions from the source ``Data`` survive into the trial tensor.

    Critical Note #38: the encoder input zeros NaNs, but the dataset
    itself must hand them through so masks computed downstream stay
    correct.
    """
    ds = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
    )
    x, _, _ = ds[0]
    # Channel 0 of area 0 was NaN for the whole trial.
    assert torch.isnan(x[:, :, 0, 0]).all()
    # Channel 1 of area 0 has data, not NaN.
    assert not torch.isnan(x[:, :, 0, 1]).any()


def test_correct_trial_target_returns_shape_one(synthetic_corpus: Path) -> None:
    """1-D ``CorrectTrial`` targets must NOT be squeezed to a scalar.

    The MATLAB CM-table aggregator did a naked ``squeeze()`` on its
    ``(NumTrials, NumDims)`` table and broke when ``NumDims == 1``
    (e.g. ``CorrectTrial``). Python preserves the singleton dim
    explicitly — this test pins that contract.
    """
    ds = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
        target_type="CorrectTrial",
    )
    assert ds.num_dimensions == 1
    assert ds.num_classes_per_dim == [2]  # True / False both observed
    _, y, _ = ds[0]
    assert y.shape == (1,)
    # Trial 1 was set with correct_trial=True → raw 1, mapped to class 1.
    assert int(y.item()) == 1


def test_outcome_alias_resolves_to_correct_trial(synthetic_corpus: Path) -> None:
    """``target_type='Outcome'`` and ``'Trial Outcome'`` are aliases of CorrectTrial."""
    for alias in ("Outcome", "Trial Outcome"):
        ds = MatFileTrialDataset(
            data_dir=synthetic_corpus / "data",
            target_dir=synthetic_corpus / "target",
            data_width=50,
            window_stride=25,
            target_type=alias,
        )
        assert ds.num_dimensions == 1


def test_scalar_targets_return_shape_one(synthetic_corpus: Path) -> None:
    """``Dimensionality`` / ``Gain`` / ``Loss`` / ``DataNumber`` all return ``(1,)``."""
    for target_type in ("Dimensionality", "Gain", "Loss", "DataNumber"):
        ds = MatFileTrialDataset(
            data_dir=synthetic_corpus / "data",
            target_dir=synthetic_corpus / "target",
            data_width=50,
            window_stride=25,
            target_type=target_type,
        )
        assert ds.num_dimensions == 1, target_type
        _, y, _ = ds[0]
        assert y.shape == (1,), target_type


def test_unknown_target_type_raises(synthetic_corpus: Path) -> None:
    """An unsupported ``target_type`` produces a clear error."""
    with pytest.raises(ValueError, match="Unsupported target_type"):
        MatFileTrialDataset(
            data_dir=synthetic_corpus / "data",
            target_dir=synthetic_corpus / "target",
            data_width=50,
            window_stride=25,
            target_type="NotAThing",
        )


def test_class_mapping_can_be_pre_built(synthetic_corpus: Path) -> None:
    """An external ``class_mapping_per_dim`` is honored verbatim."""
    # Pre-build a mapping that covers every observed value plus extras.
    mapping = [
        {0: 0, 5: 1, 99: 2},
        {1: 0, 2: 1, 99: 2},
        {2: 0, 7: 1, 99: 2},
        {4: 0, 9: 1, 99: 2},
    ]
    ds = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
        class_mapping_per_dim=mapping,
    )
    assert ds.num_classes_per_dim == [3, 3, 3, 3]
    # Trial 1 had raw dim-0 value 0 → class 0; dim-3 (FeatureDim index 4) = 4 → 0.
    _, y, _ = ds[0]
    assert y.tolist() == [0, 0, 0, 0]


def test_class_mapping_rejects_unknown_value(synthetic_corpus: Path) -> None:
    """A mapping that lacks an observed value raises with a clear message."""
    mapping = [{0: 0}, {1: 0}, {2: 0}, {4: 0}]  # No entry for trial-2's '5' in dim 0.
    with pytest.raises(KeyError, match="not in the provided class_mapping_per_dim"):
        MatFileTrialDataset(
            data_dir=synthetic_corpus / "data",
            target_dir=synthetic_corpus / "target",
            data_width=50,
            window_stride=25,
            class_mapping_per_dim=mapping,
        )


def test_class_mapping_arity_mismatch_raises(synthetic_corpus: Path) -> None:
    """Wrong-length ``class_mapping_per_dim`` is rejected at construction."""
    with pytest.raises(ValueError, match="does not match target arity"):
        MatFileTrialDataset(
            data_dir=synthetic_corpus / "data",
            target_dir=synthetic_corpus / "target",
            data_width=50,
            window_stride=25,
            class_mapping_per_dim=[{0: 0, 5: 1}],  # only 1 dim, dataset has 4
        )


def test_unpaired_files_raise(tmp_path: Path) -> None:
    """An orphan Decision_Data file (no matching Target) is rejected."""
    data_dir = tmp_path / "data"
    target_dir = tmp_path / "target"
    data_dir.mkdir()
    target_dir.mkdir()
    _save_synthetic_trial(
        data_dir, target_dir, trial_id=1, session_name="X", dim_vals=[0, 1, 2, 3, 4]
    )
    # Add an orphan data file.
    sio.savemat(
        str(data_dir / "Decision_Data_0000099.mat"),
        {"Data": np.zeros((2, 50, 2))},
    )
    with pytest.raises(ValueError, match="Unpaired trial files"):
        MatFileTrialDataset(
            data_dir=data_dir,
            target_dir=target_dir,
            data_width=25,
            window_stride=25,
        )


def test_preload_caches_windowed_arrays(synthetic_corpus: Path) -> None:
    """``preload=True`` returns identical content to lazy loading."""
    lazy = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
        preload=False,
    )
    eager = MatFileTrialDataset(
        data_dir=synthetic_corpus / "data",
        target_dir=synthetic_corpus / "target",
        data_width=50,
        window_stride=25,
        preload=True,
    )
    for i in range(len(lazy)):
        x_lazy, y_lazy, _ = lazy[i]
        x_eager, y_eager, _ = eager[i]
        # NaN-aware equality
        torch.testing.assert_close(
            torch.nan_to_num(x_lazy), torch.nan_to_num(x_eager)
        )
        assert torch.equal(y_lazy, y_eager)


def test_co_located_dir_works_with_default_patterns(synthetic_corpus: Path) -> None:
    """When data and target share a directory, the default globs still pair correctly."""
    combined = synthetic_corpus / "combined"
    combined.mkdir()
    for src in (synthetic_corpus / "data").iterdir():
        (combined / src.name).symlink_to(src)
    for src in (synthetic_corpus / "target").iterdir():
        (combined / src.name).symlink_to(src)
    ds = MatFileTrialDataset(
        data_dir=combined, target_dir=combined, data_width=50, window_stride=25
    )
    assert len(ds) == 6


# ----------------------------------------------------------------------
# Real-fixture sanity (skip if the sample .mat pair is missing)
# ----------------------------------------------------------------------


pytestmark_real = pytest.mark.skipif(
    not (REAL_DATA_PATH.is_file() and REAL_TARGET_PATH.is_file()),
    reason="Sample Decision_Data_0000011.mat fixture not present in results/Decision",
)


@pytestmark_real
def test_real_fixture_windowing_matches_raw_indexing() -> None:
    """End-to-end load of the real fixture matches direct ``Data[c, s+t, a]`` indexing."""
    ds = MatFileTrialDataset(
        data_dir=REAL_FIXTURE_DIR,
        target_dir=REAL_FIXTURE_DIR,
        data_width=100,
        window_stride=50,
    )
    assert len(ds) == 1
    x, _, meta = ds[0]
    # (58, 3001, 6) with width 100, stride 50 starting at 0:
    # → 59 windows × 100 samples × 6 areas × 58 channels.
    assert x.shape == (59, 100, 6, 58)
    assert meta["trial_id"] == 11
    assert meta["session_name"] == "Wo_Probe_01_23_02_13_003_01"

    raw = np.asarray(load_mat(REAL_DATA_PATH)["Data"])
    # NaN-aware comparison on a sample of cells. The dataset returns
    # float32 while the underlying .mat is float64 — use float32 epsilon.
    for w, s in ((0, 0), (1, 50), (58, 2900)):
        for c, t, a in ((0, 0, 0), (57, 99, 5), (29, 50, 3)):
            lhs = x[w, t, a, c].item()
            rhs = float(raw[c, s + t, a])
            if np.isnan(rhs):
                assert np.isnan(lhs)
            else:
                assert abs(lhs - rhs) < 1e-5 * max(1.0, abs(rhs))


@pytestmark_real
def test_real_fixture_dimension_target_matches_matlab() -> None:
    """``SelectedObjectDimVals(FeatureDimensions=[1,2,3,5])`` = ``[0, 0, 8, 5]``.

    Verified independently via MATLAB MCP on
    ``Decision_Data_0000011.mat``. With a 1-trial dataset every raw
    value collapses to class 0 — the test checks the class-mapping
    construction reflects that.
    """
    ds = MatFileTrialDataset(
        data_dir=REAL_FIXTURE_DIR,
        target_dir=REAL_FIXTURE_DIR,
        data_width=100,
        window_stride=50,
        target_type="Dimension",
    )
    _, y, _ = ds[0]
    assert y.shape == (4,)
    assert y.tolist() == [0, 0, 0, 0]
    # Raw → class maps mirror the trial's raw values.
    assert ds.class_mapping_per_dim == [{0: 0}, {0: 0}, {8: 0}, {5: 0}]
