"""Tests for :mod:`neural_data_decoding.data.dataset`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from neural_data_decoding.data.dataset import (
    SyntheticTrialDataset,
    collate_trials,
)
from neural_data_decoding.data.samplers import SingleSessionBatchSampler


@pytest.fixture()
def small_dataset() -> SyntheticTrialDataset:
    """A small 2-session / 4-trial-each dataset for fast iteration."""
    return SyntheticTrialDataset(
        num_sessions=2,
        trials_per_session=4,
        num_samples=10,
        num_features=3,
        num_classes_per_dim=[3, 2],
        seed=0,
    )


# ───────────────────────── Constructor validation ─────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_sessions": 0},
        {"trials_per_session": 0},
        {"num_samples": 0},
        {"num_features": 0},
        {"num_classes_per_dim": []},
        {"num_classes_per_dim": [3, 0]},
    ],
)
def test_constructor_rejects_invalid_args(kwargs: dict) -> None:
    """Each individual bad arg produces a clear ``ValueError``."""
    defaults = {
        "num_sessions": 2,
        "trials_per_session": 4,
        "num_samples": 10,
        "num_features": 3,
        "num_classes_per_dim": [3],
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        SyntheticTrialDataset(**defaults)


# ───────────────────────── Shape & dtype ─────────────────────────


def test_len_matches_session_count(small_dataset: SyntheticTrialDataset) -> None:
    """Total length = sessions × trials_per_session."""
    assert len(small_dataset) == 8


def test_getitem_shapes_and_dtypes(small_dataset: SyntheticTrialDataset) -> None:
    """``__getitem__`` returns tensors with the documented shape & dtype."""
    x, t, meta = small_dataset[0]
    assert x.shape == (10, 3)
    assert x.dtype == torch.float32
    assert t.shape == (2,)
    assert t.dtype == torch.int64
    assert meta["session_id"] in (0, 1)
    assert meta["trial_id"] == 0


def test_session_ids_correctly_repeated() -> None:
    """``session_ids`` is a length-N array with each session ID appearing
    ``trials_per_session`` times."""
    ds = SyntheticTrialDataset(
        num_sessions=3,
        trials_per_session=5,
        num_samples=4,
        num_features=2,
        num_classes_per_dim=[2],
    )
    np.testing.assert_array_equal(ds.session_ids, np.repeat([0, 1, 2], 5))


def test_index_out_of_range_raises(small_dataset: SyntheticTrialDataset) -> None:
    """Negative or too-large indices raise ``IndexError``."""
    with pytest.raises(IndexError):
        small_dataset[len(small_dataset)]
    with pytest.raises(IndexError):
        small_dataset[-1]


# ───────────────────────── Determinism ─────────────────────────


def test_same_seed_produces_identical_data() -> None:
    """Two datasets constructed with the same seed produce identical samples."""
    a = SyntheticTrialDataset(
        num_sessions=2, trials_per_session=3, num_samples=5,
        num_features=2, num_classes_per_dim=[2], seed=7,
    )
    b = SyntheticTrialDataset(
        num_sessions=2, trials_per_session=3, num_samples=5,
        num_features=2, num_classes_per_dim=[2], seed=7,
    )
    for i in range(len(a)):
        xa, ta, _ = a[i]
        xb, tb, _ = b[i]
        torch.testing.assert_close(xa, xb)
        torch.testing.assert_close(ta, tb)


# ───────────────────────── Signal causality (the tracer-bullet contract) ─────────────────────────


def test_label_signal_is_recoverable() -> None:
    """A trivial linear probe can recover labels at well-above-chance rate.

    This is the property that makes the synthetic dataset useful for the
    Logistic Regression tracer bullet: if a from-scratch linear classifier
    can't beat chance here, the dataset isn't doing its job.
    """
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=200,
        num_samples=1, num_features=8,
        num_classes_per_dim=[4],
        signal_strength=2.0, noise_std=0.2, seed=0,
    )
    # Use the per-class signal directly as a "classifier" (oracle linear probe).
    signal = ds._class_signals[0]  # (num_classes, num_features)
    correct = 0
    for trial_idx in range(len(ds)):
        x, t, _ = ds[trial_idx]
        scores = x.numpy().mean(axis=0) @ signal.T  # (num_classes,)
        pred = int(np.argmax(scores))
        correct += int(pred == int(t[0]))
    accuracy = correct / len(ds)
    assert accuracy > 0.5, f"Expected oracle probe > 50% accuracy; got {accuracy:.2f}"


# ───────────────────────── Collate + Sampler integration ─────────────────────────


def test_collate_trials_returns_batched_dict(
    small_dataset: SyntheticTrialDataset,
) -> None:
    """``collate_trials`` stacks features+targets and preserves metadata list."""
    batch = [small_dataset[i] for i in range(3)]
    collated = collate_trials(batch)
    assert collated["x"].shape == (3, 10, 3)
    assert collated["targets"].shape == (3, 2)
    assert len(collated["metadata"]) == 3
    assert all("session_id" in m for m in collated["metadata"])


def test_dataloader_with_single_session_sampler(
    small_dataset: SyntheticTrialDataset,
) -> None:
    """Plumbed end-to-end: DataLoader + SingleSessionBatchSampler + collate."""
    sampler = SingleSessionBatchSampler(
        session_ids=small_dataset.session_ids,
        batch_size=2,
        drop_last=False,
        seed=0,
    )
    loader = DataLoader(
        small_dataset, batch_sampler=sampler, collate_fn=collate_trials
    )
    for batch in loader:
        # Each batch must be single-session (the Critical Note #9 invariant).
        sessions = {m["session_id"] for m in batch["metadata"]}
        assert len(sessions) == 1
        assert batch["x"].ndim == 3
        assert batch["targets"].ndim == 2
