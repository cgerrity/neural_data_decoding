"""Tests for :mod:`neural_data_decoding.data.samplers`.

The key parity contract from Critical Note #9 is verified here: every
emitted minibatch contains trials from exactly one session.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_data_decoding.data.samplers import SingleSessionBatchSampler


@pytest.fixture()
def three_session_ids() -> np.ndarray:
    """Per-trial session IDs for a 30-trial / 3-session toy dataset."""
    return np.repeat([1, 2, 3], 10)


# ───────────────────────── Constructor validation ─────────────────────────


def test_zero_batch_size_rejected() -> None:
    """``batch_size=0`` is a programming error."""
    with pytest.raises(ValueError, match="batch_size"):
        SingleSessionBatchSampler(session_ids=[1, 1], batch_size=0)


def test_empty_session_ids_rejected() -> None:
    """An empty dataset has nothing to sample from — error early."""
    with pytest.raises(ValueError, match="non-empty"):
        SingleSessionBatchSampler(session_ids=[], batch_size=4)


def test_non_1d_session_ids_rejected() -> None:
    """A 2-D session-id array is malformed input."""
    bad = np.zeros((3, 3))
    with pytest.raises(ValueError, match="1-D"):
        SingleSessionBatchSampler(session_ids=bad, batch_size=2)


# ───────────────────────── Core contract: single-session batches ─────────────────────────


def test_every_batch_is_single_session(three_session_ids: np.ndarray) -> None:
    """The critical Note-#9 invariant: every emitted batch is one-session."""
    sampler = SingleSessionBatchSampler(
        session_ids=three_session_ids, batch_size=4, drop_last=False, seed=0
    )
    for batch in sampler:
        sessions_in_batch = {three_session_ids[i] for i in batch}
        assert len(sessions_in_batch) == 1, (
            f"Batch {batch} mixes sessions {sessions_in_batch}"
        )


def test_every_index_appears_exactly_once_when_drop_last_false(
    three_session_ids: np.ndarray,
) -> None:
    """With ``drop_last=False`` every trial index is yielded exactly once."""
    sampler = SingleSessionBatchSampler(
        session_ids=three_session_ids, batch_size=3, drop_last=False, seed=0
    )
    seen: list[int] = []
    for batch in sampler:
        seen.extend(batch)

    assert sorted(seen) == list(range(len(three_session_ids)))


def test_drop_last_discards_partial_batches(three_session_ids: np.ndarray) -> None:
    """With ``drop_last=True`` only full-size batches are yielded."""
    sampler = SingleSessionBatchSampler(
        session_ids=three_session_ids, batch_size=3, drop_last=True, seed=0
    )
    for batch in sampler:
        assert len(batch) == 3
    # 10 trials / 3 per batch = 3 full + 1 partial. With drop_last, only
    # 3 full per session × 3 sessions = 9 batches.
    assert len(list(sampler)) == 9


def test_length_matches_iter_count(three_session_ids: np.ndarray) -> None:
    """``__len__`` must agree with the actual number of yielded batches."""
    for drop_last in (False, True):
        sampler = SingleSessionBatchSampler(
            session_ids=three_session_ids,
            batch_size=4,
            drop_last=drop_last,
            seed=0,
        )
        assert len(sampler) == len(list(sampler))


# ───────────────────────── Determinism ─────────────────────────


def test_same_seed_and_epoch_produces_identical_order(
    three_session_ids: np.ndarray,
) -> None:
    """Two samplers with identical (seed, epoch) yield the same batches."""
    s1 = SingleSessionBatchSampler(
        session_ids=three_session_ids, batch_size=4, seed=42
    )
    s2 = SingleSessionBatchSampler(
        session_ids=three_session_ids, batch_size=4, seed=42
    )
    assert list(s1) == list(s2)


def test_set_epoch_changes_batch_order(three_session_ids: np.ndarray) -> None:
    """Advancing the epoch yields a different shuffle of the same trials."""
    sampler = SingleSessionBatchSampler(
        session_ids=three_session_ids, batch_size=4, seed=42
    )
    epoch0 = list(sampler)

    sampler.set_epoch(1)
    epoch1 = list(sampler)

    # Different ordering is overwhelmingly likely with 30 trials.
    assert epoch0 != epoch1


# ───────────────────────── Edge cases ─────────────────────────


def test_single_session_works() -> None:
    """A dataset with only one session still partitions into batches correctly."""
    sampler = SingleSessionBatchSampler(
        session_ids=np.zeros(10, dtype=int), batch_size=3, drop_last=False
    )
    batches = list(sampler)
    # 10 / 3 = 3 full + 1 partial.
    assert len(batches) == 4
    assert sum(len(b) for b in batches) == 10


def test_batch_size_larger_than_session_with_drop_last_yields_nothing() -> None:
    """When ``drop_last=True`` and no session has ``batch_size`` trials, yield 0."""
    sampler = SingleSessionBatchSampler(
        session_ids=[1, 1, 2, 2, 3, 3],
        batch_size=5,
        drop_last=True,
    )
    assert len(sampler) == 0
    assert list(sampler) == []


def test_string_session_ids_supported() -> None:
    """Session IDs can be strings, not just integers."""
    session_ids = np.array(["sess_A", "sess_A", "sess_B", "sess_B"])
    sampler = SingleSessionBatchSampler(session_ids=session_ids, batch_size=2)
    for batch in sampler:
        sessions = {session_ids[i] for i in batch}
        assert len(sessions) == 1
