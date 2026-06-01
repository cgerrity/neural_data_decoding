"""Tests for hardware-aware gradient accumulation (Milestone C #9).

Pure-function tests for :mod:`neural_data_decoding.training.accumulation`
helpers, plus end-to-end equivalence tests showing that splitting a
mini-batch into micro-batches produces the same gradient direction
(and a comparable training trajectory) as a single full-batch pass.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader

from neural_data_decoding.data.dataset import (
    SyntheticTrialDataset,
    collate_trials,
)
from neural_data_decoding.models.classifier import MultiHeadClassifier
from neural_data_decoding.training.accumulation import (
    get_accumulation_size_for_current_system,
    micro_batch_chunks,
)
from neural_data_decoding.training.loop import train_one_epoch


# ───────────────────────── micro_batch_chunks ─────────────────────────


def test_single_chunk_fast_path_when_max_size_none() -> None:
    """When max_size is None, yield one chunk covering the whole batch."""
    assert list(micro_batch_chunks(8, None)) == [(0, 8, 1.0)]


def test_single_chunk_fast_path_when_max_size_larger_than_batch() -> None:
    """max_size >= batch_size → no accumulation needed."""
    assert list(micro_batch_chunks(8, 100)) == [(0, 8, 1.0)]


def test_even_split_into_micro_batches() -> None:
    """batch_size=8, max_size=4 → two chunks of size 4, weight 0.5 each."""
    chunks = list(micro_batch_chunks(8, 4))
    assert chunks == [(0, 4, 0.5), (4, 8, 0.5)]


def test_uneven_last_chunk() -> None:
    """batch_size=7, max_size=3 → chunks of 3, 3, 1; weights match fractions."""
    chunks = list(micro_batch_chunks(7, 3))
    assert chunks == [
        (0, 3, 3 / 7),
        (3, 6, 3 / 7),
        (6, 7, 1 / 7),
    ]
    assert sum(w for _, _, w in chunks) == pytest.approx(1.0, abs=1e-12)


def test_empty_batch_yields_nothing() -> None:
    """An empty batch (n_total=0) yields zero chunks."""
    assert list(micro_batch_chunks(0, 4)) == []


def test_chunks_partition_the_full_range() -> None:
    """For any (n, m), the chunks cover [0, n) exactly once."""
    for n_total in [1, 5, 13, 100]:
        for max_size in [None, 1, 3, 7, n_total, n_total + 5]:
            chunks = list(micro_batch_chunks(n_total, max_size))
            covered = sum(end - start for start, end, _ in chunks)
            assert covered == n_total
            # Weights sum to 1 (within FP).
            assert sum(w for _, _, w in chunks) == pytest.approx(1.0, abs=1e-12)


# ───────────────────────── get_accumulation_size_for_current_system ─────────────────────────


def test_returns_none_for_empty_table() -> None:
    """Empty config → no accumulation."""
    assert get_accumulation_size_for_current_system({}) is None


def test_returns_cpu_entry_when_no_cuda() -> None:
    """No CUDA → look up 'CPU' entry."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        size = get_accumulation_size_for_current_system({"CPU": 50, "GPU_X": 20})
    assert size == 50


def test_returns_none_when_no_matching_device() -> None:
    """Devices detected but none in the table → None (caller falls back to no-accum)."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        size = get_accumulation_size_for_current_system({"NVIDIA RTX A4000": 20})
    assert size is None


def test_returns_min_across_detected_gpus() -> None:
    """Multiple GPUs in the table → min of their max-sizes (the bottleneck)."""
    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.device_count", return_value=2), \
         mock.patch(
             "torch.cuda.get_device_name",
             side_effect=["NVIDIA GPU X", "NVIDIA GPU Y"],
         ):
        size = get_accumulation_size_for_current_system({
            "NVIDIA GPU X": 64,
            "NVIDIA GPU Y": 32,
            "CPU": 100,
        })
    assert size == 32


# ───────────────────────── End-to-end: accumulation produces equivalent gradients ─────────────────────────


def _make_setup(seed: int = 0):
    """Tiny synthetic loader + model + AdamW optimizer for repeatable tests."""
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=8,
        num_samples=3, num_features=4, num_classes_per_dim=[3],
        signal_strength=2.0, seed=seed,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_trials)
    torch.manual_seed(seed)
    model = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)  # lr=0 so weights don't drift
    return loader, model, optimizer


def test_full_batch_equals_micro_batch_gradient() -> None:
    """A single forward+backward on the full batch produces the same gradient
    direction as splitting into micro-batches (up to FP)."""
    # Setup A: no accumulation (full batch in one pass).
    loader_a, model_a, opt_a = _make_setup(seed=0)
    train_one_epoch(
        model=model_a, dataloader=loader_a, optimizer=opt_a,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        accumulation_max_size=None,
    )
    grads_full = [p.grad.detach().clone() for p in model_a.parameters() if p.grad is not None]

    # Setup B: micro-batch accumulation (split into 4 chunks of 2).
    loader_b, model_b, opt_b = _make_setup(seed=0)
    train_one_epoch(
        model=model_b, dataloader=loader_b, optimizer=opt_b,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        accumulation_max_size=2,
    )
    grads_micro = [p.grad.detach().clone() for p in model_b.parameters() if p.grad is not None]

    # The gradients must be (approximately) identical.
    assert len(grads_full) == len(grads_micro)
    for gf, gm in zip(grads_full, grads_micro):
        torch.testing.assert_close(gf, gm, rtol=1e-5, atol=1e-6)


def test_accumulation_size_above_batch_is_no_op() -> None:
    """When accumulation_max_size >= batch_size, behavior is bit-identical
    to the no-accumulation case."""
    loader_a, model_a, opt_a = _make_setup(seed=0)
    train_one_epoch(
        model=model_a, dataloader=loader_a, optimizer=opt_a,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        accumulation_max_size=None,
    )
    grads_a = [p.grad.detach().clone() for p in model_a.parameters() if p.grad is not None]

    loader_b, model_b, opt_b = _make_setup(seed=0)
    train_one_epoch(
        model=model_b, dataloader=loader_b, optimizer=opt_b,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        accumulation_max_size=100,  # >> batch_size of 8 → single chunk
    )
    grads_b = [p.grad.detach().clone() for p in model_b.parameters() if p.grad is not None]

    for ga, gb in zip(grads_a, grads_b):
        # Bit-identical: same code path internally (single-chunk fast path).
        assert torch.equal(ga, gb)


def test_accumulation_step_count_equals_full_batch_step_count() -> None:
    """Same number of optimizer.step() calls regardless of micro-batch count.

    This is the key invariant — accumulation doesn't change the number of
    weight updates per epoch.
    """
    loader, model, opt = _make_setup(seed=0)

    step_count = 0
    original_step = opt.step

    def counting_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    opt.step = counting_step  # type: ignore[method-assign]

    train_one_epoch(
        model=model, dataloader=loader, optimizer=opt,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        accumulation_max_size=2,  # 4 micro-batches per mini-batch
    )
    # Loader yields 1 mini-batch of 8 trials → 1 optimizer.step() despite 4
    # micro-batch forward+backward passes.
    assert step_count == 1
