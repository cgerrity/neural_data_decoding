"""Tests for the Milestone A training engine: checkpoint + loop + lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from neural_data_decoding.data.dataset import (
    SyntheticTrialDataset,
    collate_trials,
)
from neural_data_decoding.models.classifier import MultiHeadClassifier
from neural_data_decoding.training.checkpoint import (
    CURRENT_CHECKPOINT_FILENAME,
    OPTIMAL_CHECKPOINT_FILENAME,
    has_existing_checkpoint,
    load_current_checkpoint,
    load_optimal_checkpoint,
    save_current_checkpoint,
    save_optimal_checkpoint,
)
from neural_data_decoding.training.lifecycle import fit_supervised
from neural_data_decoding.training.loop import train_one_epoch, validate


# ───────────────────────── checkpoint ─────────────────────────


def _toy_model() -> MultiHeadClassifier:
    return MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])


def test_save_and_load_current_checkpoint_roundtrips(tmp_path: Path) -> None:
    """Save → load restores the weights and bookkeeping fields exactly."""
    saved = _toy_model()
    for p in saved.parameters():
        torch.nn.init.normal_(p, std=0.7)

    save_current_checkpoint(
        tmp_path, model=saved, epoch=3, iteration=42, best_metric=0.81
    )
    loaded_model = _toy_model()
    state = load_current_checkpoint(tmp_path, model=loaded_model)

    assert state is not None
    assert state.epoch == 3
    assert state.iteration == 42
    assert state.best_metric == pytest.approx(0.81)
    for p_saved, p_loaded in zip(saved.parameters(), loaded_model.parameters()):
        assert torch.allclose(p_saved, p_loaded)


def test_load_current_checkpoint_returns_none_when_missing(tmp_path: Path) -> None:
    """A fresh directory yields ``None`` so fit_supervised starts at epoch 0."""
    assert load_current_checkpoint(tmp_path, model=_toy_model()) is None


def test_optimal_checkpoint_is_separate_from_current(tmp_path: Path) -> None:
    """Saving Optimal does not affect the Current snapshot or vice versa."""
    save_current_checkpoint(
        tmp_path, model=_toy_model(), epoch=1, iteration=1, best_metric=0.5
    )
    save_optimal_checkpoint(tmp_path, model=_toy_model(), epoch=1, metric=0.5)
    assert (tmp_path / CURRENT_CHECKPOINT_FILENAME).exists()
    assert (tmp_path / OPTIMAL_CHECKPOINT_FILENAME).exists()

    payload = load_optimal_checkpoint(tmp_path, model=_toy_model())
    assert payload is not None
    assert payload["epoch"] == 1
    assert payload["metric"] == pytest.approx(0.5)


def test_has_existing_checkpoint_detects_either_file(tmp_path: Path) -> None:
    """Pre-flight check must catch either Current or Optimal."""
    assert not has_existing_checkpoint(tmp_path)
    save_current_checkpoint(
        tmp_path, model=_toy_model(), epoch=0, iteration=0, best_metric=0.0
    )
    assert has_existing_checkpoint(tmp_path)


# ───────────────────────── loop ─────────────────────────


def _make_loader(seed: int = 0) -> DataLoader:
    """Small synthetic loader good enough to exercise the training kernel."""
    ds = SyntheticTrialDataset(
        num_sessions=1,
        trials_per_session=16,
        num_samples=5,
        num_features=4,
        num_classes_per_dim=[3],
        signal_strength=2.0,
        seed=seed,
    )
    return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_trials)


def test_train_one_epoch_reduces_loss() -> None:
    """A single epoch of AdamW should reduce loss on a learnable synthetic signal."""
    torch.manual_seed(0)
    loader = _make_loader()
    model = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    # Initial loss measured via validate (no gradient updates).
    initial = validate(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
    )
    trained, _ = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
    )
    final = validate(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
    )
    assert final.classification_loss < initial.classification_loss
    assert trained.num_iterations == 4  # 16 trials / batch_size 4


def test_validate_does_not_update_weights() -> None:
    """Validation passes must leave parameters byte-identical."""
    torch.manual_seed(0)
    loader = _make_loader()
    model = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    before = [p.detach().clone() for p in model.parameters()]
    validate(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
    )
    for b, p in zip(before, model.parameters()):
        assert torch.equal(b, p)


def test_train_one_epoch_grad_clip_limits_norm() -> None:
    """``grad_clip_norm`` must keep gradient L2 below the threshold."""
    torch.manual_seed(0)
    loader = _make_loader()
    model = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    # Inflate weights so loss is huge → without clipping, grad norm would explode.
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(50.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    clip = 0.5

    captured_norms: list[float] = []

    # Patch the step to record grad norm after clipping.
    original_step = optimizer.step

    def recording_step(*args, **kwargs):  # type: ignore[no-untyped-def]
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().pow(2).sum())
        captured_norms.append(total_sq ** 0.5)
        return original_step(*args, **kwargs)

    optimizer.step = recording_step  # type: ignore[method-assign]

    train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        grad_clip_norm=clip,
    )
    # Every measured norm must be at or below the clip (with tiny fp slack).
    assert captured_norms
    assert all(n <= clip + 1e-5 for n in captured_norms)


# ───────────────────────── lifecycle ─────────────────────────


def test_fit_supervised_runs_and_writes_checkpoints(tmp_path: Path) -> None:
    """A fresh fit writes Current every epoch and Optimal on best-val improvements."""
    torch.manual_seed(0)
    loader = _make_loader()
    model = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    history = fit_supervised(
        model=model,
        train_loader=loader,
        val_loader=loader,  # tiny synthetic — reuse loader for val
        optimizer=optimizer,
        num_epochs=3,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        checkpoint_dir=tmp_path,
    )

    assert len(history) == 3
    assert (tmp_path / CURRENT_CHECKPOINT_FILENAME).exists()
    assert (tmp_path / OPTIMAL_CHECKPOINT_FILENAME).exists()
    assert any(e.is_best for e in history)


def test_fit_supervised_resumes_from_current(tmp_path: Path) -> None:
    """Calling fit twice resumes from the Current snapshot — second call adds the gap."""
    torch.manual_seed(0)
    loader = _make_loader()
    model = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    first = fit_supervised(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        num_epochs=2,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        checkpoint_dir=tmp_path,
    )
    assert len(first) == 2

    # Fresh model + optimizer to simulate process restart.
    model2 = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.05)
    second = fit_supervised(
        model=model2,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer2,
        num_epochs=5,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        checkpoint_dir=tmp_path,
    )
    # Already ran 2 epochs (indices 0, 1); 3 more remain (indices 2, 3, 4).
    assert len(second) == 3
    assert [e.epoch for e in second] == [2, 3, 4]


def test_fit_supervised_callback_runs_every_epoch(tmp_path: Path) -> None:
    """``epoch_callback`` fires after each completed epoch with the history entry."""
    torch.manual_seed(0)
    loader = _make_loader()
    model = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    seen: list[int] = []
    fit_supervised(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        num_epochs=2,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        checkpoint_dir=tmp_path,
        epoch_callback=lambda h: seen.append(h.epoch),
    )
    assert seen == [0, 1]
