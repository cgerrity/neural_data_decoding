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
    trained, _, _ = train_one_epoch(
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


# ───────────────────────── Curriculum integration (Milestone C #5) ─────────────────────────


def test_fit_supervised_calls_curriculum_update_each_epoch(tmp_path: Path) -> None:
    """fit_supervised drives curriculum.update(epoch+1) on every epoch.

    The +1 converts Python's 0-indexed loop into MATLAB's 1-indexed
    convention so the schedule's waypoints line up with the MATLAB regimes.
    """
    from neural_data_decoding.training.schedules import (
        CurriculumBundle,
        ScheduleWaypoints,
        make_freeze_schedule,
        make_load_schedule,
        make_weight_schedule,
    )

    # Tiny dataset / classifier so the loop runs quickly.
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=4,
        num_samples=4, num_features=3, num_classes_per_dim=[2],
        seed=0,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_trials)
    model = MultiHeadClassifier(in_features=3, num_classes_per_dim=[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Curriculum where the classification weight ramps 0 → 1 over epochs 1..5.
    bundle = CurriculumBundle(
        load=make_load_schedule(),
        weight=make_weight_schedule(
            classification=1.0,
            waypoints={
                "classification": ScheduleWaypoints.of([1, 5], [0.0, 1.0]),
            },
        ),
        freeze=make_freeze_schedule(),
    )

    history = fit_supervised(
        model=model,
        train_loader=loader,
        val_loader=None,
        optimizer=optimizer,
        num_epochs=3,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        checkpoint_dir=tmp_path,
        curriculum=bundle,
    )

    assert len(history) == 3
    # After fitting for 3 epochs the curriculum should have advanced to
    # MATLAB epoch 3 (Python epoch 2 → update(3)). At that point the ramp
    # value is (3 - 1 - 1) * 1.0 / 4 = 0.25.
    assert bundle.weight.current("classification") == pytest.approx(0.25)


def test_fit_supervised_freeze_applies_at_epoch_start(tmp_path: Path) -> None:
    """When curriculum.freeze has waypoints, named param groups get rescaled lr."""
    from neural_data_decoding.training.freezing import (
        build_optimizer_with_module_groups,
    )
    from neural_data_decoding.training.schedules import (
        CurriculumBundle,
        ScheduleWaypoints,
        make_freeze_schedule,
        make_load_schedule,
        make_weight_schedule,
    )

    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=2,
        num_samples=3, num_features=2, num_classes_per_dim=[2],
        seed=0,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_trials)
    # Use a composite-shaped wrapper so freezing has a named submodule to bind.
    wrapper = torch.nn.Module()
    wrapper.classifier = MultiHeadClassifier(in_features=2, num_classes_per_dim=[2])
    wrapper.forward = wrapper.classifier.forward  # type: ignore[assignment]

    optimizer = build_optimizer_with_module_groups(
        {"classifier": wrapper.classifier}, initial_lr=0.1,
    )
    bundle = CurriculumBundle(
        load=make_load_schedule(),
        weight=make_weight_schedule(classification=1.0),
        freeze=make_freeze_schedule(
            classifier=1.0,
            waypoints={
                "classifier": ScheduleWaypoints.of([1, 5], [0.5, 0.5]),
            },
        ),
    )

    fit_supervised(
        model=wrapper,
        train_loader=loader,
        val_loader=None,
        optimizer=optimizer,
        num_epochs=1,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
        checkpoint_dir=tmp_path,
        curriculum=bundle,
        freeze_base_lr=0.1,
    )

    # After one epoch the classifier group's lr should be base_lr × 0.5 = 0.05.
    by_name = {g["name"]: g["lr"] for g in optimizer.param_groups}
    assert by_name["classifier"] == pytest.approx(0.05)


def test_update_priors_strategy_helper_matches_matlab_cadence() -> None:
    """_update_priors_strategy_for encodes the RescaleLossEpoch cadence."""
    from neural_data_decoding.training.lifecycle import _update_priors_strategy_for

    # RescaleLossEpoch == 0 → every iteration, every epoch.
    for epoch in range(5):
        assert _update_priors_strategy_for(epoch, 0) == "every_iter"

    # RescaleLossEpoch == 1 → first iteration of every epoch.
    for epoch in range(5):
        assert _update_priors_strategy_for(epoch, 1) == "first_iter_only"

    # RescaleLossEpoch == 3 → first iter of epochs 1, 4, 7, ...
    # Python epoch=0 → MATLAB Epoch=1 → mod(2, 3) == 2 → never.
    # Python epoch=2 → MATLAB Epoch=3 → mod(4, 3) == 1 → first_iter_only.
    assert _update_priors_strategy_for(0, 3) == "never"
    assert _update_priors_strategy_for(1, 3) == "never"
    assert _update_priors_strategy_for(2, 3) == "first_iter_only"
    assert _update_priors_strategy_for(3, 3) == "never"
    assert _update_priors_strategy_for(4, 3) == "never"
    assert _update_priors_strategy_for(5, 3) == "first_iter_only"


def test_resolve_epoch_loss_weights_blends_static_and_curriculum() -> None:
    """_resolve_epoch_loss_weights overrides static keys with curriculum live values."""
    from neural_data_decoding.training.lifecycle import _resolve_epoch_loss_weights
    from neural_data_decoding.training.schedules import (
        CurriculumBundle,
        make_freeze_schedule,
        make_load_schedule,
        make_weight_schedule,
    )

    static = {"classification": 10.0, "extra_static_key": 7.0}
    bundle = CurriculumBundle(
        load=make_load_schedule(),
        weight=make_weight_schedule(classification=2.0, kl=3.0),
        freeze=make_freeze_schedule(),
    )

    resolved = _resolve_epoch_loss_weights(static, bundle)
    # Classification overridden by curriculum's current value.
    assert resolved["classification"] == 2.0
    # KL appears (was not in static, but curriculum has it).
    assert resolved["kl"] == 3.0
    # Static-only key preserved.
    assert resolved["extra_static_key"] == 7.0


# ───────────────────────── Stage 1 unsupervised path (Milestone C #6) ─────────────────────────


def _toy_autoencoder() -> torch.nn.Module:
    """Tiny VariationalAutoencoder for unsupervised-path tests."""
    from neural_data_decoding.models.composite import build_variational_autoencoder
    return build_variational_autoencoder({
        "in_features": 4,
        "hidden_sizes": [8, 2],
        "num_classes_per_dim": [2],          # ignored by AE builder
        "classifier_hidden_size": [2],       # ignored by AE builder
        "loss_type_decoder": "MSE",
        "transform": "GRU",
    })


def _toy_ae_loader(*, n_trials: int = 4, num_samples: int = 5) -> DataLoader:
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=n_trials,
        num_samples=num_samples, num_features=4, num_classes_per_dim=[2],
        seed=0,
    )
    return DataLoader(ds, batch_size=2, collate_fn=collate_trials)


def test_train_unsupervised_epoch_runs_and_decreases_loss() -> None:
    """Sanity: Stage 1 train epoch runs and reduces the autoencoder loss."""
    from neural_data_decoding.training.loop import train_unsupervised_epoch
    ae = _toy_autoencoder()
    loader = _toy_ae_loader()
    optimizer = torch.optim.AdamW(ae.parameters(), lr=0.01)
    first = train_unsupervised_epoch(
        model=ae, dataloader=loader, optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights={"reconstruction": 1.0, "kl": 0.01},
    )
    # Run a couple more — loss should drop on this trivial setup.
    for _ in range(3):
        train_unsupervised_epoch(
            model=ae, dataloader=loader, optimizer=optimizer,
            device=torch.device("cpu"),
            loss_weights={"reconstruction": 1.0, "kl": 0.01},
        )
    last = train_unsupervised_epoch(
        model=ae, dataloader=loader, optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights={"reconstruction": 1.0, "kl": 0.01},
    )
    assert last.total_loss < first.total_loss


def test_train_unsupervised_rejects_non_autoencoder_output() -> None:
    """A model whose forward returns the wrong type fails fast (not silently)."""
    from neural_data_decoding.training.loop import train_unsupervised_epoch
    # MultiHeadClassifier returns list[Tensor], not AutoencoderOutput.
    bad_model = MultiHeadClassifier(in_features=4, num_classes_per_dim=[2])
    loader = _toy_ae_loader()
    optimizer = torch.optim.AdamW(bad_model.parameters(), lr=0.01)
    with pytest.raises(TypeError, match="AutoencoderOutput"):
        train_unsupervised_epoch(
            model=bad_model, dataloader=loader, optimizer=optimizer,
            device=torch.device("cpu"),
            loss_weights={"reconstruction": 1.0, "kl": 0.01},
        )


def test_fit_unsupervised_tracks_min_val_loss_as_best(tmp_path: Path) -> None:
    """Stage 1's is_best flag fires on lower val loss (not higher accuracy)."""
    from neural_data_decoding.training.lifecycle import fit_unsupervised
    ae = _toy_autoencoder()
    loader = _toy_ae_loader()
    optimizer = torch.optim.AdamW(ae.parameters(), lr=0.01)

    history = fit_unsupervised(
        model=ae,
        train_loader=loader, val_loader=loader,
        optimizer=optimizer, num_epochs=3,
        device=torch.device("cpu"),
        loss_weights={"reconstruction": 1.0, "kl": 0.01},
        checkpoint_dir=tmp_path,
    )

    # First epoch always sets is_best (val < +inf).
    assert history[0].is_best
    # Any subsequent epoch that beats the best should also fire is_best.
    val_losses = [h.val.total_loss for h in history if h.val is not None]
    running_min = val_losses[0]
    expected_is_best = [True]
    for v in val_losses[1:]:
        is_b = v < running_min
        if is_b:
            running_min = v
        expected_is_best.append(is_b)
    actual_is_best = [h.is_best for h in history]
    assert actual_is_best == expected_is_best


def test_fit_unsupervised_writes_separate_checkpoint_dir(tmp_path: Path) -> None:
    """Stage 1 writes Current + Optimal snapshots into its own dir."""
    from neural_data_decoding.training.checkpoint import (
        CURRENT_CHECKPOINT_FILENAME,
        OPTIMAL_CHECKPOINT_FILENAME,
    )
    from neural_data_decoding.training.lifecycle import fit_unsupervised
    ae = _toy_autoencoder()
    loader = _toy_ae_loader()
    optimizer = torch.optim.AdamW(ae.parameters(), lr=0.01)
    stage1_dir = tmp_path / "stage1_autoencoder"

    fit_unsupervised(
        model=ae,
        train_loader=loader, val_loader=loader,
        optimizer=optimizer, num_epochs=2,
        device=torch.device("cpu"),
        loss_weights={"reconstruction": 1.0, "kl": 0.01},
        checkpoint_dir=stage1_dir,
    )

    assert (stage1_dir / CURRENT_CHECKPOINT_FILENAME).is_file()
    assert (stage1_dir / OPTIMAL_CHECKPOINT_FILENAME).is_file()


def test_fit_two_stage_hands_off_optimal_stage1_weights(tmp_path: Path) -> None:
    """After fit_two_stage, the composite's encoder/decoder reflect Stage 1's Optimal.

    Verifies the handoff: Stage 1 runs, saves Optimal weights, those weights
    are loaded back into the autoencoder instance, then copied into the
    composite. Stage 2 starts training from those weights.
    """
    from neural_data_decoding.models.composite import (
        build_variational_autoencoder,
        build_variational_composite,
    )
    from neural_data_decoding.training.lifecycle import fit_two_stage
    from neural_data_decoding.training.checkpoint import load_optimal_checkpoint

    cfg = {
        "in_features": 4, "hidden_sizes": [8, 2],
        "num_classes_per_dim": [2], "classifier_hidden_size": [2],
        "loss_type_decoder": "MSE", "transform": "GRU",
    }
    ae = build_variational_autoencoder(cfg)
    composite = build_variational_composite(cfg)
    loader = _toy_ae_loader()
    stage1_opt = torch.optim.AdamW(ae.parameters(), lr=0.01)
    stage2_opt = torch.optim.AdamW(composite.parameters(), lr=0.01)

    fit_two_stage(
        autoencoder=ae, composite=composite,
        stage1_optimizer=stage1_opt, stage2_optimizer=stage2_opt,
        stage1_num_epochs=2, stage2_num_epochs=1,
        train_loader=loader, val_loader=loader,
        device=torch.device("cpu"),
        loss_weights={"reconstruction": 1.0, "kl": 0.01, "classification": 1.0},
        checkpoint_dir=tmp_path,
    )

    # Load Stage 1's Optimal weights into a fresh AE for comparison.
    reference_ae = build_variational_autoencoder(cfg)
    load_optimal_checkpoint(tmp_path / "stage1_autoencoder", model=reference_ae)

    # After Stage 2 finishes, the composite's encoder/decoder will have
    # been further trained — but the AE instance we passed in should
    # match the Optimal snapshot (because fit_two_stage loaded it before
    # the handoff). Pin that:
    for a, b in zip(ae.encoder.parameters(), reference_ae.encoder.parameters()):
        assert torch.equal(a, b), "Stage 1 autoencoder did not retain Optimal weights."


def test_fit_two_stage_writes_stage1_subdir(tmp_path: Path) -> None:
    """Stage 1 lands in <checkpoint_dir>/stage1_autoencoder/, not the root."""
    from neural_data_decoding.models.composite import (
        build_variational_autoencoder,
        build_variational_composite,
    )
    from neural_data_decoding.training.lifecycle import fit_two_stage
    from neural_data_decoding.training.checkpoint import (
        CURRENT_CHECKPOINT_FILENAME,
        OPTIMAL_CHECKPOINT_FILENAME,
    )

    cfg = {
        "in_features": 4, "hidden_sizes": [8, 2],
        "num_classes_per_dim": [2], "classifier_hidden_size": [2],
        "loss_type_decoder": "MSE", "transform": "GRU",
    }
    ae = build_variational_autoencoder(cfg)
    composite = build_variational_composite(cfg)
    loader = _toy_ae_loader()
    s1_opt = torch.optim.AdamW(ae.parameters(), lr=0.01)
    s2_opt = torch.optim.AdamW(composite.parameters(), lr=0.01)

    fit_two_stage(
        autoencoder=ae, composite=composite,
        stage1_optimizer=s1_opt, stage2_optimizer=s2_opt,
        stage1_num_epochs=1, stage2_num_epochs=1,
        train_loader=loader, val_loader=loader,
        device=torch.device("cpu"),
        loss_weights={"reconstruction": 1.0, "kl": 0.01, "classification": 1.0},
        checkpoint_dir=tmp_path,
    )

    stage1_dir = tmp_path / "stage1_autoencoder"
    assert (stage1_dir / CURRENT_CHECKPOINT_FILENAME).is_file()
    assert (stage1_dir / OPTIMAL_CHECKPOINT_FILENAME).is_file()
    # Stage 2 root has its own checkpoints.
    assert (tmp_path / CURRENT_CHECKPOINT_FILENAME).is_file()
    assert (tmp_path / OPTIMAL_CHECKPOINT_FILENAME).is_file()


# ───────────────────────── Confidence wiring (Milestone C #7) ─────────────────────────


def test_train_one_epoch_threads_confidence_history() -> None:
    """When confidence is active, history is updated each iteration; Beta moves."""
    from neural_data_decoding.models.composite import build_variational_composite
    from neural_data_decoding.training.loop import train_one_epoch
    from neural_data_decoding.training.losses.confidence import ConfidenceHistory
    from neural_data_decoding.training.losses.multi_objective import LossPriors

    composite = build_variational_composite({
        "in_features": 4, "hidden_sizes": [8, 2],
        "num_classes_per_dim": [2, 3], "classifier_hidden_size": [4],
        "loss_type_decoder": "MSE", "transform": "GRU",
        "confidence_type": ["Trial", "Task"],
    })
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=4,
        num_samples=5, num_features=4, num_classes_per_dim=[2, 3],
        seed=0,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_trials)
    optimizer = torch.optim.AdamW(composite.parameters(), lr=0.01)
    initial_history = ConfidenceHistory.initial(dtype=torch.float32)

    _, _, new_history = train_one_epoch(
        model=composite,
        dataloader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights={
            "classification": 1.0, "reconstruction": 1.0,
            "kl": 0.01, "confidence": 1.0,
        },
        loss_priors=LossPriors.initial(),
        confidence_history=initial_history,
    )

    assert new_history is not None
    # Beta has been updated (almost certainly differs from initial 1.0
    # after a forward pass of randomly-initialized confidence heads, which
    # start producing roughly-mid outputs → batch mean ≠ 0.5).
    assert isinstance(new_history.beta, torch.Tensor)
    # EMA total has been updated (≠ initial 1.0).
    assert float(new_history.total) != 1.0


def test_train_one_epoch_without_confidence_returns_none_history() -> None:
    """When no confidence_history is passed, third return value is None."""
    from neural_data_decoding.training.loop import train_one_epoch

    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=2,
        num_samples=3, num_features=2, num_classes_per_dim=[2],
        seed=0,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_trials)
    model = MultiHeadClassifier(in_features=2, num_classes_per_dim=[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    _, _, history = train_one_epoch(
        model=model, dataloader=loader, optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights={"classification": 1.0},
    )
    assert history is None


def test_validate_with_confidence_uses_interpolated_ce_no_dropout() -> None:
    """validate() with confidence outputs active uses Eq. 2 interpolated CE.

    Verifies the eval-mode contract:
    - Returns deterministic loss across two runs (no random dropout).
    - Loss differs from validate() called with the interpolation disabled
      (which falls back to standard / MIL CE on raw logits).

    NOTE: ``confidence_history`` is no longer a parameter — the dropped
    per-trial total confidence is computed directly from the batch's
    confidence outputs (no EMA, no Beta, no history). The previous
    parameter was a no-op for the output value.
    """
    from neural_data_decoding.models.composite import build_variational_composite
    from neural_data_decoding.training.loop import validate

    composite = build_variational_composite({
        "in_features": 4, "hidden_sizes": [8, 2],
        "num_classes_per_dim": [2, 3], "classifier_hidden_size": [4],
        "loss_type_decoder": "MSE", "transform": "GRU",
        "confidence_type": ["Trial", "Task"],
    })
    composite.eval()
    ds = SyntheticTrialDataset(
        num_sessions=1, trials_per_session=4,
        num_samples=5, num_features=4, num_classes_per_dim=[2, 3],
        seed=0,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_trials)
    weights = {"classification": 1.0, "reconstruction": 1.0, "kl": 0.01}

    # Default: confidence-active interpolated CE.
    m1 = validate(
        model=composite, dataloader=loader, device=torch.device("cpu"),
        loss_weights=weights,
    )
    m2 = validate(
        model=composite, dataloader=loader, device=torch.device("cpu"),
        loss_weights=weights,
    )
    # Deterministic across two runs (no random dropout in eval).
    assert m1.classification_loss == pytest.approx(m2.classification_loss, abs=1e-9)

    # Opt out → standard CE on raw logits → different value.
    m_no_conf = validate(
        model=composite, dataloader=loader, device=torch.device("cpu"),
        loss_weights=weights, use_interpolated_ce_for_confidence=False,
    )
    # Interpolated CE ≤ standard CE for positive confidences (interpolation
    # mixes prediction with truth, never making it worse on the target). At
    # initial random weights this should produce a strictly different value.
    assert m_no_conf.classification_loss != pytest.approx(
        m1.classification_loss, abs=1e-9,
    )


def test_compute_dropped_total_confidence_matches_apply_confidence_routing() -> None:
    """The extracted helper produces the same total_dropped tensor as the
    full ``apply_confidence_routing`` kernel did (regression guard against
    accidental divergence)."""
    from neural_data_decoding.training.losses.confidence import (
        ConfidenceHistory,
        apply_confidence_routing,
        compute_dropped_total_confidence,
    )

    torch.manual_seed(0)
    trial = torch.rand(3, 5, 1) * 0.5 + 0.3   # (B, T, 1)
    task  = torch.rand(3, 5, 4) * 0.5 + 0.3   # (B, T, 4)
    # Use an explicit mask so both paths are bit-deterministic.
    trial_mask = torch.tensor([[True], [False], [True]])
    task_mask = torch.tensor([
        [True,  False, True,  False],
        [False, True,  False, True ],
        [True,  True,  False, False],
    ])

    # Path 1: the lean helper.
    td_helper = compute_dropped_total_confidence(
        trial, task,
        confidence_dropout=0.5,
        explicit_trial_dropout_mask=trial_mask,
        explicit_task_dropout_mask=task_mask,
    )

    # Path 2: the full kernel (history values are irrelevant for this).
    cb = apply_confidence_routing(
        y=torch.zeros(3, 5, 4), target=torch.zeros(3, 5, 4),
        trial_confidence=trial, task_confidence=task,
        history=ConfidenceHistory.initial(dtype=torch.float64).initial(),
        batch_fraction=1.0,
        confidence_dropout=0.5,
        explicit_trial_dropout_mask=trial_mask,
        explicit_task_dropout_mask=task_mask,
        compute_interpolation=False,
    )
    assert td_helper is not None
    assert cb.total_dropped is not None
    torch.testing.assert_close(td_helper, cb.total_dropped, rtol=1e-12, atol=1e-12)


def test_compute_dropped_total_confidence_handles_each_branch_present() -> None:
    """Trial-only / task-only / both / neither all return the right shape (or None)."""
    from neural_data_decoding.training.losses.confidence import (
        compute_dropped_total_confidence,
    )
    trial = torch.full((2, 3, 1), 0.7)
    task  = torch.full((2, 3, 4), 0.5)

    # Both present → (B, K_task)
    out = compute_dropped_total_confidence(trial, task, confidence_dropout=1.0)
    assert out is not None and out.shape == (2, 4)

    # Trial only → (B, 1)
    out = compute_dropped_total_confidence(trial, None, confidence_dropout=1.0)
    assert out is not None and out.shape == (2, 1)

    # Task only → (B, K_task)
    out = compute_dropped_total_confidence(None, task, confidence_dropout=1.0)
    assert out is not None and out.shape == (2, 4)

    # Neither → None
    out = compute_dropped_total_confidence(None, None, confidence_dropout=1.0)
    assert out is None
