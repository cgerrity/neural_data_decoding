"""Tests for the per-module freeze applier (Milestone C #5)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neural_data_decoding.training.freezing import (
    apply_freeze_to_optimizer,
    build_optimizer_with_module_groups,
)
from neural_data_decoding.training.schedules import (
    Schedule,
    ScheduleWaypoints,
    make_freeze_schedule,
)


def _composite() -> nn.Module:
    """Tiny composite with two named submodules (encoder + classifier)."""
    m = nn.Module()
    m.encoder = nn.Linear(4, 8)
    m.classifier = nn.Linear(8, 3)
    return m


# ───────────────────────── build_optimizer_with_module_groups ─────────────────────────


def test_one_param_group_per_named_submodule() -> None:
    """Each entry in module_groups becomes a separate param_group with its name."""
    m = _composite()
    opt = build_optimizer_with_module_groups(
        {"encoder": m.encoder, "classifier": m.classifier}, initial_lr=0.01,
    )
    names = sorted(g["name"] for g in opt.param_groups)
    assert names == ["classifier", "encoder"]
    for g in opt.param_groups:
        assert g["lr"] == 0.01


def test_empty_module_groups_raises() -> None:
    """A model with no learnable parameters under any group raises ValueError."""
    m = nn.Module()
    m.frozen = nn.Module()  # no params
    with pytest.raises(ValueError, match="no learnable parameters"):
        build_optimizer_with_module_groups({"frozen": m.frozen}, initial_lr=0.01)


def test_shared_parameters_are_deduplicated() -> None:
    """A parameter appearing in two submodules is assigned to only one group."""
    shared_linear = nn.Linear(3, 3)
    m = nn.Module()
    m.a = nn.Sequential(shared_linear)
    m.b = nn.Sequential(shared_linear)  # SAME object
    opt = build_optimizer_with_module_groups(
        {"a": m.a, "b": m.b}, initial_lr=0.01,
    )
    # Total params across all groups = unique parameter count (= shared_linear's 2).
    counted = sum(len(g["params"]) for g in opt.param_groups)
    assert counted == 2


# ───────────────────────── apply_freeze_to_optimizer ─────────────────────────


def test_apply_freeze_scales_named_groups_by_factor() -> None:
    """Each named group's lr = base_lr × freeze_schedule.current(name)."""
    m = _composite()
    opt = build_optimizer_with_module_groups(
        {"encoder": m.encoder, "classifier": m.classifier}, initial_lr=0.1,
    )
    sched: Schedule = make_freeze_schedule(
        waypoints={
            "encoder":    ScheduleWaypoints.of([0, 100], [0.5, 0.5]),
            "classifier": ScheduleWaypoints.of([0, 100], [1.0, 1.0]),
        },
    )
    sched.update(50)
    apply_freeze_to_optimizer(opt, sched, base_lr=0.1)

    by_name = {g["name"]: g["lr"] for g in opt.param_groups}
    assert by_name["encoder"] == pytest.approx(0.05)
    assert by_name["classifier"] == pytest.approx(0.1)


def test_apply_freeze_leaves_groups_without_name_alone() -> None:
    """Param groups without a ``name`` key (or not in the schedule) are untouched."""
    m = _composite()
    opt = build_optimizer_with_module_groups(
        {"encoder": m.encoder, "classifier": m.classifier}, initial_lr=0.1,
    )
    # Schedule only knows about encoder.
    sched = make_freeze_schedule(
        encoder=1.0,
        waypoints={"encoder": ScheduleWaypoints.of([0, 100], [0.5, 0.5])},
    )
    sched.update(50)
    apply_freeze_to_optimizer(opt, sched, base_lr=0.1)

    by_name = {g["name"]: g["lr"] for g in opt.param_groups}
    assert by_name["encoder"] == pytest.approx(0.05)
    # classifier IS in the schedule by default (factor=1.0, no waypoints),
    # so it should be re-scaled to base_lr * 1.0 = 0.1 (unchanged).
    assert by_name["classifier"] == pytest.approx(0.1)


def test_apply_freeze_with_factor_zero_freezes_a_group() -> None:
    """Factor 0 sets lr to exactly 0 — gradient update has no effect."""
    m = _composite()
    opt = build_optimizer_with_module_groups(
        {"encoder": m.encoder, "classifier": m.classifier}, initial_lr=0.1,
    )
    sched = make_freeze_schedule(
        waypoints={"encoder": ScheduleWaypoints.of([0, 100], [0.0, 0.0])},
    )
    sched.update(50)
    apply_freeze_to_optimizer(opt, sched, base_lr=0.1)
    by_name = {g["name"]: g["lr"] for g in opt.param_groups}
    assert by_name["encoder"] == 0.0


def test_apply_freeze_does_not_step_frozen_params() -> None:
    """End-to-end: a frozen group's parameters do not move after an optimizer step.

    Pinning this behavior because PyTorch's AdamW with lr=0 still computes
    moments; we want zero actual weight movement.
    """
    m = _composite()
    opt = build_optimizer_with_module_groups(
        {"encoder": m.encoder, "classifier": m.classifier}, initial_lr=0.1,
    )
    sched = make_freeze_schedule(
        waypoints={"encoder": ScheduleWaypoints.of([0, 100], [0.0, 0.0])},
    )
    sched.update(50)
    apply_freeze_to_optimizer(opt, sched, base_lr=0.1)

    before_w = m.encoder.weight.detach().clone()
    # Fabricate gradients and step.
    for p in m.encoder.parameters():
        p.grad = torch.ones_like(p)
    for p in m.classifier.parameters():
        p.grad = torch.ones_like(p)
    opt.step()

    assert torch.equal(before_w, m.encoder.weight.detach()), \
        "Frozen encoder weights should NOT have moved with lr=0."
