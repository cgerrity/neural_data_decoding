"""Unit tests for :mod:`neural_data_decoding.training.schedules`.

Covers :class:`ScheduledParameter` / :class:`Schedule` semantics that are
deterministic from the Python side — no MATLAB fixture needed. The full
MATLAB parity (per-epoch value table for the real Soft Three-Stage
Curriculum regime) lives in
``tests/parity/test_t2_dynamic_schedule_interpolator_parity.py``.
"""

from __future__ import annotations

import math

import pytest

from neural_data_decoding.training.schedules import (
    CurriculumBundle,
    KLBaseAnneal,
    Schedule,
    ScheduledParameter,
    ScheduleWaypoints,
    make_freeze_schedule,
    make_load_schedule,
    make_schedule,
    make_weight_schedule,
    piecewise_anneal_value,
)


# ───────────────────────── piecewise_anneal_value ─────────────────────────


def test_empty_schedule_returns_base() -> None:
    """An empty epoch_points sequence means the parameter is not scheduled."""
    assert piecewise_anneal_value(2.5, (), (), epoch=42) == 2.5


def test_nan_base_propagates() -> None:
    """If the base is NaN the result is NaN (matches cgg_annealWeight)."""
    assert math.isnan(piecewise_anneal_value(float("nan"), [10], [1.0], epoch=5))


def test_mismatched_waypoint_lengths_raise() -> None:
    """``epoch_points`` and ``magnitude_points`` must have matching length."""
    with pytest.raises(ValueError, match="must have the same length"):
        piecewise_anneal_value(1.0, [10, 20], [0.0], epoch=5)


def test_left_and_right_clamp() -> None:
    """Outside the waypoint range, the value clamps to the nearest magnitude."""
    assert piecewise_anneal_value(1.0, [10, 20], [0.3, 0.9], epoch=1) == pytest.approx(0.3)
    assert piecewise_anneal_value(1.0, [10, 20], [0.3, 0.9], epoch=100) == pytest.approx(0.9)


def test_off_by_one_at_segment_start() -> None:
    """At epoch == epoch_points[i] + 1 the ramp position is 0 (segment start magnitude)."""
    # Single segment [10, 20], ramp 0.0 → 1.0.
    assert piecewise_anneal_value(1.0, [10, 20], [0.0, 1.0], epoch=11) == pytest.approx(0.0)


def test_off_by_one_at_last_in_segment_epoch() -> None:
    """At epoch == epoch_points[-1] the in-segment branch reaches only (span-1)/span."""
    # Span = 10; at epoch=20 ramp = 9 * (1/10) = 0.9, NOT 1.0.
    assert piecewise_anneal_value(1.0, [10, 20], [0.0, 1.0], epoch=20) == pytest.approx(0.9)
    # The clamp kicks in at epoch=21.
    assert piecewise_anneal_value(1.0, [10, 20], [0.0, 1.0], epoch=21) == pytest.approx(1.0)


# ───────────────────────── ScheduledParameter ─────────────────────────


def test_scheduled_parameter_initial_current_equals_base() -> None:
    """Before any update, current == base."""
    p = ScheduledParameter(base=1.5)
    assert p.current == 1.5


def test_scheduled_parameter_update_returns_and_stores_new_value() -> None:
    """update(epoch) both returns and persists the new value on .current."""
    p = ScheduledParameter(base=2.0, epoch_points=[10, 20], magnitude_points=[0.0, 1.0])
    result = p.update(15)
    assert p.current == result
    assert result == pytest.approx(2.0 * 0.4)  # base * (epoch-1-10)/10 = 2 * 0.4


def test_scheduled_parameter_repeated_update_overwrites() -> None:
    """Calling update with different epochs replaces .current each time."""
    p = ScheduledParameter(base=1.0, epoch_points=[10, 20], magnitude_points=[0.0, 1.0])
    p.update(11)
    assert p.current == pytest.approx(0.0)
    p.update(21)
    assert p.current == pytest.approx(1.0)


def test_scheduled_parameter_validates_waypoint_lengths() -> None:
    """Mismatched waypoint lengths raise at construction time."""
    with pytest.raises(ValueError, match="must have the same length"):
        ScheduledParameter(base=1.0, epoch_points=[10, 20], magnitude_points=[0.0])


# ───────────────────────── Schedule ─────────────────────────


def test_schedule_bulk_update_advances_all_parameters() -> None:
    """Schedule.update(epoch) updates every parameter at once."""
    sched = Schedule({
        "a": ScheduledParameter(base=1.0, epoch_points=[10, 20], magnitude_points=[0.0, 1.0]),
        "b": ScheduledParameter(base=2.0, epoch_points=[10, 20], magnitude_points=[1.0, 0.0]),
    })
    sched.update(15)
    assert sched.current("a") == pytest.approx(0.4)            # 1 * 4/10
    assert sched.current("b") == pytest.approx(2.0 * 0.6)      # 2 * (1 - 4/10)


def test_schedule_unscheduled_parameter_stays_at_base() -> None:
    """A parameter with no waypoints keeps current == base after update."""
    sched = Schedule({
        "static": ScheduledParameter(base=7.0),
    })
    sched.update(epoch=42)
    assert sched.current("static") == 7.0


def test_schedule_contains_and_iter() -> None:
    """Schedule supports ``in`` membership and iteration over names."""
    sched = Schedule({
        "x": ScheduledParameter(base=1.0),
        "y": ScheduledParameter(base=2.0),
    })
    assert "x" in sched
    assert "z" not in sched
    assert set(iter(sched)) == {"x", "y"}
    assert len(sched) == 2


def test_schedule_getitem_returns_parameter_object() -> None:
    """Schedule[name] yields the ScheduledParameter — useful for inspection."""
    p = ScheduledParameter(base=1.0)
    sched = Schedule({"x": p})
    assert sched["x"] is p


def test_schedule_constructor_takes_a_copy() -> None:
    """Mutating the input mapping after construction does not change the schedule."""
    d = {"a": ScheduledParameter(base=1.0)}
    sched = Schedule(d)
    d["b"] = ScheduledParameter(base=2.0)
    assert "b" not in sched


def test_schedule_unknown_parameter_raises_keyerror() -> None:
    """Accessing an unknown name through .current() raises KeyError."""
    sched = Schedule({"a": ScheduledParameter(base=1.0)})
    with pytest.raises(KeyError):
        sched.current("missing")


# ───────────────────────── Live-read contract (Critical Note #8) ─────────────────────────


# ───────────────────────── ScheduleWaypoints ─────────────────────────


def test_schedule_waypoints_validates_lengths() -> None:
    """Mismatched waypoint sequences raise at construction."""
    with pytest.raises(ValueError, match="must have the same length"):
        ScheduleWaypoints(epoch_points=(10, 20), magnitude_points=(0.0,))


def test_schedule_waypoints_is_frozen() -> None:
    """ScheduleWaypoints is frozen — sharing across factories is safe."""
    wp = ScheduleWaypoints.of([10, 20], [0.0, 1.0])
    with pytest.raises(Exception):  # FrozenInstanceError, subclass of Exception
        wp.epoch_points = (5,)  # type: ignore[misc]


# ───────────────────────── Factory: make_schedule (generic) ─────────────────────────


def test_make_schedule_with_no_waypoints_leaves_params_unscheduled() -> None:
    """``waypoints=None`` means every param stays at its base value."""
    sched = make_schedule(bases={"a": 1.0, "b": 2.0})
    sched.update(50)
    assert sched.current("a") == 1.0
    assert sched.current("b") == 2.0


def test_make_schedule_with_shared_waypoints_applies_to_all_params() -> None:
    """A single ScheduleWaypoints (the flat MATLAB form) modulates every param."""
    shared = ScheduleWaypoints.of([10, 20], [0.0, 1.0])
    sched = make_schedule(bases={"a": 1.0, "b": 2.0}, waypoints=shared)
    sched.update(15)
    # Magnitude at epoch 15 = (15 - 1 - 10) / 10 = 0.4. Both bases ramp together.
    assert sched.current("a") == pytest.approx(0.4 * 1.0)
    assert sched.current("b") == pytest.approx(0.4 * 2.0)


def test_make_schedule_with_per_parameter_waypoints_isolates_each() -> None:
    """A per-parameter mapping (nested MATLAB form) gives each param its own schedule."""
    sched = make_schedule(
        bases={"a": 1.0, "b": 1.0, "static": 5.0},
        waypoints={
            "a": ScheduleWaypoints.of([10, 20], [0.0, 1.0]),
            "b": ScheduleWaypoints.of([10, 20], [1.0, 0.0]),
            # 'static' has no entry — stays at base.
        },
    )
    sched.update(15)
    assert sched.current("a") == pytest.approx(0.4)
    assert sched.current("b") == pytest.approx(0.6)
    assert sched.current("static") == pytest.approx(5.0)


# ───────────────────────── Use-case factories ─────────────────────────


def test_make_load_schedule_has_expected_parameter_names() -> None:
    """Load schedule exposes the four STD augmentation parameters."""
    sched = make_load_schedule(
        std_channel_offset=0.1, std_white_noise=0.05,
        std_random_walk=0.02, std_time_shift=0.5,
    )
    assert set(sched.names()) == {
        "std_channel_offset", "std_white_noise",
        "std_random_walk", "std_time_shift",
    }
    assert sched.current("std_channel_offset") == 0.1


def test_make_weight_schedule_has_expected_parameter_names() -> None:
    """Weight schedule exposes the five loss components."""
    sched = make_weight_schedule(
        reconstruction=1.0, kl=1.0, classification=1.0,
    )
    assert set(sched.names()) == {
        "reconstruction", "kl", "classification", "confidence", "offset_and_scale",
    }


def test_make_freeze_schedule_has_expected_parameter_names_and_default_unfrozen() -> None:
    """Freeze schedule defaults every network to factor=1.0 (unfrozen)."""
    sched = make_freeze_schedule()
    assert set(sched.names()) == {"encoder", "decoder", "classifier"}
    assert sched.current("encoder") == 1.0
    assert sched.current("decoder") == 1.0
    assert sched.current("classifier") == 1.0


# ───────────────────────── KLBaseAnneal + CurriculumBundle ─────────────────────────


def test_kl_base_anneal_matches_legacy_cgg_anneal_weight_shape() -> None:
    """KLBaseAnneal is exactly the 2-waypoint piecewise_anneal_value of [0, 1]."""
    anneal = KLBaseAnneal(initial_weight=2.0, delay_epoch=10, epoch_ramp=10)
    # Before delay: 0.
    assert anneal.value_at(5) == 0.0
    assert anneal.value_at(10) == 0.0
    # After ramp: full initial_weight.
    assert anneal.value_at(25) == 2.0
    # Mid-ramp: (15 - 1 - 10) / 10 * 2.0 = 0.8 — preserves the off-by-one.
    assert anneal.value_at(15) == pytest.approx(0.8)


def test_curriculum_bundle_update_advances_all_three_schedules() -> None:
    """CurriculumBundle.update(epoch) calls update on load, weight, and freeze."""
    bundle = CurriculumBundle(
        load=make_load_schedule(std_white_noise=0.1, waypoints=ScheduleWaypoints.of([10, 20], [0.0, 1.0])),
        weight=make_weight_schedule(reconstruction=1.0, waypoints={
            "reconstruction": ScheduleWaypoints.of([10, 20], [1.0, 0.0]),
        }),
        freeze=make_freeze_schedule(waypoints={
            "encoder": ScheduleWaypoints.of([10, 20], [0.0, 1.0]),
        }),
    )
    bundle.update(15)
    assert bundle.load.current("std_white_noise") == pytest.approx(0.1 * 0.4)
    assert bundle.weight.current("reconstruction") == pytest.approx(0.6)
    assert bundle.freeze.current("encoder") == pytest.approx(0.4)
    # Untouched (no waypoints) freeze factors stay at their default.
    assert bundle.freeze.current("decoder") == 1.0


def test_curriculum_bundle_applies_kl_base_anneal_before_dynamic_multiply() -> None:
    """Bundle's KL anneal rewrites weight['kl'].base before the dynamic schedule.

    Mirrors cgg_trainNetwork.m's two-step pipeline:
      WeightKL_Anneal = cgg_annealWeight(...);
      WeightParameters.WeightKL = WeightKL_Anneal;     # base now annealed
      WeightParameters.updateAllParameters(Epoch);     # then dynamic multiply
    """
    # KL base anneals from 0 → 2.0 over epochs 10..20.
    anneal = KLBaseAnneal(initial_weight=2.0, delay_epoch=10, epoch_ramp=10)
    # Dynamic KL magnitude is 0.5 (constant via two equal waypoints).
    weight_sched = make_weight_schedule(
        kl=2.0,
        waypoints={"kl": ScheduleWaypoints.of([0, 100], [0.5, 0.5])},
    )
    bundle = CurriculumBundle(
        load=make_load_schedule(),
        weight=weight_sched,
        freeze=make_freeze_schedule(),
        kl_anneal=anneal,
    )

    # At epoch 5: base annealed to 0, multiplied by 0.5 → 0.
    bundle.update(5)
    assert bundle.weight.current("kl") == pytest.approx(0.0)

    # At epoch 15: base annealed to 0.8, multiplied by 0.5 → 0.4.
    bundle.update(15)
    assert bundle.weight.current("kl") == pytest.approx(0.4)

    # At epoch 25 (after ramp): base = 2.0, multiplied by 0.5 → 1.0.
    bundle.update(25)
    assert bundle.weight.current("kl") == pytest.approx(1.0)


def test_curriculum_bundle_without_kl_anneal_leaves_weight_kl_alone() -> None:
    """No KL anneal → weight['kl'].base stays at its initial value across updates."""
    bundle = CurriculumBundle(
        load=make_load_schedule(),
        weight=make_weight_schedule(kl=3.0),
        freeze=make_freeze_schedule(),
    )
    initial_base = bundle.weight["kl"].base
    bundle.update(100)
    assert bundle.weight["kl"].base == initial_base


def test_live_read_contract_consumer_sees_updates_immediately() -> None:
    """A consumer holding a reference to a Schedule sees magnitude changes
    on the next ``.current(name)`` call — no snapshot, no caching.

    This is the contract the Dataset relies on for per-trial augmentation:
    each ``__getitem__`` reads the schedule's current magnitudes fresh,
    so when the training loop advances the epoch the next batch picks
    up the new magnitudes without rebuilding the dataset.
    """
    sched = Schedule({
        "noise_std": ScheduledParameter(
            base=1.0, epoch_points=[10, 20], magnitude_points=[0.1, 1.0],
        ),
    })
    sched.update(11)
    seen_at_epoch_11 = sched.current("noise_std")
    sched.update(20)
    seen_at_epoch_20 = sched.current("noise_std")
    assert seen_at_epoch_11 != seen_at_epoch_20
    # Consumer never re-imports anything; just calls .current() again.
    assert sched.current("noise_std") == seen_at_epoch_20
