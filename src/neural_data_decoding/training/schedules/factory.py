"""Factory functions for building :class:`Schedule` instances.

Replaces MATLAB's three ``cgg_generate*`` subclasses with composition:
a single :class:`Schedule` class plus per-use-case factories that know
the right parameter names and default base values.

The waypoint config supports two shapes (auto-detected, mirroring
MATLAB's ``cgg_hasIndividualDynamicParameters``):

* **Shared**: a single :class:`ScheduleWaypoints` is applied to every
  base in the factory (MATLAB's flat ``DynamicAugmentation`` /
  ``DynamicFreezing`` form).
* **Per-parameter**: a ``Mapping[str, ScheduleWaypoints]`` gives each
  named parameter its own waypoint set (MATLAB's nested
  ``DynamicWeighting`` form). Parameters absent from the mapping get
  no schedule and stay at their base value.

snake_case throughout: YAML keys, factory kwargs, schedule names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, Union

from neural_data_decoding.training.schedules.parameter import ScheduledParameter
from neural_data_decoding.training.schedules.schedule import Schedule


WaypointConfig: TypeAlias = Union[
    Mapping[str, "ScheduleWaypoints"], "ScheduleWaypoints", None,
]


@dataclass(frozen=True)
class ScheduleWaypoints:
    """A pair of (epoch_points, magnitude_points) describing one ramp.

    Parameters
    ----------
    epoch_points
        Monotonically-increasing waypoint epochs (1-indexed).
    magnitude_points
        Multipliers at each waypoint. Must match ``epoch_points`` length.
    """

    epoch_points: tuple[float, ...]
    magnitude_points: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate that the two sequences have matching length."""
        if len(self.epoch_points) != len(self.magnitude_points):
            raise ValueError(
                f"epoch_points (len={len(self.epoch_points)}) and "
                f"magnitude_points (len={len(self.magnitude_points)}) "
                "must have the same length."
            )

    @classmethod
    def of(
        cls,
        epoch_points: Sequence[float],
        magnitude_points: Sequence[float],
    ) -> ScheduleWaypoints:
        """Build from arbitrary sequences (convenience constructor)."""
        return cls(
            epoch_points=tuple(epoch_points),
            magnitude_points=tuple(magnitude_points),
        )


def make_schedule(
    bases: Mapping[str, float],
    waypoints: WaypointConfig = None,
) -> Schedule:
    """Build a :class:`Schedule` from a dict of bases plus optional waypoints.

    Parameters
    ----------
    bases
        Mapping from parameter name to its static base value.
    waypoints
        Either ``None`` (no schedule — every parameter stays at base),
        a single :class:`ScheduleWaypoints` (shared across all bases),
        or a ``Mapping[str, ScheduleWaypoints]`` (per-parameter). In the
        per-parameter case, bases without a waypoint entry stay static.

    Returns
    -------
    Schedule
        Schedule keyed by the same names as ``bases``.
    """
    params: dict[str, ScheduledParameter] = {}
    if waypoints is None:
        for name, base in bases.items():
            params[name] = ScheduledParameter(base=base)
    elif isinstance(waypoints, ScheduleWaypoints):
        for name, base in bases.items():
            params[name] = ScheduledParameter(
                base=base,
                epoch_points=waypoints.epoch_points,
                magnitude_points=waypoints.magnitude_points,
            )
    else:
        for name, base in bases.items():
            wp = waypoints.get(name)
            if wp is None:
                params[name] = ScheduledParameter(base=base)
            else:
                params[name] = ScheduledParameter(
                    base=base,
                    epoch_points=wp.epoch_points,
                    magnitude_points=wp.magnitude_points,
                )
    return Schedule(params)


# ───────────────────────── Per-use-case factories ─────────────────────────


def make_load_schedule(
    *,
    std_channel_offset: float = float("nan"),
    std_white_noise: float = float("nan"),
    std_random_walk: float = float("nan"),
    std_time_shift: float = float("nan"),
    waypoints: WaypointConfig = None,
) -> Schedule:
    """Build the augmentation-magnitude schedule consumed by the Dataset.

    Replaces ``cgg_generateLoadParameters_v2``. Parameter names map to
    the MATLAB ``CurrentSTD*`` properties (snake_case Python convention).
    Consumers (Dataset) read ``schedule.current("std_white_noise")`` etc.
    live within each ``__getitem__`` call.

    Parameters
    ----------
    std_channel_offset, std_white_noise, std_random_walk, std_time_shift
        Static base STD values per augmentation. NaN disables the
        augmentation (matches MATLAB default).
    waypoints
        See :func:`make_schedule`. The Soft Three-Stage regime uses a
        single shared :class:`ScheduleWaypoints` (every augmentation
        ramps together); other regimes may give each STD its own.
    """
    return make_schedule(
        bases={
            "std_channel_offset": std_channel_offset,
            "std_white_noise": std_white_noise,
            "std_random_walk": std_random_walk,
            "std_time_shift": std_time_shift,
        },
        waypoints=waypoints,
    )


def make_weight_schedule(
    *,
    reconstruction: float = float("nan"),
    kl: float = float("nan"),
    classification: float = float("nan"),
    confidence: float = 0.0,
    offset_and_scale: float = 0.0,
    waypoints: WaypointConfig = None,
) -> Schedule:
    """Build the loss-weight schedule consumed by the loss orchestrator.

    Replaces ``cgg_generateLossWeights_v2``. Parameter names: ``"kl"``,
    ``"classification"``, ``"confidence"``, ``"reconstruction"``,
    ``"offset_and_scale"`` (snake_case translations of the MATLAB
    ``WeightKL`` etc.).

    The legacy ``cgg_annealWeight`` ramp on the KL base value is handled
    separately by :class:`KLBaseAnneal` (in ``bundle.py``) so this
    factory stays generic — there is no Weight-specific subclass.

    Parameters
    ----------
    reconstruction, kl, classification, confidence, offset_and_scale
        Static base weights per loss component. NaN disables the
        component (matches MATLAB behavior, which skips NaN losses
        from the gradient root sum).
    waypoints
        See :func:`make_schedule`. The Soft Three-Stage regime gives
        each weight its own waypoint set.
    """
    return make_schedule(
        bases={
            "reconstruction": reconstruction,
            "kl": kl,
            "classification": classification,
            "confidence": confidence,
            "offset_and_scale": offset_and_scale,
        },
        waypoints=waypoints,
    )


def make_freeze_schedule(
    *,
    encoder: float = 1.0,
    decoder: float = 1.0,
    classifier: float = 1.0,
    waypoints: WaypointConfig = None,
) -> Schedule:
    """Build the freeze-factor schedule consumed by the freeze applier.

    Replaces ``cgg_generateFreezeParameters``. Parameter names:
    ``"encoder"``, ``"decoder"``, ``"classifier"``. Values are
    learning-rate multipliers: ``1.0`` = unfrozen, ``0.0`` = fully
    frozen, ``1e-2`` = "mostly frozen but learning slowly" (the value
    the Soft Three-Stage regime uses to keep momentum alive on a
    nominally-frozen network).

    Parameters
    ----------
    encoder, decoder, classifier
        Static base learning-rate factors (default 1.0 = unfrozen).
    waypoints
        See :func:`make_schedule`. The Soft Three-Stage regime gives
        each network its own freeze waypoints.
    """
    return make_schedule(
        bases={
            "encoder": encoder,
            "decoder": decoder,
            "classifier": classifier,
        },
        waypoints=waypoints,
    )


# Re-export for convenience.
__all__ = [
    "ScheduleWaypoints",
    "WaypointConfig",
    "make_schedule",
    "make_load_schedule",
    "make_weight_schedule",
    "make_freeze_schedule",
]
