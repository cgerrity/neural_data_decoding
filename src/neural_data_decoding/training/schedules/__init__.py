"""Curriculum-learning schedules (read live by Dataset and loss orchestrator).

Exports the pure interpolator :func:`piecewise_anneal_value` (port of
MATLAB ``cgg_calculateDynamicValue`` + ``cgg_annealWeight``); higher-level
``Schedule`` / factory APIs land in follow-up commits in this milestone.
"""

from neural_data_decoding.training.schedules.bundle import (
    CurriculumBundle,
    KLBaseAnneal,
)
from neural_data_decoding.training.schedules.factory import (
    ScheduleWaypoints,
    WaypointConfig,
    make_freeze_schedule,
    make_load_schedule,
    make_schedule,
    make_weight_schedule,
)
from neural_data_decoding.training.schedules.interpolator import (
    piecewise_anneal_value,
)
from neural_data_decoding.training.schedules.library import (
    DEFAULT_LIBRARY_DIR,
    load_curriculum_by_name,
    load_curriculum_from_yaml,
    slugify_regime,
)
from neural_data_decoding.training.schedules.parameter import ScheduledParameter
from neural_data_decoding.training.schedules.schedule import Schedule

__all__ = [
    "CurriculumBundle",
    "DEFAULT_LIBRARY_DIR",
    "KLBaseAnneal",
    "Schedule",
    "ScheduleWaypoints",
    "ScheduledParameter",
    "WaypointConfig",
    "load_curriculum_by_name",
    "load_curriculum_from_yaml",
    "make_freeze_schedule",
    "make_load_schedule",
    "make_schedule",
    "make_weight_schedule",
    "piecewise_anneal_value",
    "slugify_regime",
]
