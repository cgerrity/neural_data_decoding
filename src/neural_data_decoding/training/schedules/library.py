"""Curriculum-regime library: load named YAML presets into :class:`CurriculumBundle`.

The presets in ``configs/schedule/*.yaml`` mirror the regimes defined in
MATLAB's ``PARAMETERS_cgg_selectDynamicParameters.m``. Each preset
declares the waypoints (epoch points + magnitudes) for the three
schedules; the base values come from the caller (the training config).

Lookup is by snake_case slug. The MATLAB string regime name (e.g.
``"Soft Three-Stage Curriculum - Shortened"``) deterministically maps to
``"soft_three_stage_curriculum_shortened"`` via :func:`slugify_regime`
and from there to the file ``configs/schedule/<slug>.yaml``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from neural_data_decoding.training.schedules.bundle import (
    CurriculumBundle,
    KLBaseAnneal,
)
from neural_data_decoding.training.schedules.factory import (
    ScheduleWaypoints,
    WaypointConfig,
    make_freeze_schedule,
    make_load_schedule,
    make_weight_schedule,
)


# library.py lives at src/neural_data_decoding/training/schedules/library.py
# parents[0]=schedules, [1]=training, [2]=neural_data_decoding, [3]=src, [4]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "configs" / "schedule"


def slugify_regime(regime_name: str) -> str:
    """Convert a MATLAB regime name to a YAML filename slug.

    Examples
    --------
    >>> slugify_regime("Soft Three-Stage Curriculum - Shortened")
    'soft_three_stage_curriculum_shortened'
    >>> slugify_regime("None")
    'none'
    >>> slugify_regime("KL Annealing")
    'kl_annealing'
    """
    s = regime_name.strip().lower()
    s = s.replace("-", " ")            # collapse hyphens to spaces
    s = re.sub(r"\s+", "_", s)         # any run of whitespace becomes one _
    s = re.sub(r"[^a-z0-9_]", "", s)   # drop anything non-alphanumeric
    return s


def _parse_waypoint_dict(d: Mapping[str, Any]) -> ScheduleWaypoints:
    """Parse a ``{epoch_points, magnitude_points}`` mapping into ScheduleWaypoints."""
    return ScheduleWaypoints.of(
        epoch_points=list(d["epoch_points"]),
        magnitude_points=list(d["magnitude_points"]),
    )


def _parse_per_param_waypoints(
    raw: Mapping[str, Any] | None,
) -> Mapping[str, ScheduleWaypoints] | None:
    """Parse a per-parameter waypoint mapping. Returns None if absent/empty."""
    if not raw:
        return None
    return {name: _parse_waypoint_dict(cfg) for name, cfg in raw.items()}


def _parse_shared_or_per_param_waypoints(raw: Mapping[str, Any] | None) -> WaypointConfig:
    """Parse a waypoint block that may be flat (shared) or nested (per-parameter).

    Mirrors MATLAB's ``cgg_hasIndividualDynamicParameters``: a block is
    "flat" iff it has ``epoch_points`` / ``magnitude_points`` at its top
    level; otherwise each key is a parameter name with its own waypoint
    dict underneath.
    """
    if not raw:
        return None
    if "epoch_points" in raw and "magnitude_points" in raw:
        return _parse_waypoint_dict(raw)
    return _parse_per_param_waypoints(raw)


def load_curriculum_from_yaml(
    yaml_path: Path,
    *,
    base_loads: Mapping[str, float] | None = None,
    base_weights: Mapping[str, float] | None = None,
    base_freezes: Mapping[str, float] | None = None,
    kl_anneal: KLBaseAnneal | None = None,
) -> CurriculumBundle:
    """Load a curriculum preset YAML and build a :class:`CurriculumBundle`.

    Parameters
    ----------
    yaml_path
        Path to a preset YAML (see ``configs/schedule/*.yaml``).
    base_loads, base_weights, base_freezes
        Mappings of base values for the load / weight / freeze schedules.
        Each is forwarded as keyword arguments to the corresponding
        ``make_*_schedule`` factory; missing keys take the factory default.
    kl_anneal
        Optional legacy KL base anneal pre-step (passed straight through
        to :class:`CurriculumBundle`).

    Returns
    -------
    CurriculumBundle
        The three schedules wired with the YAML's waypoints + caller's bases.
    """
    raw = yaml.safe_load(yaml_path.read_text())
    if raw is None:
        raw = {}

    weight_wp = _parse_per_param_waypoints(raw.get("weights"))
    freeze_wp = _parse_per_param_waypoints(raw.get("freeze"))
    aug_wp = _parse_shared_or_per_param_waypoints(raw.get("augmentation"))

    load_sched = make_load_schedule(**(base_loads or {}), waypoints=aug_wp)
    weight_sched = make_weight_schedule(**(base_weights or {}), waypoints=weight_wp)
    freeze_sched = make_freeze_schedule(**(base_freezes or {}), waypoints=freeze_wp)

    return CurriculumBundle(
        load=load_sched,
        weight=weight_sched,
        freeze=freeze_sched,
        kl_anneal=kl_anneal,
    )


def load_curriculum_by_name(
    regime_name: str,
    *,
    library_dir: Path = DEFAULT_LIBRARY_DIR,
    base_loads: Mapping[str, float] | None = None,
    base_weights: Mapping[str, float] | None = None,
    base_freezes: Mapping[str, float] | None = None,
    kl_anneal: KLBaseAnneal | None = None,
) -> CurriculumBundle:
    """Look up a curriculum preset by MATLAB-style regime name.

    Parameters
    ----------
    regime_name
        The MATLAB regime string (e.g. ``"Soft Three-Stage Curriculum - Shortened"``).
        Converted to a slug via :func:`slugify_regime` and looked up as
        ``<library_dir>/<slug>.yaml``.
    library_dir
        Directory containing preset YAMLs. Defaults to
        ``configs/schedule/`` at the project root.
    base_loads, base_weights, base_freezes, kl_anneal
        Forwarded to :func:`load_curriculum_from_yaml`.

    Raises
    ------
    FileNotFoundError
        If no preset matches the regime name.
    """
    slug = slugify_regime(regime_name)
    yaml_path = library_dir / f"{slug}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No curriculum preset for regime {regime_name!r} "
            f"(expected: {yaml_path})"
        )
    return load_curriculum_from_yaml(
        yaml_path,
        base_loads=base_loads,
        base_weights=base_weights,
        base_freezes=base_freezes,
        kl_anneal=kl_anneal,
    )


__all__ = [
    "DEFAULT_LIBRARY_DIR",
    "load_curriculum_by_name",
    "load_curriculum_from_yaml",
    "slugify_regime",
]
