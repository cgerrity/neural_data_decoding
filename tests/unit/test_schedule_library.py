"""Unit tests for the curriculum YAML library loader (no MATLAB needed).

Pinning these locally because the MATLAB-name → slug → YAML lookup is
the surface most likely to silently misroute when a new regime is added.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_data_decoding.training.schedules import (
    CurriculumBundle,
    DEFAULT_LIBRARY_DIR,
    load_curriculum_by_name,
    load_curriculum_from_yaml,
    slugify_regime,
)


# ───────────────────────── slugify_regime ─────────────────────────


@pytest.mark.parametrize(
    "regime,expected",
    [
        ("Soft Three-Stage Curriculum - Shortened", "soft_three_stage_curriculum_shortened"),
        ("None",                                    "none"),
        ("No Dynamic Parameters",                   "no_dynamic_parameters"),
        ("KL Annealing",                            "kl_annealing"),
        ("Hard Two-Stage",                          "hard_two_stage"),
        ("  Combined  ",                            "combined"),
    ],
)
def test_slugify_regime_handles_matlab_names(regime: str, expected: str) -> None:
    """slugify_regime is the canonical MATLAB-name → preset-filename mapping."""
    assert slugify_regime(regime) == expected


# ───────────────────────── load_curriculum_by_name ─────────────────────────


def test_unknown_regime_raises_filenotfound() -> None:
    """An unknown regime gives a clear error pointing at the expected YAML path."""
    with pytest.raises(FileNotFoundError, match="regime"):
        load_curriculum_by_name("Nonexistent Regime XYZ")


def test_load_none_yields_empty_bundle_that_stays_at_base_values() -> None:
    """The 'None' regime has no waypoints; every parameter stays at its base."""
    bundle = load_curriculum_by_name(
        "None",
        base_weights={"kl": 7.0},
        base_loads={"std_white_noise": 0.5},
        base_freezes={"encoder": 0.25},
    )
    bundle.update(100)
    assert bundle.weight.current("kl") == 7.0
    assert bundle.load.current("std_white_noise") == 0.5
    assert bundle.freeze.current("encoder") == 0.25


def test_load_no_dynamic_parameters_is_same_as_none() -> None:
    """The 'No Dynamic Parameters' alias produces a functionally identical bundle."""
    a = load_curriculum_by_name("No Dynamic Parameters", base_weights={"kl": 1.0})
    b = load_curriculum_by_name("None", base_weights={"kl": 1.0})
    a.update(50); b.update(50)
    assert a.weight.current("kl") == b.weight.current("kl")


def test_load_kl_annealing_ramps_kl_only() -> None:
    """The KL Annealing regime ramps KL but leaves other weights static."""
    bundle = load_curriculum_by_name(
        "KL Annealing",
        base_weights={"kl": 1.0, "classification": 5.0},
    )
    bundle.update(10)
    # KL hasn't started ramping yet (waypoint at 10 → magnitude 1e-4).
    assert bundle.weight.current("kl") == pytest.approx(1.0 * 1.0e-4)
    # Classification has no schedule → stays at base.
    assert bundle.weight.current("classification") == 5.0

    bundle.update(101)
    # KL has reached the upper waypoint (magnitude 1.0).
    assert bundle.weight.current("kl") == pytest.approx(1.0)


# ───────────────────────── Soft Three-Stage Curriculum - Shortened wiring ─────────────────────────


def test_soft_three_stage_shortened_loads_and_wires_all_three_schedules() -> None:
    """The marquee regime exercises both per-parameter (weights/freeze) and shared (aug) forms."""
    bundle = load_curriculum_by_name(
        "Soft Three-Stage Curriculum - Shortened",
        base_loads={
            "std_channel_offset": 0.03, "std_white_noise": 0.015,
            "std_random_walk": 0.0007, "std_time_shift": 100.0,
        },
        base_weights={
            "reconstruction": 100.0, "kl": 1.0,
            "classification": 10.0, "confidence": 1.0,
        },
        base_freezes={"encoder": 1.0, "decoder": 1.0, "classifier": 1.0},
    )
    # All schedule names should be present.
    assert set(bundle.weight.names()) == {
        "reconstruction", "kl", "classification", "confidence", "offset_and_scale",
    }
    assert set(bundle.freeze.names()) == {"encoder", "decoder", "classifier"}
    assert set(bundle.load.names()) == {
        "std_channel_offset", "std_white_noise",
        "std_random_walk", "std_time_shift",
    }

    # At epoch 1 the freeze.encoder schedule's left-clamp gives 1.0;
    # the classifier ramps from 1e-2.
    bundle.update(1)
    assert bundle.freeze.current("encoder") == pytest.approx(1.0)
    assert bundle.freeze.current("classifier") == pytest.approx(1.0e-2)

    # By epoch 16 the classifier should be at the fully-unfrozen plateau.
    bundle.update(16)
    assert bundle.freeze.current("classifier") == pytest.approx(1.0)


# ───────────────────────── load_curriculum_from_yaml direct path ─────────────────────────


def test_load_curriculum_from_yaml_direct_path_works() -> None:
    """Loading by direct YAML path (skipping the slug lookup) works the same."""
    path = DEFAULT_LIBRARY_DIR / "soft_three_stage_curriculum_shortened.yaml"
    bundle = load_curriculum_from_yaml(path)
    assert isinstance(bundle, CurriculumBundle)


def test_default_library_dir_resolves_to_configs_schedule() -> None:
    """DEFAULT_LIBRARY_DIR points at the in-repo configs/schedule/ directory."""
    assert DEFAULT_LIBRARY_DIR.is_dir()
    assert (DEFAULT_LIBRARY_DIR / "none.yaml").is_file()
    assert (DEFAULT_LIBRARY_DIR / "soft_three_stage_curriculum_shortened.yaml").is_file()


def test_loader_accepts_yaml_with_empty_or_missing_sections(tmp_path: Path) -> None:
    """A custom YAML with missing/empty blocks degrades gracefully to no-op schedules."""
    yaml_path = tmp_path / "experimental.yaml"
    yaml_path.write_text("matlab_name: 'Experimental'\n")
    bundle = load_curriculum_from_yaml(yaml_path, base_weights={"kl": 3.0})
    bundle.update(50)
    assert bundle.weight.current("kl") == 3.0
