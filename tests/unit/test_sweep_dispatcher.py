"""Tests for :mod:`neural_data_decoding.sweeps.dispatcher`.

The dispatcher is data, not logic — these tests verify that the
MATLAB ``SLURMPARAMETERS_cgg_runAutoEncoder_v2.m`` port preserves the
choice/idx grid, that field-name translation is exhaustive, and that
representative entries carry the expected overrides.
"""

from __future__ import annotations

import math

import pytest

from neural_data_decoding.sweeps import (
    SWEEP_ENTRIES,
    iter_by_choice,
    lookup,
    lookup_by_choice,
    total_sweep_count,
)
from neural_data_decoding.sweeps.dispatcher import _MATLAB_TO_PYTHON_FIELD


# ----------------------------------------------------------------------
# Counts + grid integrity
# ----------------------------------------------------------------------


def test_total_count_matches_matlab_named_entries() -> None:
    """14 choices × 10 entries + 7 from SC 15 = 147 named sweep entries."""
    assert total_sweep_count() == 147
    assert len(SWEEP_ENTRIES) == 147


def test_sweep_indices_are_dense_and_one_based() -> None:
    """sweep_index runs 1..N contiguously."""
    for i, entry in enumerate(SWEEP_ENTRIES, start=1):
        assert entry.sweep_index == i


def test_choice_blocks_have_expected_sizes() -> None:
    """SC 1-14 → 10 entries each; SC 15 → 7 entries."""
    sizes = {choice: len(entries) for choice, entries in iter_by_choice()}
    for sc in range(1, 15):
        assert sizes[sc] == 10, f"SLURMChoice {sc} should have 10 entries"
    assert sizes[15] == 7


def test_each_choice_has_dense_idx() -> None:
    """Within each SLURMChoice, idx values run 1..N without gaps."""
    for choice, entries in iter_by_choice():
        idxs = [e.matlab_idx for e in entries]
        assert idxs == list(range(1, len(entries) + 1)), choice


# ----------------------------------------------------------------------
# Lookup
# ----------------------------------------------------------------------


def test_lookup_returns_expected_first_and_last_entries() -> None:
    """sweep_index 1 = (SC 1, IDX 1) = Feedforward Network."""
    first = lookup(1)
    assert first.matlab_choice == 1
    assert first.matlab_idx == 1
    assert first.description == "Feedforward Network"
    assert first.overrides["model_name"] == "Feedforward"

    last = lookup(147)
    assert last.matlab_choice == 15
    assert last.matlab_idx == 7
    assert last.description.startswith("No Dynamic Parameters")


def test_lookup_out_of_range_raises() -> None:
    """Outside ``[1, 147]`` raises ``IndexError``."""
    for bad in (0, -1, 148, 1_000_000):
        with pytest.raises(IndexError):
            lookup(bad)


def test_lookup_by_choice_returns_correct_entry() -> None:
    """``(matlab_choice, matlab_idx)`` lookup matches flat lookup."""
    for entry in SWEEP_ENTRIES:
        assert lookup_by_choice(entry.matlab_choice, entry.matlab_idx) is entry


def test_lookup_by_choice_rejects_unknown_pair() -> None:
    """SC 15 has only 7 entries — IDX 8 must raise ``KeyError``."""
    with pytest.raises(KeyError, match="SLURMChoice=15"):
        lookup_by_choice(15, 8)
    with pytest.raises(KeyError):
        lookup_by_choice(99, 1)


def test_sweep_entry_is_frozen() -> None:
    """SweepEntry is immutable — required for safe module-global publishing."""
    entry = lookup(1)
    with pytest.raises((AttributeError, TypeError)):
        entry.sweep_index = 999  # type: ignore[misc]


# ----------------------------------------------------------------------
# Field-name translation
# ----------------------------------------------------------------------


def test_every_override_key_uses_snake_case() -> None:
    """No MATLAB CamelCase key leaks through to the SweepEntry.overrides dict."""
    for entry in SWEEP_ENTRIES:
        for key in entry.overrides:
            assert key.islower() or "_" in key, (
                f"Entry sweep_index={entry.sweep_index} has non-snake_case "
                f"override key {key!r}"
            )
            # No common CamelCase patterns
            for cc in ("Name", "Size", "Width", "Weight", "Threshold"):
                assert cc not in key, (
                    f"Entry sweep_index={entry.sweep_index} key {key!r} "
                    f"looks like CamelCase"
                )


def test_field_map_is_exhaustive() -> None:
    """Every Python cfg key produced by the dispatcher came from the map."""
    valid_python_keys = set(_MATLAB_TO_PYTHON_FIELD.values())
    for entry in SWEEP_ENTRIES:
        for key in entry.overrides:
            assert key in valid_python_keys, (
                f"Entry sweep_index={entry.sweep_index} produced unknown "
                f"Python cfg key {key!r}"
            )


# ----------------------------------------------------------------------
# Representative-entry spot checks (catch regressions in the literals)
# ----------------------------------------------------------------------


def test_sc3_data_width_entry_pairs_with_stride() -> None:
    """``Data Width 50`` also sets the matching ``WindowStride``."""
    entry = lookup_by_choice(3, 2)
    assert entry.description == "Data Width 50"
    assert entry.overrides == {"data_width": 50, "window_stride": 25}


def test_sc3_unweighted_loss_uses_empty_string() -> None:
    """CC.7's ``WeightedLoss=''`` path is preserved with the empty string."""
    entry = lookup_by_choice(3, 10)
    assert entry.description == "Unweighted Loss"
    assert entry.overrides == {"weighted_loss": ""}


def test_sc8_no_decoder_is_flagged_in_notes() -> None:
    """``LossType_Decoder='None'`` carries a partial-support note."""
    entry = lookup_by_choice(8, 8)
    assert entry.overrides == {"loss_type_decoder": "None"}
    assert entry.notes and "reconstruction" in entry.notes[0]


def test_sc9_bottleneck_depth_gt_1_is_flagged() -> None:
    """``BottleNeckDepth > 1`` carries a partial-support note."""
    for idx in (5, 6):
        entry = lookup_by_choice(9, idx)
        assert entry.overrides["bottle_neck_depth"] == idx - 3  # idx 5→2, 6→3
        assert entry.notes and "bottle_neck_depth" in entry.notes[0]
    # And the SC 10 one too.
    entry = lookup_by_choice(10, 2)
    assert entry.overrides["bottle_neck_depth"] == 4
    assert entry.notes and "bottle_neck_depth" in entry.notes[0]


def test_sc11_small_network_carries_full_override_bundle() -> None:
    """Multi-field overrides are preserved together."""
    entry = lookup_by_choice(11, 3)
    assert entry.description == "Small Network with Large Classification Weight"
    assert entry.overrides == {
        "hidden_sizes": [250],
        "classifier_hidden_size": [100],
        "weight_reconstruction": 1,
        "weight_kl": 1e-4,
        "weight_classification": 10_000,
    }


def test_sc14_pre_feedback_uses_nan_start_end_percent() -> None:
    """``StartEndPercent = [NaN, 0.5]`` translates with ``math.nan`` preserved."""
    entry = lookup_by_choice(14, 6)
    assert entry.description == "Pre-Feedback Data with MIL"
    sep = entry.overrides["start_end_percent"]
    assert isinstance(sep, list) and len(sep) == 2
    assert math.isnan(sep[0])
    assert sep[1] == 0.5
    assert entry.overrides["multiple_instance_learning_type"] == "MIL"


def test_sc15_full_aug_curriculum_bundle() -> None:
    """SC 15 / IDX 5 sets aug + weighted loss + Soft Three-Stage curriculum together."""
    entry = lookup_by_choice(15, 5)
    assert entry.overrides["std_white_noise"] == pytest.approx(0.015)
    assert entry.overrides["std_random_walk"] == pytest.approx(0.0007)
    assert entry.overrides["std_channel_offset"] == pytest.approx(0.03)
    assert entry.overrides["std_time_shift"] == 100
    assert entry.overrides["want_separate_time_shift"] is True
    assert entry.overrides["weight_reconstruction"] == 100
    assert entry.overrides["weight_classification"] == 10
    assert entry.overrides["weight_kl"] == 1
    assert entry.overrides["dynamic_parameter_set"] == "Soft Three-Stage Curriculum"


def test_sc13_duplicate_entry_is_preserved_for_index_parity() -> None:
    """SC 13 / IDX 1 and IDX 2 are both 'Self-supervised epochs - 10'.

    The duplicate is intentional in the MATLAB source — preserved here
    so sweep_index 121 and 122 stay aligned with MATLAB SLURMChoice/IDX
    references in existing run logs.
    """
    e1 = lookup_by_choice(13, 1)
    e2 = lookup_by_choice(13, 2)
    assert e1.description == e2.description == "Self-supervised epochs - 10"
    assert e1.overrides == e2.overrides == {"num_epochs_autoencoder": 10}
    assert "Duplicate" in e2.notes[0]


# ----------------------------------------------------------------------
# Override bundles are independent (no shared mutable references)
# ----------------------------------------------------------------------


def test_overrides_can_be_mutated_without_polluting_other_entries() -> None:
    """Mutating one entry's overrides dict must not leak to siblings.

    Each call to ``_build_raw_entries`` constructs fresh dicts; this test
    pins the contract so a future refactor (e.g. caching at module level)
    can't introduce silent shared-state bugs.
    """
    entry = lookup(1)
    snapshot = dict(entry.overrides)
    entry.overrides["model_name"] = "MUTATED"
    other = lookup(2)
    assert other.overrides.get("model_name") != "MUTATED"
    # Restore so other tests don't observe the mutation
    entry.overrides.clear()
    entry.overrides.update(snapshot)


# ----------------------------------------------------------------------
# SC 1 dynamic HiddenSize resolves to base bottleneck dim
# ----------------------------------------------------------------------


def test_sc1_conv_resnet_multifilter_use_resolved_bottleneck_dim() -> None:
    """``[8, 16, 32, cfg.HiddenSizes(end)]`` resolved to ``[8, 16, 32, 250]``."""
    for idx in (3, 4, 5):
        entry = lookup_by_choice(1, idx)
        assert entry.overrides["hidden_sizes"] == [8, 16, 32, 250], (
            f"SC1/IDX{idx} hidden_sizes wrong"
        )
        assert entry.overrides["want_normalization"] == "Instance"
