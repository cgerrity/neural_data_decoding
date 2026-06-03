"""Tests for :mod:`neural_data_decoding.sweeps.cli_helpers`."""

from __future__ import annotations

import math

import pytest
from omegaconf import OmegaConf

from neural_data_decoding.sweeps.cli_helpers import (
    apply_overrides,
    apply_sweep_index,
    decompose_session_run_idx,
)


# ----------------------------------------------------------------------
# SessionRunIDX decomposition — MATLAB ordering must NOT flip
# ----------------------------------------------------------------------


def test_session_run_idx_first_fold_walks_all_sessions() -> None:
    """K = 1..NumSessions runs fold 1 across every session (MATLAB ordering)."""
    num_sessions = 25
    for k in range(1, num_sessions + 1):
        d = decompose_session_run_idx(k, num_sessions)
        assert d.fold == 1, f"K={k} should still be fold 1, got fold {d.fold}"
        assert d.session_index == k
        assert d.session_index_zero == k - 1


def test_session_run_idx_advances_fold_after_full_sweep() -> None:
    """K = NumSessions + 1 starts fold 2 at session 1."""
    num_sessions = 25
    d = decompose_session_run_idx(num_sessions + 1, num_sessions)
    assert d.fold == 2
    assert d.session_index == 1
    assert d.session_index_zero == 0


def test_session_run_idx_full_grid_round_trip() -> None:
    """All (session, fold) pairs are visited exactly once over the grid."""
    num_sessions = 25
    num_folds = 10
    seen: set[tuple[int, int]] = set()
    for k in range(1, num_sessions * num_folds + 1):
        d = decompose_session_run_idx(k, num_sessions)
        assert 1 <= d.session_index <= num_sessions
        assert 1 <= d.fold <= num_folds
        seen.add((d.session_index, d.fold))
    assert len(seen) == num_sessions * num_folds


def test_session_run_idx_rejects_nonpositive() -> None:
    """``session_run_idx`` and ``num_sessions`` must both be positive."""
    with pytest.raises(ValueError, match="num_sessions"):
        decompose_session_run_idx(1, 0)
    with pytest.raises(ValueError, match="session_run_idx"):
        decompose_session_run_idx(0, 5)
    with pytest.raises(ValueError, match="session_run_idx"):
        decompose_session_run_idx(-3, 5)


# ----------------------------------------------------------------------
# apply_sweep_index — bundles overrides into a DictConfig
# ----------------------------------------------------------------------


def test_apply_sweep_index_overrides_cfg_keys() -> None:
    """SC1/IDX1 (Feedforward Network) sets model_name and want_normalization."""
    cfg = OmegaConf.create({
        "model_name": "GRU",
        "want_normalization": False,
        "other_field": 42,
    })
    description, notes = apply_sweep_index(cfg, 1)
    assert description == "Feedforward Network"
    assert notes == ()
    assert cfg.model_name == "Feedforward"
    assert cfg.want_normalization is True
    assert cfg.other_field == 42  # unrelated field untouched


def test_apply_sweep_index_returns_notes_for_partial_support_entries() -> None:
    """SC8/IDX8 (``LossType_Decoder='None'``) returns a non-empty notes tuple."""
    cfg = OmegaConf.create({"loss_type_decoder": "MSE"})
    description, notes = apply_sweep_index(cfg, 78)  # sweep_index for (8, 8)
    assert description == "No Decoder"
    assert cfg.loss_type_decoder == "None"
    assert notes and "reconstruction" in notes[0]


def test_apply_sweep_index_handles_nan_in_list_value() -> None:
    """SC14/IDX6 (``StartEndPercent = [NaN, 0.5]``) round-trips through OmegaConf."""
    cfg = OmegaConf.create({"start_end_percent": [None, None]})
    apply_sweep_index(cfg, 136)  # (14, 6) → 13*10 + 6 = 136
    val = OmegaConf.to_container(cfg.start_end_percent, resolve=True)
    assert isinstance(val, list) and len(val) == 2
    assert math.isnan(float(val[0]))
    assert val[1] == 0.5


# ----------------------------------------------------------------------
# apply_overrides — key=value strings parsed via literal_eval
# ----------------------------------------------------------------------


def test_apply_overrides_parses_python_literals() -> None:
    """Integers, floats, bools, lists, and strings all parse correctly."""
    cfg = OmegaConf.create({
        "data_width": 100,
        "want_normalization": False,
        "dropout": 0.5,
        "hidden_sizes": [1000, 500],
        "model_name": "GRU",
    })
    applied = apply_overrides(cfg, [
        "data_width=50",
        "want_normalization=True",
        "dropout=0.25",
        "hidden_sizes=[500, 250]",
        "model_name=Feedforward",  # unquoted string fallback
    ])
    assert cfg.data_width == 50
    assert cfg.want_normalization is True
    assert cfg.dropout == 0.25
    assert list(cfg.hidden_sizes) == [500, 250]
    assert cfg.model_name == "Feedforward"
    assert len(applied) == 5


def test_apply_overrides_rejects_missing_equals() -> None:
    """An override without ``=`` raises a clear ``ValueError``."""
    cfg = OmegaConf.create({"x": 1})
    with pytest.raises(ValueError, match="missing '=' separator"):
        apply_overrides(cfg, ["x42"])


def test_apply_overrides_falls_back_to_string_for_unparseable_value() -> None:
    """Bare identifiers like ``ADAM`` parse as strings, not NameErrors."""
    cfg = OmegaConf.create({"optimizer": "SGDM"})
    apply_overrides(cfg, ["optimizer=ADAM"])
    assert cfg.optimizer == "ADAM"


def test_apply_overrides_can_set_nested_dotted_key() -> None:
    """``OmegaConf.update`` honors dotted paths, so nested cfg fields work."""
    cfg = OmegaConf.create({"outer": {"inner": 1}})
    apply_overrides(cfg, ["outer.inner=42"])
    assert cfg.outer.inner == 42
