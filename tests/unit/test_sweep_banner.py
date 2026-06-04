"""Tests for :mod:`neural_data_decoding.sweeps.banner`.

The formatter is pure — given a :class:`RunBannerData` snapshot, it
should produce a deterministic multi-line string. These tests pin
that contract; collection (impure: env + GPU + git) is exercised
by the end-to-end CLI run in ``tests/integration/test_real_data_smoke.py``.
"""

from __future__ import annotations

from pathlib import Path

from neural_data_decoding.sweeps.banner import (
    GitState,
    GpuInfo,
    RunBannerData,
    render_banner,
)
from neural_data_decoding.sweeps.user_identity import UserIdentity


def _sample_data(**overrides: object) -> RunBannerData:
    """Construct a default :class:`RunBannerData` with optional field overrides."""
    base: dict[str, object] = {
        "timestamp_utc": "2026-06-04T00:00:00Z",
        "user": UserIdentity(username="alice", git_email="", is_charles=False),
        "git": GitState(sha="0123456789abcdef" * 2, branch="main"),
        "config_name": "real_data_base",
        "sweep_index": None,
        "sweep_description": None,
        "sweep_notes": (),
        "session_run_idx": None,
        "fold": 1,
        "subset_label": "all sessions",
        "result_dir": Path("/results/Decision/Dimension/GRU/cfg-x/fold-1"),
        "cfg_headline": {
            "epoch": "Decision",
            "target": "Dimension",
            "model_name": "GRU",
        },
        "use_real_data": True,
        "num_train_trials": 600,
        "num_val_trials": 200,
        "num_test_trials": 200,
        "num_classes_per_dim": [3, 4, 5, 6],
        "sample_shape": (59, 100, 6, 58),
        "torch_version": "2.2.2",
        "gpus": (),
    }
    base.update(overrides)
    return RunBannerData(**base)  # type: ignore[arg-type]


def test_banner_includes_timestamp_user_git_torch() -> None:
    """Provenance fields all land in the rendered output."""
    text = render_banner(_sample_data())
    assert "2026-06-04T00:00:00Z" in text
    assert "alice" in text
    assert "0123456789ab" in text  # first 12 chars of the fixture SHA
    assert "(main)" in text
    assert "2.2.2" in text


def test_banner_flags_charles_when_detected() -> None:
    """``UserIdentity.is_charles`` adds the auto-detected suffix."""
    charles = UserIdentity(
        username="cgerrity",
        git_email="charles.g.gerrity@vanderbilt.edu",
        is_charles=True,
    )
    text = render_banner(_sample_data(user=charles))
    assert "cgerrity [Charles auto-detected]" in text


def test_banner_omits_charles_flag_for_others() -> None:
    """Non-Charles users get a bare username."""
    text = render_banner(_sample_data())
    assert "alice" in text
    assert "Charles auto-detected" not in text


def test_banner_handles_missing_git_state() -> None:
    """Empty SHA → a clean ``<not a git repository>`` marker."""
    text = render_banner(_sample_data(git=GitState(sha="", branch="")))
    assert "<not a git repository>" in text


def test_banner_includes_sweep_info_when_indexed() -> None:
    """``sweep_index`` + description + notes all render."""
    text = render_banner(
        _sample_data(
            sweep_index=91,
            sweep_description="Gradient Accumulation size 10",
            sweep_notes=("test note about partial support",),
        )
    )
    assert "sweep idx : 91 — Gradient Accumulation size 10" in text
    assert "note: test note about partial support" in text


def test_banner_omits_sweep_lines_when_no_index() -> None:
    """No sweep index → no sweep lines, just config/fold/subset."""
    text = render_banner(_sample_data())
    assert "sweep idx" not in text


def test_banner_includes_session_run_idx_when_provided() -> None:
    """``session_run_idx`` renders only when set."""
    text = render_banner(_sample_data(session_run_idx=26))
    assert "sessionRun: 26" in text

    text_unset = render_banner(_sample_data(session_run_idx=None))
    assert "sessionRun" not in text_unset


def test_banner_renders_sample_shape_with_axis_labels() -> None:
    """Sample shape is rendered as ``(WxTxAxC) (W, T, A, C)`` for clarity."""
    text = render_banner(_sample_data())
    assert "(59x100x6x58) (W, T, A, C)" in text


def test_banner_shows_trial_split_counts() -> None:
    """train / val / test counts all appear in the dataset block."""
    text = render_banner(_sample_data())
    assert "trials    : train=600 val=200 test=200" in text


def test_banner_lists_each_cfg_headline_key() -> None:
    """Every key in ``cfg_headline`` appears under the ``cfg headline:`` block."""
    headline = {"a": 1, "longer_key": 2, "third": "ok"}
    text = render_banner(_sample_data(cfg_headline=headline))
    assert "a          = 1" in text  # padded to the longest key
    assert "longer_key = 2" in text
    assert "third      = 'ok'" in text


def test_banner_says_cpu_only_when_no_gpus() -> None:
    """No GPUs → the explicit CPU marker, not an empty GPU section."""
    text = render_banner(_sample_data(gpus=()))
    assert "GPUs      : <none detected — CPU only>" in text


def test_banner_renders_gpu_table_when_present() -> None:
    """GPU entries appear with index, name, memory, and selection mark."""
    gpu0 = GpuInfo(index=0, name="NVIDIA RTX A6000", total_memory_gb=47.99, is_selected=True)
    gpu1 = GpuInfo(index=1, name="NVIDIA TITAN Xp", total_memory_gb=11.91, is_selected=False)
    text = render_banner(_sample_data(gpus=(gpu0, gpu1)))
    assert "NVIDIA RTX A6000" in text
    assert "47.99" in text
    assert "NVIDIA TITAN Xp" in text
    # Selection mark only on gpu0.
    selected_lines = [ln for ln in text.splitlines() if "NVIDIA" in ln]
    assert any("yes" in ln for ln in selected_lines if "A6000" in ln)
    assert all("yes" not in ln for ln in selected_lines if "Xp" in ln)


def test_banner_handles_synthetic_dataset_label() -> None:
    """Synthetic dataset gets the ``synthetic`` label, not ``real (.mat)``."""
    text = render_banner(_sample_data(use_real_data=False))
    assert "dataset   : synthetic" in text


def test_banner_ends_with_double_line_rule() -> None:
    """A trailing ``===`` rule separates the banner from training output."""
    text = render_banner(_sample_data())
    lines = text.splitlines()
    assert lines[-1].startswith("=")
    # And it's the same width throughout
    rule_lines = [ln for ln in lines if set(ln) == {"="}]
    assert len(rule_lines) >= 2
    assert len({len(ln) for ln in rule_lines}) == 1
