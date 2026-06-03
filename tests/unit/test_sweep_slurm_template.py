"""Tests for the ``.slurm`` template generator + user-identity helper."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from neural_data_decoding.sweeps.slurm_template import (
    SlurmTemplateOptions,
    render_slurm_template,
    write_slurm_template,
)
from neural_data_decoding.sweeps.user_identity import (
    UserIdentity,
    identify_user,
    maybe_default_mail,
)


# ----------------------------------------------------------------------
# User identification
# ----------------------------------------------------------------------


def test_identify_user_returns_username_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``$USER`` is read first; if unset, ``$LOGNAME`` is the fallback."""
    monkeypatch.setenv("USER", "alice")
    monkeypatch.delenv("LOGNAME", raising=False)
    identity = identify_user()
    assert identity.username == "alice"

    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setenv("LOGNAME", "bob")
    identity = identify_user()
    assert identity.username == "bob"


def test_identify_user_flags_charles_usernames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both ``cgerrity`` and ``gerritcg`` resolve to ``is_charles=True``."""
    for username in ("cgerrity", "gerritcg"):
        monkeypatch.setenv("USER", username)
        identity = identify_user()
        assert identity.is_charles is True, username


def test_identify_user_flags_unknown_username_as_not_charles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-Charles username + no Charles git email → ``is_charles=False``.

    The git lookup is shielded from the dev's global ``~/.gitconfig`` by
    redirecting ``GIT_CONFIG_GLOBAL`` / ``HOME`` / ``XDG_CONFIG_HOME`` to
    empty paths so ``git config --get user.email`` finds nothing.
    """
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "nonexistent.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(tmp_path / "nonexistent.gitconfig"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    identity = identify_user(cwd=tmp_path)
    assert identity.is_charles is False
    assert identity.git_email == ""


def test_maybe_default_mail_returns_email_only_for_charles() -> None:
    """``maybe_default_mail`` emits the email iff ``is_charles`` is True."""
    charles = UserIdentity(username="cgerrity", git_email="", is_charles=True)
    assert maybe_default_mail(charles) == "charles.g.gerrity@vanderbilt.edu"

    other = UserIdentity(username="alice", git_email="", is_charles=False)
    assert maybe_default_mail(other) is None


# ----------------------------------------------------------------------
# Slurm template rendering
# ----------------------------------------------------------------------


def _non_charles_identity() -> UserIdentity:
    return UserIdentity(username="alice", git_email="", is_charles=False)


def _charles_identity() -> UserIdentity:
    return UserIdentity(
        username="gerritcg",
        git_email="charles.g.gerrity@vanderbilt.edu",
        is_charles=True,
    )


def test_render_includes_array_for_full_session_x_fold_grid() -> None:
    """``--array=1-{NumSessions*NumFolds}%{throttle}``."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
        num_sessions=25, num_folds=10, array_throttle=1,
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert "#SBATCH --array=1-250%1" in text


def test_render_omits_mail_for_unidentified_user() -> None:
    """When the user is not Charles and no ``--mail-user`` is passed, no mail line."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert "--mail-user" not in text
    assert "--mail-type" not in text


def test_render_uses_charles_email_when_detected() -> None:
    """``is_charles=True`` → mail-user line is filled in."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
    )
    text = render_slurm_template(options, identity=_charles_identity())
    assert "#SBATCH --mail-user=charles.g.gerrity@vanderbilt.edu" in text
    assert "#SBATCH --mail-type=ALL" in text


def test_render_explicit_mail_user_wins_over_detection() -> None:
    """An explicit ``--mail-user`` overrides Charles auto-detection."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
        mail_user="other@example.com",
    )
    text = render_slurm_template(options, identity=_charles_identity())
    assert "#SBATCH --mail-user=other@example.com" in text
    assert "charles.g.gerrity" not in text


def test_render_bakes_sweep_index_and_description_into_echo_line() -> None:
    """The MATLAB-equivalent banner echoes sweep index + description."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",  # SC1/IDX1 = Feedforward Network
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert 'echo "SweepIndex: 1' in text
    assert "Feedforward Network" in text


def test_render_passes_session_run_idx_to_train_command() -> None:
    """The inner train invocation references ``$SLURM_ARRAY_TASK_ID``."""
    options = SlurmTemplateOptions(
        sweep_index=5, config_name="real_data_base",
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert "--sweep-index 5" in text
    assert "--session-run-idx $SLURM_ARRAY_TASK_ID" in text
    assert "--config-name real_data_base" in text


def test_render_forwards_extra_overrides_to_train_command() -> None:
    """``extra_overrides`` become repeated ``--override`` flags on the train call."""
    options = SlurmTemplateOptions(
        sweep_index=10, config_name="real_data_base",
        extra_overrides=("data_dir=/scratch/data", "target_dir=/scratch/target"),
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert "--override data_dir=/scratch/data" in text
    assert "--override target_dir=/scratch/target" in text


def test_render_optional_repo_dir_emits_cd_line() -> None:
    """When ``repo_dir`` is set, a ``cd <path>`` line precedes the venv activate."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
        repo_dir="/home/cgerrity/neural_data_decoding",
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert "cd /home/cgerrity/neural_data_decoding" in text


def test_render_skips_cd_line_when_repo_dir_unset() -> None:
    """No ``cd`` line is emitted when ``repo_dir`` is ``None``."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert "cd " not in text


def test_write_slurm_template_creates_parents(tmp_path: Path) -> None:
    """The parent directory is created if it doesn't exist."""
    out = tmp_path / "sub" / "deep" / "sweep_1.slurm"
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
    )
    written = write_slurm_template(options, out, identity=_non_charles_identity())
    assert written.exists()
    assert "#!/bin/bash" in written.read_text()


def test_write_slurm_template_overwrites_existing(tmp_path: Path) -> None:
    """Re-writing the same path replaces the previous contents."""
    out = tmp_path / "sweep.slurm"
    out.write_text("stale junk")
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
    )
    write_slurm_template(options, out, identity=_non_charles_identity())
    assert "stale junk" not in out.read_text()


def test_invalid_sweep_index_propagates_index_error() -> None:
    """Bad ``sweep_index`` triggers the dispatcher's ``IndexError``."""
    options = SlurmTemplateOptions(
        sweep_index=999, config_name="real_data_base",
    )
    with pytest.raises(IndexError):
        render_slurm_template(options, identity=_non_charles_identity())


def test_output_filename_encodes_sweep_index_choice_and_idx() -> None:
    """Output filename embeds flat sweep_index + MATLAB (SC, IDX) for log cross-ref.

    Pins the MATLAB-parity convention: ``python_sweep-{N}-SC{c}-IDX{i}-SessionRunIDX-%a.txt``.
    Old MATLAB logs use ``...SLURMChoice-{c}_SLURMIDX-{i}-Fold-%a.txt`` — sharing
    the SC/IDX tokens lets a Python sweep be correlated to its MATLAB sibling
    just by filename.
    """
    # sweep_index 91 = MATLAB (SLURMChoice=10, SLURMIDX=1).
    options = SlurmTemplateOptions(
        sweep_index=91, config_name="real_data_base",
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert (
        "#SBATCH --output=Output_Files/"
        "python_sweep-91-SC10-IDX1-SessionRunIDX-%a.txt" in text
    )


def test_script_has_safe_bash_options() -> None:
    """``set -euo pipefail`` is at the top so failures don't get silently swallowed."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert lines[1].startswith("set -euo pipefail")


def test_header_comment_documents_array_id_semantics() -> None:
    """A leading comment explains ``%a = SessionRunIDX`` (it was Fold in MATLAB)."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
        num_sessions=25, num_folds=10,
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert "Array task ID is SessionRunIDX" in text
    assert "NumSessions=25, NumFolds=10" in text


def test_renders_timestamp_echo_for_run_provenance() -> None:
    """A ``Start: <iso8601>`` echo lands in stdout for log forensics."""
    options = SlurmTemplateOptions(
        sweep_index=1, config_name="real_data_base",
    )
    text = render_slurm_template(options, identity=_non_charles_identity())
    assert "date -u +%Y-%m-%dT%H:%M:%SZ" in text


# Sanity guard so the test process never silently consumes a real Charles
# detection from environment leakage.
def test_identify_user_smoke_does_not_crash() -> None:
    """The real-environment lookup should produce a valid object."""
    identity = identify_user()
    assert isinstance(identity.username, str)
    assert isinstance(identity.git_email, str)
    assert isinstance(identity.is_charles, bool)
    # Don't assert on the actual values — they depend on who runs the test.
    _ = os.environ.get("USER", "")
