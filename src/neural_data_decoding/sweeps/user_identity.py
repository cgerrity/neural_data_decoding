"""User-identification heuristic for SLURM convenience defaults.

This module exists so the ``.slurm`` template generator can fill in
a sensible ``--mail-user`` line when the calling process is being run
by Charles (the project owner), but leaves the field blank for anyone
else. The detection layer is intentionally read-only: it never writes
the email to disk or to a git commit, and it does NOT consult the cfg
or any other persisted state — only the running process's environment
and the project's local git config.

User-directive: never auto-leak the email in git-side actions
(commit author stays the bot account). The :func:`maybe_default_mail`
function is the ONLY public callable that returns the email, and only
when the heuristic identifies the caller as Charles.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Recognized $USER values for Charles across the laptop and ACCRE.
_CHARLES_USERNAMES: frozenset[str] = frozenset({"cgerrity", "gerritcg"})

# Authoritative email for SLURM mail-user when Charles is detected.
_CHARLES_EMAIL = "charles.g.gerrity@vanderbilt.edu"


@dataclass(frozen=True, slots=True)
class UserIdentity:
    """Result of the user-identification heuristic.

    Attributes
    ----------
    username
        The detected OS-level user name (``$USER`` or ``$LOGNAME``,
        whichever is set first). Empty string when neither is set.
    git_email
        The project's local ``git config user.email`` value, or empty
        string when git is unavailable or the value is unset.
    is_charles
        ``True`` iff the username matches one of the known Charles
        accounts OR the git email matches the recognized address.
    """

    username: str
    git_email: str
    is_charles: bool


def identify_user(cwd: Path | None = None) -> UserIdentity:
    """Identify the caller via ``$USER`` and ``git config user.email``.

    Parameters
    ----------
    cwd
        Working directory for the ``git config`` lookup. Defaults to
        the current process's cwd. The repository at ``cwd`` is read
        in **read-only** mode — no git state is mutated.
    """
    username = (os.environ.get("USER") or os.environ.get("LOGNAME") or "").strip()
    git_email = _read_git_email(cwd)
    is_charles = (
        username in _CHARLES_USERNAMES
        or git_email.lower() == _CHARLES_EMAIL.lower()
    )
    return UserIdentity(username=username, git_email=git_email, is_charles=is_charles)


def maybe_default_mail(identity: UserIdentity | None = None) -> str | None:
    """Return the default ``--mail-user`` value, or ``None`` to leave blank.

    The email is emitted **only** when :func:`identify_user` flags the
    caller as Charles. Callers running as a CI bot, a different user,
    or any unidentified context get ``None`` and must pass ``--mail-user``
    explicitly to opt in.
    """
    if identity is None:
        identity = identify_user()
    return _CHARLES_EMAIL if identity.is_charles else None


def _read_git_email(cwd: Path | None) -> str:
    """Best-effort ``git config user.email``; returns ``""`` on failure."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "user.email"],
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


__all__ = [
    "UserIdentity",
    "identify_user",
    "maybe_default_mail",
]
