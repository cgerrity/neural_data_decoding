"""Shared MATLAB-subprocess runner for interop tasks.

Drives MATLAB in **batch mode** (``matlab -batch "..."``) as a subprocess.
This is the cross-architecture-safe way to call MATLAB from Python: the
MATLAB binary runs natively (e.g. ``maca64`` on Apple Silicon) while the
Python interpreter can be *any* architecture — including a Rosetta-
translated ``x86_64`` build. It sidesteps `MATLAB Engine for Python
<https://www.mathworks.com/help/matlab/matlab-engine-for-python.html>`_,
which can only bind a Python interpreter to a MATLAB of the *same*
architecture (the reason ``pip install matlabengine`` fails on a
Rosetta Python paired with an Apple-Silicon MATLAB).

The machinery here was originally inlined in
``scripts/prepare_golden_fixtures.py``; it's promoted to the package so
both that script and the interop helpers (e.g.
:func:`neural_data_decoding.interop.matlab_table_writer.promote_struct_to_table`)
share one implementation.

Apple Silicon note
------------------
When the Python interpreter is itself running under Rosetta, MATLAB's
launcher script can pick up the wrong ``ARCH`` from its parent process
and look for ``bin/maci64/`` (Intel) instead of ``bin/maca64/``
(Apple Silicon). We detect the *real* hardware via ``sysctl``
(``os.uname()`` lies under Rosetta) and prefix the call with
``arch -arm64`` to force the launcher onto the right code path.

Examples
--------
>>> from neural_data_decoding.interop.matlab_runner import matlab_available
>>> isinstance(matlab_available(), bool)
True
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class MatlabNotFoundError(FileNotFoundError):
    """Raised when no MATLAB executable can be located on the system."""


def real_macos_arch_is_arm64() -> bool:
    """Return ``True`` iff the *hardware* is Apple Silicon, ignoring Rosetta.

    Uses ``sysctl hw.optional.arm64`` because ``os.uname().machine``
    reports ``x86_64`` for a Rosetta-translated Python — a misleading
    answer when we need to know what MATLAB's child processes will see
    natively.

    Returns
    -------
    bool
        ``True`` on Apple-Silicon hardware, ``False`` otherwise (including
        when ``sysctl`` is unavailable).
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return result.stdout.strip() == "1"


def find_matlab_executable() -> str:
    """Locate the MATLAB executable.

    Search order:

    1. ``$MATLAB_EXECUTABLE`` (explicit override), if set and present.
    2. ``$PATH`` lookup for ``matlab``.
    3. macOS standard location ``/Applications/MATLAB_R*.app/bin/matlab``
       (newest release wins).
    4. Linux standard location ``/usr/local/MATLAB/R*/bin/matlab``
       (newest release wins).

    Returns
    -------
    str
        Absolute path to the ``matlab`` executable.

    Raises
    ------
    MatlabNotFoundError
        If MATLAB cannot be located anywhere. The message lists every
        location searched so the caller can fix their environment.
    """
    env_path = os.environ.get("MATLAB_EXECUTABLE")
    if env_path:
        if not Path(env_path).is_file():
            raise MatlabNotFoundError(
                f"MATLAB_EXECUTABLE points to a missing file: {env_path}"
            )
        return env_path

    on_path = shutil.which("matlab")
    if on_path:
        return on_path

    mac_candidates = sorted(
        Path("/Applications").glob("MATLAB_R*.app/bin/matlab"),
        key=lambda p: p.parent.parent.name,  # MATLAB_R2025b > MATLAB_R2024b
        reverse=True,
    )
    if mac_candidates:
        return str(mac_candidates[0])

    linux_candidates = sorted(
        Path("/usr/local/MATLAB").glob("R*/bin/matlab"),
        key=lambda p: p.parent.parent.name,
        reverse=True,
    )
    if linux_candidates:
        return str(linux_candidates[0])

    raise MatlabNotFoundError(
        "MATLAB not found. Looked in:\n"
        "  - $MATLAB_EXECUTABLE\n"
        "  - $PATH\n"
        "  - /Applications/MATLAB_R*.app/bin/matlab (macOS)\n"
        "  - /usr/local/MATLAB/R*/bin/matlab (Linux)\n"
        "Set MATLAB_EXECUTABLE to the absolute path of your matlab binary, "
        "or add it to $PATH."
    )


def matlab_available() -> bool:
    """Return ``True`` iff a MATLAB executable can be located.

    Thin wrapper around :func:`find_matlab_executable` that swallows the
    not-found error — convenient for pytest skip-gating.

    Returns
    -------
    bool
        Whether MATLAB is callable on this system.
    """
    try:
        find_matlab_executable()
        return True
    except MatlabNotFoundError:
        return False


def run_matlab_batch(
    commands: str,
    *,
    cwd: Path | str | None = None,
    check: bool = True,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a string of MATLAB commands in batch mode.

    Parameters
    ----------
    commands
        MATLAB statements to execute (e.g.
        ``"addpath(...); doStuff(); exit(0);"``). Passed verbatim to
        ``matlab -batch``.
    cwd
        Working directory for the MATLAB process. Defaults to the current
        process working directory when ``None``.
    check
        If ``True`` (default), raise
        :class:`subprocess.CalledProcessError` when MATLAB exits non-zero
        (which ``-batch`` does on any uncaught MATLAB error).
    capture_output
        If ``True``, capture stdout/stderr on the returned object (text
        mode). Needed by callers that parse MATLAB's printed output (e.g.
        :func:`~neural_data_decoding.interop.matlab_table_writer.describe_table_mat`).
    timeout
        Optional seconds before the call is killed. ``None`` waits
        indefinitely. MATLAB cold-start is ~10–20 s, so set generous
        values.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process. ``stdout``/``stderr`` are populated only
        when ``capture_output=True``.

    Raises
    ------
    MatlabNotFoundError
        If no MATLAB executable can be located.
    subprocess.CalledProcessError
        If ``check`` is ``True`` and MATLAB exits non-zero.
    subprocess.TimeoutExpired
        If ``timeout`` elapses first.
    """
    matlab = find_matlab_executable()
    cmd: list[str] = [matlab, "-batch", commands]

    if sys.platform == "darwin" and real_macos_arch_is_arm64():
        cmd = ["arch", "-arm64", *cmd]

    return subprocess.run(
        cmd,
        check=check,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


def quote_matlab_path(path: Path | str) -> str:
    """Return a MATLAB single-quoted string literal for a filesystem path.

    MATLAB escapes a literal single-quote inside a single-quoted string
    by doubling it (``''``). This helper applies that escaping and wraps
    the result in quotes, so the returned token can be dropped straight
    into a command string.

    Parameters
    ----------
    path
        A filesystem path (may contain spaces — the repo's parent
        directory is "Neural Data Reading").

    Returns
    -------
    str
        e.g. ``"'/Users/.../Neural Data Reading/x.mat'"``.
    """
    escaped = str(path).replace("'", "''")
    return f"'{escaped}'"


__all__ = [
    "MatlabNotFoundError",
    "find_matlab_executable",
    "matlab_available",
    "quote_matlab_path",
    "real_macos_arch_is_arm64",
    "run_matlab_batch",
]
