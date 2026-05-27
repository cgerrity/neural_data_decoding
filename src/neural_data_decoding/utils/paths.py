"""Environment-aware path resolution for the neural_data_decoding pipeline.

Replicates the behavior of MATLAB's ``cgg_getBaseFolders.m`` and
``cgg_checkACCREMounted.m`` using a Pythonic typed API. The pipeline runs
in three environments — a personal workstation (Local), the TEBA server,
and the ACCRE cluster — each with its own base directories for raw data,
results, and scratch space. This module detects which environment we're in
and returns the matching base paths.

Examples
--------
Typical usage::

    from neural_data_decoding.utils.paths import get_base_paths

    paths = get_base_paths()
    print(paths.input)        # raw data lives here
    print(paths.output)       # final results land here
    print(paths.temporary)    # scratch / checkpoints

Force a specific environment for testing::

    paths = get_base_paths(environment=RuntimeEnvironment.ACCRE)
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class RuntimeEnvironment(str, Enum):
    """The host environment the pipeline is running in.

    Attributes
    ----------
    LOCAL
        A personal workstation. Data is read from a mounted external volume
        (typically ``/Volumes/Womelsdorf Lab``) or a locally-cached ACCRE mount.
    TEBA
        The TEBA shared server. Data lives under ``/data``.
    ACCRE
        The ACCRE cluster. Data lives under ``/home`` and scratch under
        ``/data/womelsdorf_lab``.
    """

    LOCAL = "local"
    TEBA = "teba"
    ACCRE = "accre"


@dataclass(frozen=True, slots=True)
class BasePaths:
    """Base directories for the active runtime environment.

    Attributes
    ----------
    input
        Root of read-only raw / preprocessed data.
    output
        Root for final results that should be preserved.
    temporary
        Root for scratch space — checkpoints, intermediate artifacts, sweep outputs.
    environment
        The :class:`RuntimeEnvironment` these paths correspond to.
    """

    input: Path
    output: Path
    temporary: Path
    environment: RuntimeEnvironment


# ───────────────────────── Detection ─────────────────────────


def _accre_mount_visible(local_root: Path = Path("/Users/cgerrity/Documents")) -> bool:
    """Return True iff the ACCRE data mount appears to be reachable locally.

    Mirrors ``cgg_checkACCREMounted.m`` — checks for the presence of a known
    directory under the workstation's local ACCRE mount point.

    On macOS, a stale or disconnected network mount can cause ``stat`` to
    raise ``OSError: Device not configured`` rather than silently returning
    false; this function swallows that and reports the mount as unreachable.

    Parameters
    ----------
    local_root
        Workstation directory under which a mounted ACCRE share would appear.
        Defaults to ``/Users/cgerrity/Documents`` (matches the MATLAB heuristic).

    Returns
    -------
    bool
        True if ``<local_root>/ACCRE/Data_Neural`` exists as a directory.
        False on any I/O error (interpreted as "mount not currently usable").
    """
    target = local_root / "ACCRE" / "Data_Neural"
    try:
        return target.is_dir()
    except OSError:
        # Stale or disconnected mount, permission denied, etc. — treat as unreachable.
        return False


def detect_environment() -> RuntimeEnvironment:
    """Identify the host environment via hostname and filesystem heuristics.

    Detection order:

    1. **TEBA** — hostname contains ``"teba"`` or ``/data/users`` exists.
    2. **ACCRE** — hostname contains ``"accre"``, ``"panfs"``, or
       ``"vampire"``, or ``/data/womelsdorf_lab`` exists.
    3. **LOCAL** — anything else.

    Returns
    -------
    RuntimeEnvironment
        The detected environment.

    Notes
    -----
    Detection can be overridden by setting the ``NDD_FORCE_ENV`` environment
    variable to ``"local"``, ``"teba"``, or ``"accre"`` — useful for tests
    and for forcing TEBA paths from a workstation that has TEBA mounted.
    """
    override = os.environ.get("NDD_FORCE_ENV")
    if override:
        try:
            return RuntimeEnvironment(override.lower())
        except ValueError as exc:  # pragma: no cover
            raise ValueError(
                f"NDD_FORCE_ENV='{override}' is not one of "
                f"{', '.join(e.value for e in RuntimeEnvironment)}."
            ) from exc

    hostname = platform.node().lower()

    if "teba" in hostname or Path("/data/users").is_dir():
        return RuntimeEnvironment.TEBA

    accre_hostname = any(tag in hostname for tag in ("accre", "panfs", "vampire"))
    if accre_hostname or Path("/data/womelsdorf_lab").is_dir():
        return RuntimeEnvironment.ACCRE

    return RuntimeEnvironment.LOCAL


# ───────────────────────── Path mapping ─────────────────────────


def _local_paths(*, want_teba: bool) -> tuple[Path, Path, Path]:
    """Return (input, output, temporary) base paths for a local workstation.

    The user ``cgerrity`` has a known set of mount points; other users get a
    sensible default. Set ``want_teba=True`` to bypass the local-ACCRE-mount
    check (useful when the ACCRE mount is present but TEBA-style paths are
    preferred).
    """
    user = os.environ.get("USER", "")

    if user == "cgerrity" and not want_teba and _accre_mount_visible():
        accre_local = Path("/Users/cgerrity/Documents/ACCRE")
        accre_data = Path("/Users/cgerrity/Documents/ACCRE_DATA")
        return accre_local, accre_local, accre_data

    return (
        Path("/Volumes/Womelsdorf Lab"),
        Path("/Volumes/gerritcg's home"),
        Path("/Volumes/gerritcg's home"),
    )


def _teba_paths() -> tuple[Path, Path, Path]:
    """Return (input, output, temporary) base paths for the TEBA server."""
    user = os.environ.get("USER", "newuser")
    user_dir = Path(f"/data/users/{user}")
    return Path("/data"), user_dir, user_dir


def _accre_paths() -> tuple[Path, Path, Path]:
    """Return (input, output, temporary) base paths for the ACCRE cluster."""
    user = os.environ.get("USER", "newuser")
    home = Path(f"/home/{user}")
    scratch = Path(f"/data/womelsdorf_lab/{user}")
    return home, home, scratch


def get_base_paths(
    *,
    environment: RuntimeEnvironment | None = None,
    want_teba: bool = False,
) -> BasePaths:
    """Resolve the input / output / temporary base paths for the host.

    Parameters
    ----------
    environment
        Force a specific environment; if ``None``, the environment is
        auto-detected via :func:`detect_environment`.
    want_teba
        Only honored when ``environment`` resolves to
        :attr:`RuntimeEnvironment.LOCAL`. When ``True``, the local ACCRE-mount
        shortcut is bypassed and TEBA-style external-volume paths are returned.

    Returns
    -------
    BasePaths
        The resolved base paths plus the resolved environment.

    Examples
    --------
    >>> paths = get_base_paths(environment=RuntimeEnvironment.ACCRE)
    >>> str(paths.input).startswith("/home/")
    True
    """
    env = environment if environment is not None else detect_environment()

    if env is RuntimeEnvironment.TEBA:
        in_, out, tmp = _teba_paths()
    elif env is RuntimeEnvironment.ACCRE:
        in_, out, tmp = _accre_paths()
    else:
        in_, out, tmp = _local_paths(want_teba=want_teba)

    return BasePaths(input=in_, output=out, temporary=tmp, environment=env)


__all__ = [
    "BasePaths",
    "RuntimeEnvironment",
    "detect_environment",
    "get_base_paths",
]
