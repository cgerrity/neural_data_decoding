"""Locate the MATLAB source tree (``Processing_Functions_cgg`` and siblings).

The Python project's fixture-generation and parity tests reference the
MATLAB pipeline (`Processing_Functions_cgg`, `LoopUtil`, `exp-utils-cjt-4`,
…). Those sources live OUTSIDE this Python project. Originally they lived
in the parent directory (``../``) when the project was nested inside the
MATLAB repo; once the project moves to a standalone location (e.g.
``~/Documents/Projects/neural_data_decoding/``) the relative path breaks.

This module decouples MATLAB-source discovery from filesystem layout. The
search order is:

1. ``$NDD_MATLAB_SOURCE_ROOT`` (explicit override; set in your shell rc).
2. ``<project_root>/..`` — the legacy nested-in-MATLAB-repo layout.
3. ``/Users/cgerrity/Documents/MATLAB/Neural Data Reading`` — a known
   absolute fallback for the dev workstation.

The first location that contains ``Processing_Functions_cgg/`` wins. If
no candidate works the function raises :class:`MatlabSourceNotFoundError`
with a message naming every path tried.

Examples
--------
>>> from neural_data_decoding.utils.matlab_source import find_matlab_source_root
>>> root = find_matlab_source_root()                 # doctest: +SKIP
>>> (root / "Processing_Functions_cgg").is_dir()    # doctest: +SKIP
True
"""

from __future__ import annotations

import os
from pathlib import Path


_REQUIRED_SUBDIR = "Processing_Functions_cgg"
_KNOWN_FALLBACK = Path("/Users/cgerrity/Documents/MATLAB/Neural Data Reading")


class MatlabSourceNotFoundError(FileNotFoundError):
    """Raised when no MATLAB-source root contains ``Processing_Functions_cgg``."""


def _project_root() -> Path:
    """Return the ``neural_data_decoding`` project root directory.

    Derived from this file's location:
    ``src/neural_data_decoding/utils/matlab_source.py`` → project root is
    three ``parents`` up.
    """
    return Path(__file__).resolve().parents[3]


def candidate_matlab_source_roots() -> list[Path]:
    """Return the candidate roots in search order, without testing them.

    Useful for diagnostic messages.

    Returns
    -------
    list of pathlib.Path
        Candidates in priority order (env override first, then fallbacks).
    """
    candidates: list[Path] = []
    env = os.environ.get("NDD_MATLAB_SOURCE_ROOT")
    if env:
        candidates.append(Path(env))
    # Legacy: project nested inside the MATLAB repo (parent of project root).
    candidates.append(_project_root().parent)
    candidates.append(_KNOWN_FALLBACK)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        c_resolved = c.expanduser()
        if c_resolved not in seen:
            seen.add(c_resolved)
            unique.append(c_resolved)
    return unique


def find_matlab_source_root() -> Path:
    """Locate the MATLAB pipeline source tree.

    Returns
    -------
    pathlib.Path
        The first candidate containing ``Processing_Functions_cgg/``.

    Raises
    ------
    MatlabSourceNotFoundError
        If no candidate contains the required subdirectory. The message
        enumerates every path that was tried.
    """
    candidates = candidate_matlab_source_roots()
    for candidate in candidates:
        if (candidate / _REQUIRED_SUBDIR).is_dir():
            return candidate
    listed = "\n".join(f"  - {c}" for c in candidates)
    raise MatlabSourceNotFoundError(
        "MATLAB source tree not found. Looked for "
        f"'{_REQUIRED_SUBDIR}/' under:\n{listed}\n\n"
        "Set NDD_MATLAB_SOURCE_ROOT to the directory containing "
        f"'{_REQUIRED_SUBDIR}/' to override."
    )


def matlab_source_available() -> bool:
    """Return ``True`` iff a MATLAB source tree can be located.

    Convenience wrapper around :func:`find_matlab_source_root` that
    swallows the not-found error — useful for pytest skip-gating.
    """
    try:
        find_matlab_source_root()
        return True
    except MatlabSourceNotFoundError:
        return False


__all__ = [
    "MatlabSourceNotFoundError",
    "candidate_matlab_source_roots",
    "find_matlab_source_root",
    "matlab_source_available",
]
