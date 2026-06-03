"""CLI-side helpers for sweep flags: override application, session-run decomposition.

The dispatcher module (:mod:`.dispatcher`) provides the static table of
sweep entries; this module hosts the runtime utilities that take a
CLI-parsed namespace and mutate a resolved cfg accordingly.

Separation rationale: ``dispatcher.py`` is pure data and trivially
testable without importing torch / Hydra / OmegaConf. Keeping the
DictConfig manipulation here means ``parameter_coverage.py`` and the
sweep-emit tooling can import the dispatcher without pulling in the
training stack.
"""

from __future__ import annotations

import ast
from typing import Any, NamedTuple

from omegaconf import DictConfig, OmegaConf

from neural_data_decoding.sweeps.dispatcher import lookup as _lookup_sweep


class SessionRunDecomposition(NamedTuple):
    """Result of decomposing a ``SessionRunIDX`` into ``(session_idx, fold)``.

    Both fields are 1-based to match the MATLAB convention; the
    ``session_index_zero`` field exposes the 0-based equivalent for
    Python list indexing.
    """

    session_index: int          # 1-based, matches MATLAB SessionIDX
    session_index_zero: int     # 0-based, ready to index a Python list
    fold: int                   # 1-based, matches MATLAB Fold


def decompose_session_run_idx(
    session_run_idx: int, num_sessions: int
) -> SessionRunDecomposition:
    """Split a flat ``SessionRunIDX`` into ``(session_idx, fold)`` MATLAB-style.

    Mirrors ``cgg_assignSLURMSession.m`` lines 11-12:

    ::

        SessionIDX = mod(SessionRunIDX - 1, NumSessions) + 1
        Fold       = floor((SessionRunIDX - 1) / NumSessions) + 1

    This ordering means ``SessionRunIDX = 1..NumSessions`` runs fold 1
    across **every** session before fold 2 starts. The user prefers this
    over the flipped order (all folds of one session before moving on)
    because it lets them see initial-fold accuracy per session early in
    a sweep without committing to running every fold for every session.

    Parameters
    ----------
    session_run_idx
        The flat 1-based index a SLURM array task receives.
    num_sessions
        Total number of sessions in the cohort.

    Raises
    ------
    ValueError
        If either argument is non-positive.
    """
    if num_sessions <= 0:
        raise ValueError(f"num_sessions must be > 0, got {num_sessions}")
    if session_run_idx <= 0:
        raise ValueError(f"session_run_idx must be >= 1, got {session_run_idx}")
    zero_based = session_run_idx - 1
    session_index_zero = zero_based % num_sessions
    fold = zero_based // num_sessions + 1
    return SessionRunDecomposition(
        session_index=session_index_zero + 1,
        session_index_zero=session_index_zero,
        fold=fold,
    )


def apply_sweep_index(cfg: DictConfig, sweep_index: int) -> tuple[str, tuple[str, ...]]:
    """Merge the sweep entry's overrides into ``cfg`` in place.

    Returns ``(description, notes)`` so the caller can echo them in
    the run banner.
    """
    entry = _lookup_sweep(sweep_index)
    for key, value in entry.overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return entry.description, entry.notes


def apply_overrides(cfg: DictConfig, overrides: list[str]) -> list[tuple[str, Any]]:
    """Parse ``key=value`` strings and merge into ``cfg``.

    Values are parsed with :func:`ast.literal_eval` first, so ``True``,
    ``42``, ``1e-4``, ``[1, 2, 3]``, and ``"text"`` all behave as
    expected. A bare value that cannot be parsed (e.g. a plain word
    like ``MIL`` or ``ADAM``) falls back to a string.

    Returns the list of ``(key, parsed_value)`` pairs that were
    applied — useful for the run banner.

    Raises
    ------
    ValueError
        If any override is missing the ``=`` separator.
    """
    applied: list[tuple[str, Any]] = []
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(
                f"Override {raw!r} is missing '=' separator. "
                "Format: --override KEY=VALUE"
            )
        key, _, value_str = raw.partition("=")
        key = key.strip()
        value_str = value_str.strip()
        try:
            value: Any = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        OmegaConf.update(cfg, key, value, merge=False)
        applied.append((key, value))
    return applied


__all__ = [
    "SessionRunDecomposition",
    "apply_overrides",
    "apply_sweep_index",
    "decompose_session_run_idx",
]
