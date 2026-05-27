"""Deterministic result-directory generator.

The MATLAB pipeline (``cgg_generateEncoderSubFolders_v3.m``) builds a
deeply-nested folder tree where each subdirectory encodes one
hyperparameter (Epoch / Target / Fold / ModelName / HiddenSize /
LearningRate / …). Critical Note #15 in the migration plan requires
that the Python pipeline ultimately produce the **same** tree so the
MATLAB results aggregator (``DATA_cggAllNetworkEncoderResults.m``) can
discover Python output unchanged.

For Milestone A (the tracer bullet) we don't yet need MATLAB
discoverability — we just need a deterministic, config-driven result
directory the training loop can write to. The structure below is a
**clean, hyperparameter-named subset** that captures the most
identifying fields (Epoch / Target / Fold / ModelName) plus a short
hash of the remaining config so distinct sweeps don't collide.
Milestone C will extend this to mirror the MATLAB structure exactly
once the MATLAB-side aggregator becomes the integration target.

Examples
--------
>>> from pathlib import Path
>>> path = build_result_dir(
...     base_dir=Path("/tmp/results"),
...     epoch="Synthetic_Easy",
...     target="Dimension",
...     model_name="Logistic Regression",
...     fold=1,
...     identifying_config={"learning_rate": 0.01, "batch_size": 32},
... )
>>> path.parts[-4:-1]
('Synthetic_Easy', 'Dimension', 'Logistic Regression')
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_result_dir(
    *,
    base_dir: Path,
    epoch: str,
    target: str,
    model_name: str,
    fold: int,
    identifying_config: Mapping[str, Any] | None = None,
    hash_length: int = 8,
) -> Path:
    """Construct the deterministic result directory for a single training run.

    Layout produced::

        <base_dir>/<epoch>/<target>/<model_name>/cfg-<hash>/fold-<fold>/

    where ``<hash>`` is a short SHA-256 prefix of the JSON-encoded
    ``identifying_config``. Same config + same fold ⇒ same path; any
    hyperparameter change yields a fresh directory (so completed sweeps
    don't get silently overwritten).

    Parameters
    ----------
    base_dir
        Top-level results root. Typically derived from
        :func:`neural_data_decoding.utils.paths.get_base_paths`.
    epoch
        ``cfg_Encoder.Epoch`` string (e.g., ``"Synthetic_Easy"``).
    target
        ``cfg_Encoder.Target`` string (e.g., ``"Dimension"``).
    model_name
        ``cfg_Encoder.ModelName`` architecture identifier.
    fold
        Cross-validation fold index. 1-indexed to match MATLAB.
    identifying_config
        Dict of hyperparameter key/value pairs whose change should produce
        a fresh result directory. Pass the resolved Hydra config (minus
        path/seed fields) so two semantically-distinct sweeps don't share
        a directory.
    hash_length
        Number of hex characters to keep from the SHA-256 digest. Default
        8 — collision-safe at the scales we run.

    Returns
    -------
    pathlib.Path
        The result directory path. **Not created** by this function — the
        caller is responsible for ``.mkdir(parents=True)`` (so a dry-run
        path-resolution mode is possible).

    Raises
    ------
    ValueError
        If ``fold < 1`` or any of the string components are empty.
    """
    if fold < 1:
        raise ValueError(f"fold must be >= 1 (MATLAB-style 1-indexed); got {fold}.")

    for label, value in (("epoch", epoch), ("target", target), ("model_name", model_name)):
        if not value or not str(value).strip():
            raise ValueError(f"{label} must be a non-empty string.")

    cfg_hash = _config_hash(identifying_config or {}, length=hash_length)
    return (
        Path(base_dir)
        / _sanitize(epoch)
        / _sanitize(target)
        / _sanitize(model_name)
        / f"cfg-{cfg_hash}"
        / f"fold-{fold}"
    )


def _config_hash(config: Mapping[str, Any], *, length: int) -> str:
    """Return a short stable hash of a config mapping.

    The mapping is canonicalised by:

    * Sorting keys.
    * JSON-serializing with ``sort_keys=True``, ``default=str`` for
      non-serializable types (paths, enums, etc.).

    Two mappings with the same content always hash to the same value
    regardless of insertion order, so the result directory is stable
    under arbitrary YAML rewrites.
    """
    encoded = json.dumps(dict(config), sort_keys=True, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return digest[:length]


def _sanitize(component: str) -> str:
    """Replace filesystem-hostile characters with hyphens.

    The MATLAB names use spaces, ``-``, and ``,`` liberally
    (``"Variational GRU - Dropout 0.5"``). We keep spaces (they're
    legal on every platform we care about) but strip path separators
    and other characters that would create unintended nesting.
    """
    forbidden = '/\\\x00'
    return "".join(("-" if c in forbidden else c) for c in component).strip()


__all__ = ["build_result_dir"]
