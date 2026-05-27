"""Stable-schema ``EncodingParameters.yaml`` writer.

The MATLAB sweep-comparison plotter ``cgg_plotParameterSweep.m`` scans the
``EncodingParameters.yaml`` from every run in a parameter sweep field-by-field
to identify which hyperparameters varied. **All runs in a sweep must share the
same set of YAML fields** — even when individual values are defaults or zeros.

The MATLAB pipeline guarantees this via ``cgg_setBaselineDynamicParameters``
which snapshots every dynamic-parameter field up-front into a
``BaselineDynamicParameters`` struct that always exists in the saved YAML
regardless of what the active run actually uses. This module is the Python
equivalent: it takes a (potentially incomplete) resolved config plus a
**schema template** listing every field that must appear, and writes a YAML
file that always emits every field — using the template's default for any
field the run didn't override.

Critical Note #25 spells out the requirement; this module implements it.

Naming convention
-----------------
The Python codebase uses ``snake_case`` field names (Pythonic). MATLAB's
``cgg_plotParameterSweep`` reads YAML keys directly into MATLAB struct
fields, and those structs are populated by MATLAB code that uses
``PascalCase`` (mostly) plus a handful of ``camelCase`` exceptions
(``maxworkerMiniBatchSize``, ``wantStratifiedPartition``, ``wantSubset``,
``isfunction``). To stay idiomatic on the Python side **and** produce
YAMLs that drop straight into the MATLAB analysis pipeline, this module
applies a name-mapping table at write time only (option 2c from the
plan): the Python config stays snake_case, the on-disk YAML uses MATLAB
names.

Add a translation entry to :data:`PYTHON_TO_MATLAB_KEY` whenever a new
config field is introduced. The default fallback (when no explicit
mapping exists) is plain PascalCase via :func:`_default_to_matlab`. The
fallback handles most cases correctly; the override map is for the
exceptions.

Examples
--------
>>> from pathlib import Path
>>> import tempfile
>>> schema = {"weight_kl": 1.0, "weight_reconstruction": 100.0, "epoch": "Decision"}
>>> run = {"weight_kl": 5.0}
>>> with tempfile.TemporaryDirectory() as tmp:
...     out = Path(tmp) / "EncodingParameters.yaml"
...     write_encoding_parameters_yaml(out, run_config=run, schema_template=schema)
...     "WeightReconstruction" in out.read_text()
True
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


ENCODING_PARAMETERS_FILENAME = "EncodingParameters.yaml"


# ───────────────────────── Name translation table ─────────────────────────


PYTHON_TO_MATLAB_KEY: dict[str, str] = {
    # Most fields are handled by the PascalCase fallback. Below are
    # the exceptions where MATLAB uses something other than pure PascalCase
    # (verified against tests/fixtures/reference_encoding_parameters/EncodingParameters.yaml).
    "max_worker_mini_batch_size": "maxworkerMiniBatchSize",
    "want_stratified_partition": "wantStratifiedPartition",
    "want_subset": "wantSubset",
    "is_function": "isfunction",
    "is_quaddle": "IsQuaddle",
    # Names containing underscores that MATLAB keeps as-is.
    "loss_type_decoder": "LossType_Decoder",
    "loss_type_classifier": "LossType_Classifier",
    "match_type_accuracy_measure": "MatchType_Accuracy_Measure",
    "freeze_cfg": "Freeze_cfg",
    "time_start": "Time_Start",
    "time_end": "Time_End",
    "num_epochs_full_final": "NumEpochsFull_Final",
    # Acronyms — default Pascal-fallback would lowercase the trailing letters.
    "weight_kl": "WeightKL",
    "loss_factor_kl": "LossFactorKL",
    "l2_factor": "L2Factor",
    "std_channel_offset": "STDChannelOffset",
    "std_random_walk": "STDRandomWalk",
    "std_time_shift": "STDTimeShift",
    "std_white_noise": "STDWhiteNoise",
    "starting_idx": "StartingIDX",
    "ending_idx": "EndingIDX",
    # Compound words MATLAB writes with internal capitals.
    "num_epochs_autoencoder": "NumEpochsAutoEncoder",
}


def _default_to_matlab(python_key: str) -> str:
    """Convert ``snake_case`` → ``PascalCase`` as a sane default.

    The MATLAB YAML uses PascalCase for the overwhelming majority of
    fields, so this fallback gets most translations right. Edge cases
    (mixed-case, embedded underscores) belong in :data:`PYTHON_TO_MATLAB_KEY`.

    Parameters
    ----------
    python_key
        Snake-case field name from the Python config.

    Returns
    -------
    str
        PascalCase rendering — each underscore-separated chunk gets its
        first letter capitalised, then joined without underscores.
    """
    return "".join(part[:1].upper() + part[1:] for part in python_key.split("_"))


def translate_key(python_key: str) -> str:
    """Map a Python ``snake_case`` field name to its MATLAB on-disk equivalent.

    Looks up the override table first, then falls back to PascalCase.

    Parameters
    ----------
    python_key
        The field name as it appears in the Python config.

    Returns
    -------
    str
        The field name as MATLAB expects it in the YAML.
    """
    return PYTHON_TO_MATLAB_KEY.get(python_key, _default_to_matlab(python_key))


def write_encoding_parameters_yaml(
    output_path: Path,
    *,
    run_config: Mapping[str, Any],
    schema_template: Mapping[str, Any],
    translate_keys: bool = True,
) -> dict[str, Any]:
    """Write a field-symmetric ``EncodingParameters.yaml`` for one training run.

    Parameters
    ----------
    output_path
        Destination file path. Parent directory is created as needed; the
        file is overwritten if it exists.
    run_config
        The resolved Hydra config for the current run, flattened to a
        single-level mapping of parameter name → value. Anything missing
        relative to ``schema_template`` is filled in from the template.
        Keys may be Python ``snake_case``; they're translated to MATLAB's
        on-disk names at write time (see :data:`PYTHON_TO_MATLAB_KEY`).
    schema_template
        Canonical mapping of **every** field that must appear in YAMLs from
        this sweep, with the default value to use when a run doesn't supply
        one. Build this once at sweep start (e.g., from the union of all
        keys across the sweep's configs) and pass the same template to every
        run.
    translate_keys
        If ``True`` (default), apply :func:`translate_key` to each schema
        key so the on-disk YAML uses MATLAB names. Set ``False`` only for
        Python-internal use cases where MATLAB compatibility isn't needed
        (e.g., test fixtures that round-trip through Python alone).

    Returns
    -------
    dict
        The exact mapping written to disk — useful for assertions in tests
        and for echoing back into telemetry. Keys are MATLAB names when
        ``translate_keys=True`` (the default).

    Notes
    -----
    Field ordering follows ``schema_template`` insertion order. PyYAML
    preserves this with ``sort_keys=False``. ``cgg_plotParameterSweep`` is
    insensitive to ordering, but reproducible ordering helps when diffing
    runs by hand.
    """
    output_path = Path(output_path)
    merged: dict[str, Any] = {}
    for key, default in schema_template.items():
        value = run_config.get(key, default)
        out_key = translate_key(key) if translate_keys else key
        merged[out_key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)

    return merged


def read_encoding_parameters_yaml(path: Path) -> dict[str, Any]:
    """Load an ``EncodingParameters.yaml`` written by this module.

    Convenience wrapper around ``yaml.safe_load`` for symmetry with the
    writer and for use in tests that round-trip a YAML through disk.

    Parameters
    ----------
    path
        Path to a previously written ``EncodingParameters.yaml``.

    Returns
    -------
    dict
        The parsed mapping.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return dict(loaded) if loaded is not None else {}


__all__ = [
    "ENCODING_PARAMETERS_FILENAME",
    "PYTHON_TO_MATLAB_KEY",
    "read_encoding_parameters_yaml",
    "translate_key",
    "write_encoding_parameters_yaml",
]
