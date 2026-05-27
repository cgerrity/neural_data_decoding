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

Examples
--------
>>> from pathlib import Path
>>> import tempfile
>>> schema = {"WeightKL": 1.0, "WeightReconstruction": 100.0, "Epoch": "Decision"}
>>> run = {"WeightKL": 5.0}
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


def write_encoding_parameters_yaml(
    output_path: Path,
    *,
    run_config: Mapping[str, Any],
    schema_template: Mapping[str, Any],
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
    schema_template
        Canonical mapping of **every** field that must appear in YAMLs from
        this sweep, with the default value to use when a run doesn't supply
        one. Build this once at sweep start (e.g., from the union of all
        keys across the sweep's configs) and pass the same template to every
        run.

    Returns
    -------
    dict
        The exact mapping written to disk — useful for assertions in tests
        and for echoing back into telemetry.

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
        merged[key] = run_config.get(key, default)

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
    "read_encoding_parameters_yaml",
    "write_encoding_parameters_yaml",
]
