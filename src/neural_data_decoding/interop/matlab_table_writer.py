"""Promote a Python-written CM_Table ``.mat`` to a native MATLAB table.

The Python writer in :mod:`cm_table_format` saves a struct-of-arrays which
MATLAB loads as a struct. The downstream analysis pipeline
(``cgg_getAllEncoderCMTable``) consumes the file as a struct field on the
loaded mat-object and works either way — but a few analysis scripts
explicitly call ``istable`` / use table-specific methods. For those
callsites we need an actual ``table`` class instance on disk.

This module drives MATLAB via :mod:`neural_data_decoding.interop.matlab_runner`
(``matlab -batch`` as a subprocess) rather than the in-process MATLAB
Engine for Python. The subprocess approach works regardless of the
Python interpreter's architecture — important on Apple Silicon where a
Rosetta-translated Python can't load the native ``matlab.engine``
extension. The flow:

1. Python writes the struct via
   :func:`~neural_data_decoding.interop.cm_table_format.write_cm_table_mat`.
2. :func:`promote_struct_to_table` shells out to MATLAB, which loads the
   struct, calls ``struct2table``, and re-saves under the variable name
   ``CM_Table`` — exactly what ``cgg_saveValidationCMTable`` would have
   written.

This is the **hand-off** step — not on the training hot path. Call it
once at the end of a run (or once per sweep, batching all CM_Tables via
:func:`promote_structs_to_tables`) so MATLAB cold-start cost is amortized.

Examples
--------
>>> from pathlib import Path
>>> # Assume a struct file was written by write_cm_table_mat.
>>> # promote_struct_to_table(  # doctest: +SKIP
>>> #     Path("CM_Table_struct.mat"),
>>> #     Path("CM_Table_table.mat"),
>>> # )
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neural_data_decoding.interop.matlab_runner import (
    quote_matlab_path,
    run_matlab_batch,
)


_JSON_MARKER = "__NDD_JSON__"


def promote_struct_to_table(
    struct_mat_path: Path,
    table_mat_path: Path,
    *,
    timeout: float | None = 300.0,
) -> Path:
    """Promote one struct-of-arrays ``.mat`` to a native MATLAB ``table`` ``.mat``.

    Parameters
    ----------
    struct_mat_path
        Input ``.mat`` written by
        :func:`~neural_data_decoding.interop.cm_table_format.write_cm_table_mat`.
        Must contain a variable named ``CM_Table`` whose value is a
        scalar struct.
    table_mat_path
        Where to write the promoted ``.mat``. Overwritten if it exists.
        Saved as MATLAB v7.3 (HDF5) to match the original pipeline
        output format.
    timeout
        Seconds before the MATLAB call is killed. Defaults to 300 s
        (generous — covers cold start + a small conversion).

    Returns
    -------
    pathlib.Path
        The path that was written (same as ``table_mat_path``).

    Raises
    ------
    neural_data_decoding.interop.matlab_runner.MatlabNotFoundError
        If no MATLAB executable can be located.
    FileNotFoundError
        If ``struct_mat_path`` does not exist.
    subprocess.CalledProcessError
        If MATLAB errors (e.g. the loaded variable isn't a scalar struct).
    """
    return promote_structs_to_tables(
        [(struct_mat_path, table_mat_path)], timeout=timeout
    )[0]


def promote_structs_to_tables(
    pairs: Sequence[tuple[Path, Path]],
    *,
    timeout: float | None = 600.0,
) -> list[Path]:
    """Promote many struct ``.mat`` files to table ``.mat`` in one MATLAB session.

    Batching amortizes MATLAB's cold-start cost (~10–20 s) across all
    conversions — important when a sweep produces hundreds of CM_Tables.

    Parameters
    ----------
    pairs
        Sequence of ``(struct_in_path, table_out_path)`` tuples.
    timeout
        Seconds before the MATLAB call is killed. Defaults to 600 s.

    Returns
    -------
    list of pathlib.Path
        The output paths, in input order.

    Raises
    ------
    FileNotFoundError
        If any input path does not exist.
    subprocess.CalledProcessError
        If MATLAB errors on any conversion (the whole batch aborts).
    """
    resolved: list[tuple[Path, Path]] = []
    for struct_in, table_out in pairs:
        struct_in = Path(struct_in).resolve()
        table_out = Path(table_out).resolve()
        if not struct_in.is_file():
            raise FileNotFoundError(f"Input struct .mat not found: {struct_in}")
        table_out.parent.mkdir(parents=True, exist_ok=True)
        resolved.append((struct_in, table_out))

    # Build one MATLAB script that converts every pair. Each conversion
    # asserts the loaded variable is a scalar struct before promoting.
    statements: list[str] = []
    for struct_in, table_out in resolved:
        in_q = quote_matlab_path(struct_in)
        out_q = quote_matlab_path(table_out)
        statements.append(
            f"payload = load({in_q}); "
            "assert(isfield(payload, 'CM_Table'), "
            "'Input .mat has no CM_Table variable.'); "
            "assert(isstruct(payload.CM_Table) && isscalar(payload.CM_Table), "
            "'CM_Table must be a scalar struct.'); "
            "CM_Table = struct2table(payload.CM_Table); "
            f"save({out_q}, 'CM_Table', '-v7.3'); "
            "clear payload CM_Table;"
        )

    run_matlab_batch(" ".join(statements), timeout=timeout)
    return [table_out for _, table_out in resolved]


def describe_table_mat(table_mat_path: Path, *, timeout: float | None = 300.0) -> dict[str, Any]:
    """Return metadata about a ``CM_Table`` ``.mat`` by querying MATLAB.

    Loads the file in MATLAB, inspects the ``CM_Table`` variable, and
    returns whether it's a genuine ``table``, its column names, and its
    row count. Used by the parity tests to verify that
    :func:`promote_struct_to_table` produced a real table on disk (not
    just a struct that happens to round-trip).

    Parameters
    ----------
    table_mat_path
        Path to a ``.mat`` containing a ``CM_Table`` variable.
    timeout
        Seconds before the MATLAB call is killed.

    Returns
    -------
    dict
        ``{"istable": bool, "variables": list[str], "num_rows": int}``.

    Raises
    ------
    FileNotFoundError
        If ``table_mat_path`` does not exist.
    RuntimeError
        If MATLAB's output can't be parsed (no JSON marker found).
    """
    table_mat_path = Path(table_mat_path).resolve()
    if not table_mat_path.is_file():
        raise FileNotFoundError(f"Table .mat not found: {table_mat_path}")

    path_q = quote_matlab_path(table_mat_path)
    # Build a struct of metadata and print it as JSON between markers so
    # we can robustly extract it from MATLAB's (possibly chatty) stdout.
    commands = (
        f"loaded = load({path_q}); "
        "v = loaded.CM_Table; "
        "info = struct(); "
        "info.istable = istable(v); "
        "if istable(v); "
        "  info.variables = v.Properties.VariableNames; "
        "  info.num_rows = height(v); "
        "else; "
        "  info.variables = {{}}; "
        "  info.num_rows = 0; "
        "end; "
        f"fprintf('{_JSON_MARKER}%s{_JSON_MARKER}\\n', jsonencode(info));"
    )
    result = run_matlab_batch(commands, capture_output=True, timeout=timeout)
    return _parse_marked_json(result.stdout)


def _parse_marked_json(stdout: str) -> dict[str, Any]:
    """Extract and parse the JSON blob between the ``__NDD_JSON__`` markers."""
    start = stdout.find(_JSON_MARKER)
    if start == -1:
        raise RuntimeError(
            "Could not find JSON marker in MATLAB output. Raw stdout:\n"
            f"{stdout}"
        )
    start += len(_JSON_MARKER)
    end = stdout.find(_JSON_MARKER, start)
    if end == -1:
        raise RuntimeError(
            "Found opening JSON marker but no closing marker. Raw stdout:\n"
            f"{stdout}"
        )
    payload = stdout[start:end].strip()
    parsed = json.loads(payload)
    # MATLAB jsonencode emits a single string (not a 1-element list) for a
    # cellstr of length 1; normalize variables to a list of str.
    variables = parsed.get("variables", [])
    if isinstance(variables, str):
        variables = [variables]
    return {
        "istable": bool(parsed.get("istable", False)),
        "variables": list(variables),
        "num_rows": int(parsed.get("num_rows", 0)),
    }


__all__ = [
    "describe_table_mat",
    "promote_struct_to_table",
    "promote_structs_to_tables",
]
