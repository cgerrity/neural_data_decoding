"""Promote a Python-written CM_Table ``.mat`` to a native MATLAB table.

The Python writer in :mod:`cm_table_format` saves a struct-of-arrays which
MATLAB loads as a struct. The downstream analysis pipeline
(``cgg_getAllEncoderCMTable``) consumes the file as a struct field on the
loaded mat-object and works either way — but a few analysis scripts
explicitly call ``istable`` / use table-specific methods. For those
callsites we need an actual ``table`` class instance on disk.

This module wraps `MATLAB Engine for Python <https://www.mathworks.com/help/matlab/matlab-engine-for-python.html>`_
to do the promotion in-process. The flow is:

1. Python writes the struct via :func:`~neural_data_decoding.interop.cm_table_format.write_cm_table_mat`.
2. ``promote_struct_to_table(in_path, out_path)`` starts a MATLAB engine,
   loads the struct, calls ``struct2table``, and saves the result with the
   variable name ``CM_Table`` so MATLAB callsites see exactly what
   ``cgg_saveValidationCMTable`` would have written.

This is the **hand-off** step — it's not on the training hot path. Calling
it once at the end of a run (or once per sweep, batching all CM_Tables)
keeps the per-iteration cost zero.

The MATLAB engine is **optional**: importing this module does not require
``matlab.engine``. The import is deferred to call time so the rest of the
pipeline runs fine without it. Install with::

    pip install matlabengine

(See MathWorks docs for version-matching constraints — the engine wrapper
version must match the installed MATLAB release.)

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

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from matlab.engine import MatlabEngine


def promote_struct_to_table(
    struct_mat_path: Path,
    table_mat_path: Path,
    *,
    engine: "MatlabEngine | None" = None,
) -> Path:
    """Promote a struct-of-arrays ``.mat`` to a native MATLAB ``table`` ``.mat``.

    Parameters
    ----------
    struct_mat_path
        Input ``.mat`` written by
        :func:`~neural_data_decoding.interop.cm_table_format.write_cm_table_mat`.
        Must contain a variable named ``CM_Table`` whose value is a
        scalar struct.
    table_mat_path
        Where to write the promoted ``.mat``. Overwritten if it exists.
        The output uses MATLAB v7.3 (HDF5) format so it matches the
        format-on-disk of the original pipeline output.
    engine
        Optional pre-started MATLAB engine. Pass one in to batch many
        promotions in a single MATLAB session (engine startup is the
        dominant cost). If ``None``, a fresh engine is started and stopped
        around this call.

    Returns
    -------
    pathlib.Path
        The path that was written (same as ``table_mat_path``).

    Raises
    ------
    ImportError
        If ``matlab.engine`` is not installed.
    FileNotFoundError
        If ``struct_mat_path`` does not exist.
    RuntimeError
        If the MATLAB load/save fails or the loaded variable isn't a
        scalar struct.
    """
    struct_mat_path = Path(struct_mat_path).resolve()
    table_mat_path = Path(table_mat_path).resolve()

    if not struct_mat_path.is_file():
        raise FileNotFoundError(f"Input struct .mat not found: {struct_mat_path}")
    table_mat_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matlab.engine  # noqa: F401  (deferred — only imported when called)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "matlab.engine is required for promote_struct_to_table. "
            "Install via `pip install matlabengine` (MATLAB-version matched)."
        ) from exc

    owned_engine = engine is None
    if owned_engine:
        import matlab.engine as _eng

        engine = _eng.start_matlab()

    try:
        # Build the MATLAB command. Quoted paths handle spaces — needed for the
        # repo's "Neural Data Reading" parent directory.
        cmd = (
            f"payload = load('{struct_mat_path}'); "
            "assert(isstruct(payload.CM_Table) && isscalar(payload.CM_Table), "
            "'Loaded CM_Table must be a scalar struct.'); "
            "CM_Table = struct2table(payload.CM_Table); "
            f"save('{table_mat_path}', 'CM_Table', '-v7.3');"
        )
        engine.eval(cmd, nargout=0)
    except Exception as exc:  # pragma: no cover - engine-dependent
        raise RuntimeError(
            f"MATLAB engine failed to promote {struct_mat_path} → {table_mat_path}: {exc}"
        ) from exc
    finally:
        if owned_engine:
            engine.quit()

    return table_mat_path


__all__ = ["promote_struct_to_table"]
