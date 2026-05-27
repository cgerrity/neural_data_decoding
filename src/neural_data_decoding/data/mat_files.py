"""HDF5-aware loader for MATLAB ``.mat`` files.

MATLAB ``.mat`` files come in two on-disk formats:

* **Pre-v7.3** — proprietary binary; read with :func:`scipy.io.loadmat`.
* **v7.3+** — HDF5; ``scipy.io.loadmat`` fails on these. Read with
  :func:`mat73.loadmat`.

This module auto-detects which format a file uses by inspecting its first
128 bytes (the MATLAB file header is ASCII text in old-format files; v7.3
files begin with the HDF5 magic bytes ``\\x89HDF``). The single
:func:`load_mat` entry point dispatches to the correct backend.

See Critical Note #17 in the migration plan for context.

Examples
--------
>>> data = load_mat("/path/to/trial_0001.mat")            # doctest: +SKIP
>>> data.keys()                                            # doctest: +SKIP
dict_keys(['Data', '__header__', '__version__', '__globals__'])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scipy.io as _sio

try:
    import mat73 as _mat73

    _MAT73_AVAILABLE = True
except ImportError:  # pragma: no cover
    _mat73 = None
    _MAT73_AVAILABLE = False


# The first 8 bytes of an HDF5 file are this magic signature.
# (MATLAB v7.3 files are HDF5 containers.)
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _is_hdf5_mat(path: Path) -> bool:
    """Return True iff ``path`` is a MATLAB v7.3 (HDF5) file.

    Parameters
    ----------
    path
        Path to the ``.mat`` file.

    Returns
    -------
    bool
        ``True`` if the file begins with the HDF5 magic signature.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    OSError
        If the file cannot be opened for reading.
    """
    with path.open("rb") as fp:
        head = fp.read(8)
    return head == _HDF5_MAGIC


def load_mat(path: str | Path) -> dict[str, Any]:
    """Load a MATLAB ``.mat`` file, auto-detecting v7.3 (HDF5) vs older.

    Parameters
    ----------
    path
        Path to the ``.mat`` file.

    Returns
    -------
    dict
        Mapping from MATLAB variable name to its NumPy / Python value. The
        exact key set depends on which scipy / mat73 version is used:

        * For pre-v7.3, scipy adds ``__header__``, ``__version__``,
          ``__globals__`` metadata keys alongside the user variables.
        * For v7.3, mat73 returns only the user variables.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ImportError
        If the file is v7.3 but the ``mat73`` package is not installed.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")

    if _is_hdf5_mat(p):
        if not _MAT73_AVAILABLE:  # pragma: no cover
            raise ImportError(
                f"{p.name} is a MATLAB v7.3 (HDF5) file, but the 'mat73' package "
                f"is not installed. Install it with `pip install mat73`."
            )
        return _mat73.loadmat(str(p))  # type: ignore[union-attr]

    return _sio.loadmat(str(p), squeeze_me=False, struct_as_record=False)


__all__ = ["load_mat"]
