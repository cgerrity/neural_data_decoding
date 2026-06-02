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

import mat73 as _mat73
import scipy.io as _sio


# MATLAB v7.3 files are HDF5 *containers*, but they have a 116-byte
# ASCII description header followed by 8 subsys bytes, then a 2-byte
# version word at offset 124. v5/v6/v7 files use 0x0100 there; v7.3
# uses 0x0200. The actual HDF5 magic signature appears at offset 512
# (start of the embedded HDF5 stream). Checking the version word is
# the cheapest reliable discriminator.
_MAT_VERSION_OFFSET = 124
_MAT_VERSION_V73 = 0x0200


def _is_hdf5_mat(path: Path) -> bool:
    """Return ``True`` iff ``path`` is a MATLAB v7.3 (HDF5) file.

    Inspects the 2-byte version field at offset 124 of the MATLAB
    header. v7.3 files use ``0x0200`` there; pre-v7.3 files use
    ``0x0100``.

    Parameters
    ----------
    path
        Path to the ``.mat`` file.

    Returns
    -------
    bool
        ``True`` if the file is MATLAB v7.3 (HDF5-backed).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    OSError
        If the file cannot be opened for reading or is shorter than
        the MATLAB header (128 bytes).
    """
    with path.open("rb") as fp:
        fp.seek(_MAT_VERSION_OFFSET)
        version_bytes = fp.read(2)
    if len(version_bytes) < 2:
        raise OSError(
            f"{path} is too short to be a MATLAB .mat file (header "
            "expected at least 128 bytes).",
        )
    # Little-endian per the MATLAB spec; the byte order marker at
    # offset 126 confirms but we don't need to inspect it for the
    # version check since both LE and BE happen to write the version
    # the same way.
    version = int.from_bytes(version_bytes, "little")
    return version == _MAT_VERSION_V73


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
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")

    if _is_hdf5_mat(p):
        return _mat73.loadmat(str(p))

    return _sio.loadmat(str(p), squeeze_me=False, struct_as_record=False)


__all__ = ["load_mat"]
