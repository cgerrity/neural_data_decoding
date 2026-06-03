"""Tests for :mod:`neural_data_decoding.data.mat_files`.

These tests cover both the pre-v7.3 and v7.3 (HDF5) paths by generating
temporary fixtures on the fly using scipy and h5py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_data_decoding.data.mat_files import load_mat


def test_load_pre_v73_mat(tmp_path: Path) -> None:
    """A v5-format ``.mat`` written by scipy round-trips through :func:`load_mat`."""
    import scipy.io as sio

    mat_path = tmp_path / "trial.mat"
    sio.savemat(str(mat_path), {"data": np.arange(10).reshape(2, 5)})

    loaded = load_mat(mat_path)
    np.testing.assert_array_equal(loaded["data"], np.arange(10).reshape(2, 5))


def test_v73_file_is_detected_as_hdf5(tmp_path: Path) -> None:
    """A file with the v7.3 version word ``0x0200`` at offset 124 is detected.

    Detection inspects the 2-byte version field of the MATLAB header —
    ``0x0200`` for v7.3 (HDF5-backed), ``0x0100`` for v5/v6/v7. A pure
    h5py file does not have this MATLAB header, so we synthesize the
    minimal header bytes here. Full round-trip parity against real
    MATLAB-generated v7.3 files lives in ``tests/parity/`` (gated on
    the ``needs_matlab`` marker) and in the smoke fixture at
    ``results/Decision/Decision_Data_0000011.mat`` (exercised by
    :mod:`tests.unit.test_mat_dataset`).
    """
    from neural_data_decoding.data.mat_files import _is_hdf5_mat

    v73_path = tmp_path / "trial_v73.mat"
    # MATLAB header layout: 116 bytes of ASCII description, 8 bytes of
    # subsys offset, then the 2-byte version word at offset 124 and the
    # endian-marker bytes "IM" at offset 126. We don't need anything
    # past offset 128 to drive the detector.
    header = b"MATLAB 7.3 MAT-file synthetic header" + b" " * 80
    header = header[:116] + (b"\x00" * 8) + b"\x00\x02" + b"IM"
    assert len(header) == 128
    v73_path.write_bytes(header)

    assert _is_hdf5_mat(v73_path) is True


def test_real_v73_fixture_is_detected_when_present() -> None:
    """The repo's real MATLAB v7.3 sample file routes through the v7.3 backend."""
    from neural_data_decoding.data.mat_files import _is_hdf5_mat

    fixture = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "Decision"
        / "Decision_Data_0000011.mat"
    )
    if not fixture.is_file():
        pytest.skip(f"Sample v7.3 .mat fixture not present: {fixture}")
    assert _is_hdf5_mat(fixture) is True


def test_v5_file_is_detected_as_non_hdf5(tmp_path: Path) -> None:
    """scipy-written ``.mat`` files (pre-v7.3) do NOT trigger the HDF5 path."""
    import scipy.io as sio

    from neural_data_decoding.data.mat_files import _is_hdf5_mat

    mat_path = tmp_path / "trial.mat"
    sio.savemat(str(mat_path), {"x": np.zeros(3)})

    assert _is_hdf5_mat(mat_path) is False


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """A nonexistent path raises :class:`FileNotFoundError`."""
    with pytest.raises(FileNotFoundError):
        load_mat(tmp_path / "nope.mat")
