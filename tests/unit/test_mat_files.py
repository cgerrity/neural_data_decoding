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
    """Files with the HDF5 magic prefix are routed to the v7.3 backend.

    A real MATLAB v7.3 file carries proprietary attributes that ``h5py``
    cannot reproduce, so we cannot round-trip without MATLAB. Instead we
    verify only that an HDF5-magic file triggers the v7.3 code path — full
    round-trip parity against MATLAB-generated fixtures lives in
    ``tests/parity/`` and is gated on the ``needs_matlab`` marker.
    """
    from neural_data_decoding.data.mat_files import _is_hdf5_mat

    h5py = pytest.importorskip("h5py")

    hdf5_path = tmp_path / "trial_v73.mat"
    with h5py.File(str(hdf5_path), "w") as fp:
        fp.create_dataset("payload", data=np.array([1.0, 2.0]))

    assert _is_hdf5_mat(hdf5_path) is True


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
