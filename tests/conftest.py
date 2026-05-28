"""Pytest fixtures shared across the test suite.

Includes:

* Seed-control fixture for deterministic tests (Critical Note #18 in the plan).
* Loaders for MATLAB-generated reference fixtures (Milestone 0 stub — fixtures
  themselves are produced by ``scripts/prepare_golden_fixtures.py``).
* Convenience markers for parity tests that require MATLAB or GPU resources.

See ``Plans/neural_data_decoding_plan.md`` for the full parity-gate matrix
(G1–G8).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest


# ───────────────────────── Paths ─────────────────────────

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    """Return the root directory of test fixtures.

    Returns
    -------
    Path
        The ``tests/fixtures/`` directory.
    """
    return FIXTURE_ROOT


@pytest.fixture(scope="session")
def golden_weights_dir(fixture_root: Path) -> Path:
    """Return the directory containing MATLAB-trained golden-weight ``.mat`` files."""
    return fixture_root / "golden_weights"


@pytest.fixture(scope="session")
def golden_batches_dir(fixture_root: Path) -> Path:
    """Return the directory containing fixed ``(X, T, expected_loss)`` tuples."""
    return fixture_root / "golden_batches"


@pytest.fixture(scope="session")
def reference_partitions_dir(fixture_root: Path) -> Path:
    """Return the directory containing MATLAB-generated K-fold partition ``.mat`` files."""
    return fixture_root / "reference_partitions"


@pytest.fixture(scope="session")
def reference_cm_tables_dir(fixture_root: Path) -> Path:
    """Return the directory containing MATLAB-generated reference ``CM_Table.mat`` files."""
    return fixture_root / "reference_cm_tables"


# ───────────────────────── Determinism ─────────────────────────


@pytest.fixture()
def seeded(request: pytest.FixtureRequest) -> int:
    """Seed all RNGs to a deterministic value for the duration of a test.

    The seed defaults to 0; override per-test with::

        @pytest.mark.parametrize("seeded", [42], indirect=True)
        def test_something(seeded): ...

    Parameters
    ----------
    request
        Pytest fixture request, used to read a parametrized seed.

    Returns
    -------
    int
        The seed value that was applied.

    Notes
    -----
    Seeds ``random``, ``numpy.random``, and ``torch`` (CPU + CUDA if available).
    Does NOT enable ``torch.backends.cudnn.deterministic=True`` by default — that
    is a performance hit and is only needed for bit-exact reproducibility, which
    is explicitly NOT a parity goal of this project (see ADR 001).
    """
    seed = getattr(request, "param", 0)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass

    return seed


# ───────────────────────── Fixture-availability helpers ─────────────────────────


def _matlab_available() -> bool:
    """Return True iff a MATLAB executable can be located.

    Some parity tests drive MATLAB via ``matlab -batch`` (e.g. to verify
    Python output loads as a native table). Those tests are gated by the
    ``needs_matlab`` marker; this helper backs that gate.

    Delegates to
    :func:`neural_data_decoding.interop.matlab_runner.matlab_available`,
    which searches ``$MATLAB_EXECUTABLE`` / ``$PATH`` / the standard
    macOS + Linux install locations — not just ``$PATH`` — so MATLAB
    installed in ``/Applications`` is found even when it isn't on the
    shell path.
    """
    from neural_data_decoding.interop.matlab_runner import matlab_available

    return matlab_available()


def _gpu_available() -> bool:
    """Return True iff a CUDA-capable device is visible to PyTorch."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip ``needs_matlab`` / ``needs_gpu`` tests when prerequisites are missing.

    Parameters
    ----------
    config
        Pytest config object (unused, but required by the hook signature).
    items
        Collected test items; tests with the relevant marker are skipped in place.
    """
    skip_no_matlab = pytest.mark.skip(reason="MATLAB not installed on this system")
    skip_no_gpu = pytest.mark.skip(reason="No CUDA-capable GPU available")

    matlab_ok = _matlab_available()
    gpu_ok = _gpu_available()

    for item in items:
        if "needs_matlab" in item.keywords and not matlab_ok:
            item.add_marker(skip_no_matlab)
        if "needs_gpu" in item.keywords and not gpu_ok:
            item.add_marker(skip_no_gpu)
