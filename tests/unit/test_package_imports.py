"""Smoke tests for the package install — verifies every subpackage imports cleanly.

These tests are the first line of defense against import-time errors and broken
``__init__.py`` declarations. They run on every PR and should always pass.
"""

from __future__ import annotations

import importlib

import pytest

SUBPACKAGES = [
    "neural_data_decoding",
    "neural_data_decoding.config",
    "neural_data_decoding.data",
    "neural_data_decoding.models",
    "neural_data_decoding.models.layers",
    "neural_data_decoding.models.stitching_fusion",
    "neural_data_decoding.training",
    "neural_data_decoding.training.losses",
    "neural_data_decoding.training.schedules",
    "neural_data_decoding.training.monitoring",
    "neural_data_decoding.interop",
    "neural_data_decoding.sweeps",
    "neural_data_decoding.utils",
    "neural_data_decoding.cli",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    """Every declared subpackage must import without error."""
    module = importlib.import_module(name)
    assert module is not None
    assert module.__doc__, f"{name} is missing a module docstring"


def test_version_is_set() -> None:
    """The top-level package exposes a non-empty ``__version__`` string."""
    import neural_data_decoding

    assert isinstance(neural_data_decoding.__version__, str)
    assert len(neural_data_decoding.__version__) > 0


def test_cli_help_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI ``main()`` entry point produces help text when called with no args."""
    from neural_data_decoding.cli import main

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "neural_data_decoding" in captured.out
