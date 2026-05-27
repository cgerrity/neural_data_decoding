"""Tests for :mod:`neural_data_decoding.utils.paths`."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_data_decoding.utils.paths import (
    BasePaths,
    RuntimeEnvironment,
    detect_environment,
    get_base_paths,
)


def test_runtime_environment_values() -> None:
    """All expected environment values are present."""
    assert {e.value for e in RuntimeEnvironment} == {"local", "teba", "accre"}


def test_get_base_paths_returns_basepaths_dataclass() -> None:
    """The function returns a :class:`BasePaths` instance with Path-typed members."""
    paths = get_base_paths()
    assert isinstance(paths, BasePaths)
    assert isinstance(paths.input, Path)
    assert isinstance(paths.output, Path)
    assert isinstance(paths.temporary, Path)
    assert isinstance(paths.environment, RuntimeEnvironment)


@pytest.mark.parametrize(
    "env",
    [RuntimeEnvironment.LOCAL, RuntimeEnvironment.TEBA, RuntimeEnvironment.ACCRE],
)
def test_forced_environment_is_returned(env: RuntimeEnvironment) -> None:
    """Passing ``environment=`` forces that branch regardless of host detection."""
    paths = get_base_paths(environment=env)
    assert paths.environment is env


def test_accre_paths_use_user_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCRE paths route through ``/home/$USER`` and ``/data/womelsdorf_lab/$USER``."""
    monkeypatch.setenv("USER", "testuser")
    paths = get_base_paths(environment=RuntimeEnvironment.ACCRE)
    assert paths.input == Path("/home/testuser")
    assert paths.output == Path("/home/testuser")
    assert paths.temporary == Path("/data/womelsdorf_lab/testuser")


def test_teba_paths_use_user_subdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """TEBA paths route through ``/data`` and ``/data/users/$USER``."""
    monkeypatch.setenv("USER", "testuser")
    paths = get_base_paths(environment=RuntimeEnvironment.TEBA)
    assert paths.input == Path("/data")
    assert paths.output == Path("/data/users/testuser")
    assert paths.temporary == Path("/data/users/testuser")


def test_ndd_force_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``NDD_FORCE_ENV`` env var overrides auto-detection."""
    monkeypatch.setenv("NDD_FORCE_ENV", "accre")
    assert detect_environment() is RuntimeEnvironment.ACCRE
    monkeypatch.setenv("NDD_FORCE_ENV", "TEBA")  # case-insensitive
    assert detect_environment() is RuntimeEnvironment.TEBA


def test_ndd_force_env_invalid_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown ``NDD_FORCE_ENV`` value produces a clear error."""
    monkeypatch.setenv("NDD_FORCE_ENV", "saturn")
    with pytest.raises(ValueError, match="saturn"):
        detect_environment()


def test_basepaths_is_immutable() -> None:
    """:class:`BasePaths` is a frozen dataclass — assignment raises."""
    paths = get_base_paths(environment=RuntimeEnvironment.ACCRE)
    with pytest.raises((AttributeError, Exception)):
        paths.input = Path("/somewhere/else")  # type: ignore[misc]
