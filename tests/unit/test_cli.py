"""CLI subcommand dispatch + exit-code tests.

Covers the previously-untested CLI surface: device resolution, the
``check-existing`` pre-flight, and the ``train`` clobber-abort path
(Critical Note #22 — training must refuse to overwrite an existing run's
checkpoints unless ``--force`` is passed).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from neural_data_decoding.cli import _resolve_device, main


def _mps_available() -> bool:
    """Return whether the Apple MPS backend is present and available."""
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def test_resolve_device_explicit_cpu() -> None:
    """An explicit ``cpu`` request resolves to the CPU device."""
    assert _resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_prefers_cuda_else_cpu() -> None:
    """``auto`` selects CUDA when available, else CPU (never MPS)."""
    resolved = _resolve_device("auto")
    assert resolved.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_cuda_falls_back_when_unavailable() -> None:
    """An explicit ``cuda`` request degrades to CPU when CUDA is absent."""
    assert _resolve_device("cuda").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_mps_is_opt_in() -> None:
    """``mps`` is honored explicitly when available, else falls back to CPU."""
    assert _resolve_device("mps").type == ("mps" if _mps_available() else "cpu")


def test_check_existing_reports_absence_then_presence(tmp_path: Path, capsys) -> None:
    """``check-existing`` returns 0 with no checkpoint, 1 once one exists."""
    argv = [
        "check-existing",
        "--config-name", "A_logistic_synthetic",
        "--fold", "1",
        "--output-root", str(tmp_path),
    ]
    assert main(argv) == 0
    result_dir = Path(json.loads(capsys.readouterr().out)["result_dir"])

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "optimal_state.pt").write_bytes(b"")
    assert main(argv) == 1


def test_train_aborts_on_existing_checkpoint_without_force(tmp_path: Path, capsys) -> None:
    """``train`` returns exit code 2 (not clobbering) when a checkpoint exists."""
    common = [
        "--config-name", "A_logistic_synthetic",
        "--fold", "1",
        "--output-root", str(tmp_path),
    ]
    # Resolve the same result dir `train` would use, and plant a checkpoint.
    assert main(["check-existing", *common]) == 0
    result_dir = Path(json.loads(capsys.readouterr().out)["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "optimal_state.pt").write_bytes(b"")

    rc = main(["train", *common])
    assert rc == 2
    assert "existing checkpoint" in capsys.readouterr().err.lower()
