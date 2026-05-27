"""Regenerate the MATLAB-side reference fixtures used by parity tests.

This script runs MATLAB in batch mode to produce the ``.mat`` files that
``tests/parity/`` reads as golden references. It is intended to be run
**once** after a relevant change in the MATLAB pipeline (e.g., new loss
kernel, new architecture) so the Python parity tests have something to
compare against.

The generated fixtures live in ``tests/fixtures/{golden_weights,
golden_batches, reference_partitions, reference_cm_tables}/`` and are
**gitignored** — they must be regenerated locally before running the
parity test suite.

This is a Milestone 0 stub. The actual MATLAB-batch invocation logic
will be filled in as each milestone introduces parity tests that need
new fixtures.

Usage
-----
::

    python scripts/prepare_golden_fixtures.py --milestone 0   # minimal smoke fixtures
    python scripts/prepare_golden_fixtures.py --milestone A   # logistic regression run
    python scripts/prepare_golden_fixtures.py --milestone B   # GRU + classifier
    python scripts/prepare_golden_fixtures.py --milestone C   # full Optimal (slow)
    python scripts/prepare_golden_fixtures.py --all           # everything
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]   # …/Neural Data Reading/
PIPELINE_ROOT = REPO_ROOT / "neural_data_decoding"
MATLAB_SOURCE_ROOT = REPO_ROOT / "Processing_Functions_cgg"
FIXTURE_ROOT = PIPELINE_ROOT / "tests" / "fixtures"


def _ensure_matlab() -> str:
    """Locate the MATLAB executable; abort if unavailable.

    Search order:

    1. The ``MATLAB_EXECUTABLE`` environment variable, if set.
    2. ``$PATH`` lookup for ``matlab``.
    3. The standard macOS install location
       ``/Applications/MATLAB_R*.app/bin/matlab`` (newest version wins).
    4. The standard Linux install location ``/usr/local/MATLAB/R*/bin/matlab``.

    Returns
    -------
    str
        Absolute path to the ``matlab`` executable.

    Raises
    ------
    SystemExit
        If MATLAB cannot be located anywhere on the system.
    """
    # 1. Explicit override.
    env_path = os.environ.get("MATLAB_EXECUTABLE")
    if env_path:
        if not Path(env_path).is_file():
            sys.exit(
                f"MATLAB_EXECUTABLE points to a missing file: {env_path}"
            )
        return env_path

    # 2. PATH lookup.
    on_path = shutil.which("matlab")
    if on_path:
        return on_path

    # 3. macOS standard locations (newest version preferred).
    mac_candidates = sorted(
        Path("/Applications").glob("MATLAB_R*.app/bin/matlab"),
        key=lambda p: p.parent.parent.name,  # MATLAB_R2025b > MATLAB_R2024b
        reverse=True,
    )
    if mac_candidates:
        return str(mac_candidates[0])

    # 4. Linux standard location.
    linux_candidates = sorted(
        Path("/usr/local/MATLAB").glob("R*/bin/matlab"),
        key=lambda p: p.parent.parent.name,
        reverse=True,
    )
    if linux_candidates:
        return str(linux_candidates[0])

    sys.exit(
        "MATLAB not found. Looked in:\n"
        "  - $MATLAB_EXECUTABLE\n"
        "  - $PATH\n"
        "  - /Applications/MATLAB_R*.app/bin/matlab (macOS)\n"
        "  - /usr/local/MATLAB/R*/bin/matlab (Linux)\n"
        "Set MATLAB_EXECUTABLE to the absolute path of your matlab binary, "
        "or add it to $PATH."
    )


def _run_matlab_batch(commands: str) -> None:
    """Run a sequence of MATLAB commands in batch mode.

    The MATLAB working directory is set to the parent repo root so the
    invoked scripts can resolve relative paths to ``Processing_Functions_cgg``
    and the other sibling utility folders. Per-script ``addpath`` calls
    handle their own dependencies.

    On Apple Silicon, MATLAB's launcher script can misidentify the
    architecture as ``maci64`` (Intel) when invoked from a non-interactive
    Python subprocess. We work around this by wrapping the call in
    ``arch -arm64`` when running on Darwin/arm64.

    Parameters
    ----------
    commands
        MATLAB statements to execute (e.g. ``"addpath(...); doStuff(...);"``).
    """
    matlab = _ensure_matlab()
    cmd: list[str] = [matlab, "-batch", commands]

    # Apple Silicon workaround: when the Python interpreter is itself
    # running under Rosetta (common with MacPorts/x86_64 builds), the
    # launcher script for MATLAB picks up the wrong ARCH from its
    # parent process and looks for ``bin/maci64/`` instead of
    # ``bin/maca64/``. We detect the *real* hardware arch via ``sysctl``
    # rather than ``os.uname()`` (which Rosetta lies about) and prefix
    # the call with ``arch -arm64`` to force the launcher into the
    # right code path.
    if sys.platform == "darwin" and _real_macos_arch_is_arm64():
        cmd = ["arch", "-arm64", *cmd]

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _real_macos_arch_is_arm64() -> bool:
    """Return ``True`` iff the *hardware* is Apple Silicon, ignoring Rosetta.

    Uses ``sysctl hw.optional.arm64`` because ``os.uname().machine`` reports
    ``x86_64`` for a Rosetta-translated Python — a misleading answer when
    we need to know what the launcher's child processes will see natively.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return result.stdout.strip() == "1"


def prepare_milestone_0() -> None:
    """Generate the Milestone-0 stratification reference fixture.

    Runs ``scripts/generate_stratification_fixture.m`` in MATLAB batch mode.
    The output is written to
    ``tests/fixtures/reference_partitions/synthetic_easy_partition.mat``
    and is consumed by ``tests/parity/test_stratification_parity.py``.

    Raises
    ------
    SystemExit
        If MATLAB is not on ``$PATH`` (handled by :func:`_ensure_matlab`).
    subprocess.CalledProcessError
        If the MATLAB script itself errors out.
    """
    print("[Milestone 0] Generating stratification reference fixture …")
    script = PIPELINE_ROOT / "scripts" / "generate_stratification_fixture.m"
    if not script.is_file():
        sys.exit(f"Missing fixture-generator script: {script}")

    expected_output = (
        FIXTURE_ROOT
        / "reference_partitions"
        / "synthetic_easy_partition.mat"
    )
    expected_output.parent.mkdir(parents=True, exist_ok=True)

    # The MATLAB script is a function file; we invoke it via -batch so MATLAB
    # exits cleanly when done. The function adds the parent repo's MATLAB
    # source folders to its own path; we just need to make sure MATLAB can
    # find the function file itself, which we do with `cd` + `addpath`.
    matlab_cmd = (
        f"addpath('{script.parent}'); "
        f"{script.stem}(); "
        f"exit(0);"
    )
    _run_matlab_batch(matlab_cmd)

    if not expected_output.is_file():
        sys.exit(
            f"MATLAB completed but the expected fixture was not produced: "
            f"{expected_output}"
        )
    print(f"  ✓ Fixture saved: {expected_output}")


def prepare_milestone_a() -> None:
    """Generate parity fixtures for the Logistic Regression tracer milestone."""
    print("[Milestone A] Logistic regression fixture preparation (stub).")
    # TODO(implementer): wire to a MATLAB script that:
    #   1. Runs cgg_runAutoEncoder(1, 'ModelName', 'Logistic Regression', 'Epoch', 'Synthetic_Easy')
    #   2. Saves the resulting CM_Table, weights, and per-iteration state into
    #      tests/fixtures/{golden_weights,reference_cm_tables}/milestone_a/


def prepare_milestone_b() -> None:
    """Generate parity fixtures for the GRU + Classifier milestone."""
    print("[Milestone B] GRU + Classifier fixture preparation (stub).")


def prepare_milestone_c() -> None:
    """Generate parity fixtures for the Full Optimal milestone."""
    print("[Milestone C] Full Optimal fixture preparation (stub).")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested milestone preparation step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--milestone",
        choices=["0", "A", "B", "C"],
        help="Generate fixtures for this milestone only.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate fixtures for every milestone (slow; useful for CI).",
    )
    args = parser.parse_args(argv)

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)

    if args.all:
        prepare_milestone_0()
        prepare_milestone_a()
        prepare_milestone_b()
        prepare_milestone_c()
        return 0

    dispatch = {
        "0": prepare_milestone_0,
        "A": prepare_milestone_a,
        "B": prepare_milestone_b,
        "C": prepare_milestone_c,
    }
    if args.milestone is None:
        parser.print_help()
        return 0

    dispatch[args.milestone]()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
