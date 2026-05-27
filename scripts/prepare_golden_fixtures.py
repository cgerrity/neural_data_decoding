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

    Returns
    -------
    str
        Absolute path to the ``matlab`` executable.

    Raises
    ------
    SystemExit
        If MATLAB is not installed or not on ``$PATH``.
    """
    matlab = shutil.which("matlab")
    if matlab is None:
        sys.exit(
            "MATLAB not found on $PATH. Install MATLAB or add it to your "
            "PATH, then re-run this script."
        )
    return matlab


def _run_matlab_batch(commands: str) -> None:
    """Run a sequence of MATLAB commands in batch mode.

    Parameters
    ----------
    commands
        MATLAB statements to execute (e.g. ``"addpath(...); doStuff(...);"``).
    """
    matlab = _ensure_matlab()
    subprocess.run(
        [matlab, "-batch", commands],
        check=True,
        cwd=str(MATLAB_SOURCE_ROOT),
    )


def prepare_milestone_0() -> None:
    """Generate minimal smoke fixtures (verifies the MATLAB toolchain works)."""
    print("[Milestone 0] Smoke fixture preparation (stub).")
    # TODO(implementer): once a MATLAB-side fixture-export function exists, call it here.


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
