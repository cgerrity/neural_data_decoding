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
import sys
from pathlib import Path

from neural_data_decoding.interop.matlab_runner import (
    MatlabNotFoundError,
    run_matlab_batch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]   # …/Neural Data Reading/
PIPELINE_ROOT = REPO_ROOT / "neural_data_decoding"
MATLAB_SOURCE_ROOT = REPO_ROOT / "Processing_Functions_cgg"
FIXTURE_ROOT = PIPELINE_ROOT / "tests" / "fixtures"


def _run_matlab_batch(commands: str) -> None:
    """Run a sequence of MATLAB commands in batch mode from the repo root.

    Thin wrapper over
    :func:`neural_data_decoding.interop.matlab_runner.run_matlab_batch`
    that pins ``cwd`` to the parent repo root (so invoked scripts can
    resolve relative paths to ``Processing_Functions_cgg`` and sibling
    utility folders) and converts a missing-MATLAB error into a clean
    ``SystemExit`` for CLI use.

    Parameters
    ----------
    commands
        MATLAB statements to execute (e.g. ``"addpath(...); doStuff(...);"``).
    """
    try:
        run_matlab_batch(commands, cwd=REPO_ROOT)
    except MatlabNotFoundError as exc:
        sys.exit(str(exc))


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
