"""CI helper for the T4 round-trip parity gate.

Confirms that ``.mat`` files written by the Python pipeline load cleanly in
MATLAB and produce the expected aggregate when passed to the downstream
analysis scripts.

This is a Milestone 0 stub. The actual implementation lands in Milestone A
once the Python pipeline first writes ``CM_Table.mat`` files.

Usage
-----
::

    python scripts/verify_mat_roundtrip.py --input-dir <python_output_dir>
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Verify that Python-generated ``.mat`` files round-trip through MATLAB."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing Python-generated CM_Table.mat files.",
    )
    args = parser.parse_args(argv)

    print(f"[stub] Would run MATLAB DATA_cggAllNetworkEncoderResults on {args.input_dir}")
    print("       This script is a Milestone 0 stub; implemented in Milestone A.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
