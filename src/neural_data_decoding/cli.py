"""Command-line entry point for the neural_data_decoding pipeline.

Currently a stub. Will be expanded in Milestone A to dispatch ``train``,
``check-existing``, ``sweep``, and other subcommands based on a
Hydra-composed config.

Examples
--------
After ``pip install -e .``, the package can be invoked as a module::

    python -m neural_data_decoding --help

or via the installed console-script entry point::

    neural_data_decoding --help
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested subcommand.

    Parameters
    ----------
    argv
        Optional list of arguments. Defaults to :data:`sys.argv` when ``None``.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``. ``0`` indicates success.

    Notes
    -----
    This is a Milestone 0 stub; the subcommand surface area is intentionally
    minimal. Milestone A will wire in ``train``, ``check-existing``, and
    related subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="neural_data_decoding",
        description=(
            "Python port of the MATLAB neural decoding pipeline. "
            "Run subcommands to train, check existing checkpoints, or launch sweeps."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Subcommand stubs — populated in Milestone A.
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "train",
        help="(stub — Milestone A) Run a training session with the given Hydra config.",
    )
    subparsers.add_parser(
        "check-existing",
        help="(stub — Milestone A) Verify no existing checkpoints would be overwritten.",
    )
    subparsers.add_parser(
        "sweep",
        help="(stub — Milestone D) Launch a hyperparameter sweep via submitit or Ray Tune.",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    print(f"Subcommand '{args.command}' is not yet implemented (Milestone 0 stub).")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
