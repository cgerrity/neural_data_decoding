"""Python port of the MATLAB neural decoding pipeline.

This package reproduces the active production path of the MATLAB pipeline in
``Processing_Functions_cgg/`` using modern PyTorch patterns, while writing
``.mat``-compatible output where MATLAB-side analysis still consumes it.

The package is organized into the following subpackages:

* :mod:`neural_data_decoding.config` — typed configuration dataclasses
* :mod:`neural_data_decoding.data` — datasets, samplers, stratification, normalization
* :mod:`neural_data_decoding.models` — encoder / decoder / classifier builders + custom layers
* :mod:`neural_data_decoding.training` — training loop, losses, schedules, monitoring
* :mod:`neural_data_decoding.interop` — MATLAB ↔ Python bridge (``.mat`` writers, folder hierarchy)
* :mod:`neural_data_decoding.sweeps` — SLURM / Ray Tune hyperparameter sweep launchers
* :mod:`neural_data_decoding.utils` — environment detection, seeding, shape converters

See ``Plans/neural_data_decoding_plan.md`` in the parent repository for the full
migration plan, including parity goals, milestone sequence, and the list of
known MATLAB quirks that must be preserved exactly.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
