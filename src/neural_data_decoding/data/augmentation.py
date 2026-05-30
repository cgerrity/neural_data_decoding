"""On-the-fly data augmentation for neural-data Datasets.

Ports the augmentation kernels from ``cgg_generateDataAugmentationSignal.m`` and
the time-shift logic at lines 240–253 of ``cgg_loadDataArray.m``. The pipeline
applies four augmentation types per trial:

1. **Channel offset** — a per-channel constant offset drawn from
   :math:`\\mathcal{N}(0, \\sigma_{co})`, broadcast across all samples.
2. **White noise** — independent Gaussian noise drawn from
   :math:`\\mathcal{N}(0, \\sigma_{wn})` at every channel-sample-probe cell.
3. **Random walk** — a smooth low-frequency drift, generated as the cumulative
   sum along the time axis of small Gaussian steps with std :math:`\\sigma_{rw}`.
4. **Time shift** — temporal jitter applied during windowing.

The first three are summed and added to the input signal, then optionally
smoothed with a Gaussian kernel (replicating MATLAB's ``smoothdata(...,
"gaussian", 50)`` low-pass step). The time-shift is applied separately
during window extraction because it indexes into the source signal.

**Critical: per-call re-randomization.**  Augmentation values are re-drawn
on every call. The MATLAB pipeline achieves this implicitly via
``fileDatastore`` re-invoking the read function each access; the Python
equivalent must explicitly re-roll noise on every ``Dataset.__getitem__``.
See Critical Note #7 in the migration plan.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> data = np.zeros((4, 100, 2))
>>> noise = additive_augmentation_signal(
...     shape=data.shape,
...     std_channel_offset=0.1,
...     std_white_noise=0.05,
...     std_random_walk=0.01,
...     rng=rng,
... )
>>> noise.shape
(4, 100, 2)
"""

from __future__ import annotations

import math
from typing import TypeGuard

import numpy as np
from scipy.ndimage import gaussian_filter1d

# MATLAB's ``smoothdata(..., "gaussian", w)`` uses a Gaussian kernel whose
# standard deviation is approximately ``w/5`` (so the 5-sigma window matches
# the requested window length). We use the same heuristic so the Python
# smoothing matches MATLAB to within numerical tolerance.
_MATLAB_GAUSSIAN_SIGMA_RATIO = 1.0 / 5.0
_DEFAULT_SMOOTH_WINDOW = 50


def additive_augmentation_signal(
    shape: tuple[int, int, int],
    *,
    std_channel_offset: float | None,
    std_white_noise: float | None,
    std_random_walk: float | None,
    rng: np.random.Generator,
    smooth_window: int = _DEFAULT_SMOOTH_WINDOW,
    want_low_pass: bool = True,
) -> np.ndarray:
    """Generate a per-trial additive augmentation tensor.

    Mirrors ``cgg_generateDataAugmentationSignal.m``. The output is shape-
    matched to ``shape`` and is intended to be **added** to the raw signal.

    Each component is independently re-drawn from ``rng`` on every call,
    matching MATLAB's per-read behavior. A component with ``None`` or NaN
    std is omitted from the sum (treated as zero).

    Parameters
    ----------
    shape
        Target tensor shape ``(num_channels, num_samples, num_probes)``.
    std_channel_offset
        Standard deviation of the per-channel constant offset. ``None`` or
        ``NaN`` to disable.
    std_white_noise
        Standard deviation of the per-sample white noise. ``None`` or
        ``NaN`` to disable.
    std_random_walk
        Standard deviation of the step size for the random-walk component.
        The resulting drift is the cumulative sum of these steps along the
        sample axis. ``None`` or ``NaN`` to disable.
    rng
        A :class:`numpy.random.Generator`. The caller is responsible for
        seeding it; pass a freshly-seeded generator if reproducibility is
        required, or a long-lived generator for ordinary training.
    smooth_window
        Width (in samples) of the post-summation Gaussian smoothing kernel.
        The Gaussian's standard deviation is set to ``smooth_window / 5``
        to match MATLAB's ``smoothdata`` convention. Defaults to 50.
    want_low_pass
        If ``True`` (default, matching MATLAB), apply Gaussian smoothing to
        the summed signal. If ``False``, return the unsmoothed sum.

    Returns
    -------
    numpy.ndarray
        Augmentation tensor of shape ``shape`` and dtype ``float64``.
    """
    if len(shape) != 3:
        raise ValueError(
            f"shape must be (channels, samples, probes); got {shape!r}."
        )
    num_channels, num_samples, num_probes = shape

    signal = np.zeros(shape, dtype=np.float64)

    if _enabled(std_channel_offset):
        # Per-channel, per-probe constant offset → broadcast across samples.
        offset = rng.standard_normal((num_channels, 1, num_probes)) * std_channel_offset
        signal = signal + np.broadcast_to(offset, shape)

    if _enabled(std_white_noise):
        signal = signal + rng.standard_normal(shape) * std_white_noise

    if _enabled(std_random_walk):
        # Step-wise Gaussian noise, integrated along the sample axis.
        steps = rng.standard_normal(shape) * std_random_walk
        signal = signal + np.cumsum(steps, axis=1)

    # The MATLAB code passes the summed signal through a Gaussian low-pass
    # smoother. The smoothing window is in *samples*, not seconds; sigma is
    # window/5 to match MATLAB's smoothdata convention.
    if want_low_pass and signal.size > 1:
        sigma = max(smooth_window * _MATLAB_GAUSSIAN_SIGMA_RATIO, 1e-6)
        signal = gaussian_filter1d(signal, sigma=sigma, axis=1, mode="reflect")

    return signal


def generate_time_shift_samples(
    *,
    num_channels: int,
    num_probes: int,
    num_windows: int,
    std_time_shift: float | None,
    sampling_frequency: float,
    want_separate: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample per-trial time-shift offsets, in integer samples.

    Mirrors lines 240–253 of ``cgg_loadDataArray.m``. The shift is drawn
    uniformly from :math:`[-\\sigma_{ts}, +\\sigma_{ts}]` (where
    :math:`\\sigma_{ts}` is in milliseconds at 1 kHz; see Notes for the
    sample-rate conversion).

    Parameters
    ----------
    num_channels, num_probes, num_windows
        Dimensions over which a shift is sampled. When ``want_separate`` is
        true each ``(channel, probe, window)`` cell gets its own shift;
        otherwise a single scalar shift is broadcast across all cells.
    std_time_shift
        Half-width (in ms-at-1-kHz) of the uniform distribution. ``None``
        or ``NaN`` disables the augmentation and returns an all-zero array.
    sampling_frequency
        Sampling rate of the data in Hz. Used to convert the ms-based
        ``std_time_shift`` into a sample-count shift.
    want_separate
        Whether each ``(channel, probe, window)`` gets an independent shift
        (mirrors ``WantSeparateTimeShift=true``) or shares one shift across
        all cells.
    rng
        Random generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Integer-sample shifts of shape ``(num_channels, num_probes, num_windows)``.

    Notes
    -----
    The MATLAB conversion is::

        TimeShiftIDX = round((1 / SamplingFrequency) * (1000 * TimeShift))

    which simplifies to ``round(TimeShift * 1000 / SamplingFrequency)``.
    At ``SamplingFrequency = 1000`` Hz, a ``std_time_shift = 100`` yields
    shifts uniformly in ``[-100, +100]`` samples (= ±100 ms).
    """
    out_shape = (num_channels, num_probes, num_windows)

    if not _enabled(std_time_shift):
        return np.zeros(out_shape, dtype=np.int64)

    if want_separate:
        shifts_ms = rng.uniform(-std_time_shift, std_time_shift, size=out_shape)
    else:
        scalar = rng.uniform(-std_time_shift, std_time_shift)
        shifts_ms = np.full(out_shape, scalar, dtype=np.float64)

    shifts_samples = np.round(shifts_ms * 1000.0 / sampling_frequency).astype(np.int64)
    return shifts_samples


def _enabled(std: float | None) -> TypeGuard[float]:
    """Return ``True`` iff ``std`` is a usable positive number.

    Treats ``None`` and ``NaN`` as disabled (matches the MATLAB
    ``~isnan(STD...)`` checks). Negative values are also rejected because a
    negative standard deviation has no meaning; the MATLAB code does not
    range-check explicitly, but the Python implementation does to catch
    obvious config errors early.

    The return is annotated as :class:`typing.TypeGuard` so that pyright /
    Pylance narrow ``std`` from ``float | None`` to ``float`` in the True
    branch — saving callers from redundant ``assert std is not None``.
    """
    if std is None:
        return False
    try:
        v = float(std)
    except (TypeError, ValueError):
        return False
    if math.isnan(v):
        return False
    if v <= 0:
        return False
    return True


__all__ = [
    "additive_augmentation_signal",
    "generate_time_shift_samples",
]
