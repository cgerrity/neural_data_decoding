"""Piecewise-anneal interpolator for curriculum schedules.

This is the single mathematical primitive shared by every dynamic schedule
(load / weight / freeze). Direct port of MATLAB's ``cgg_calculateDynamicValue``
composed with its inner helper ``cgg_annealWeight``.

The behavior is piecewise-linear with a one-epoch boundary quirk:

* **Before the first waypoint** (``epoch <= epoch_points[0]``): returns
  ``base * magnitude_points[0]`` (left-clamp).
* **After the last waypoint** (``epoch > epoch_points[-1]``): returns
  ``base * magnitude_points[-1]`` (right-clamp).
* **In-segment**: identifies the segment
  ``[epoch_points[i], epoch_points[i+1]]`` containing the current epoch,
  then linearly ramps from ``base * magnitude_points[i]`` toward
  ``base * magnitude_points[i+1]``.

MATLAB parity quirk
-------------------

``cgg_annealWeight`` uses ``Epoch - 1 - WeightDelayEpoch`` for the ramp
position, with the segment span as the denominator. At every internal
waypoint this introduces a small discontinuity: the in-segment ramp at
``epoch == epoch_points[i+1]`` reaches only ``(span - 1) / span`` of the
way to the end magnitude, and at ``epoch == epoch_points[i+1] + 1`` the
next segment's leading edge (or the right-clamp) snaps the value to
``magnitude_points[i+1]``. This quirk is preserved exactly; the parity
test pins the discontinuity at multiple waypoints.

References
----------
* ``$NDD_MATLAB_SOURCE_ROOT/Processing_Functions_cgg/Helper Classes/cgg_calculateDynamicValue.m``
* ``$NDD_MATLAB_SOURCE_ROOT/Processing_Functions_cgg/ANN Functions/Training Functions/cgg_annealWeight.m``
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def piecewise_anneal_value(
    base: float,
    epoch_points: Sequence[float],
    magnitude_points: Sequence[float],
    epoch: int,
) -> float:
    """Compute the current per-epoch value for a single scheduled parameter.

    Parameters
    ----------
    base
        Static base value the magnitudes are multiplied against.
    epoch_points
        Monotonically-increasing waypoint epochs (1-indexed to match
        MATLAB). May be empty, in which case the result is ``base``.
    magnitude_points
        Multipliers at each waypoint; must have the same length as
        ``epoch_points``.
    epoch
        The current training epoch (1-indexed, matching MATLAB).

    Returns
    -------
    float
        ``base * piecewise_magnitude(epoch)``. NaN if ``base`` is NaN.

    Raises
    ------
    ValueError
        If ``epoch_points`` and ``magnitude_points`` differ in length.
    """
    if math.isnan(base):
        return float("nan")

    n = len(epoch_points)
    if n != len(magnitude_points):
        raise ValueError(
            f"epoch_points (len={n}) and magnitude_points "
            f"(len={len(magnitude_points)}) must have the same length."
        )
    if n == 0:
        return float(base)

    if epoch <= epoch_points[0]:
        return float(base * magnitude_points[0])
    if epoch > epoch_points[-1]:
        return float(base * magnitude_points[-1])

    # Largest idx with epoch_points[idx] < epoch (mirrors MATLAB
    # ``find(EpochPoints < Epoch, 1, 'last')``). Iterating from the end
    # makes the equivalence to ``'last'`` self-evident.
    idx = 0
    for i in range(n - 1, -1, -1):
        if epoch_points[i] < epoch:
            idx = i
            break

    min_w = base * magnitude_points[idx]
    max_w = base * magnitude_points[idx + 1]
    delay = epoch_points[idx]
    ramp_len = epoch_points[idx + 1] - epoch_points[idx]
    target_diff = max_w - min_w

    # Inline cgg_annealWeight with the (epoch - 1) off-by-one preserved.
    if ramp_len <= 0 or math.isnan(ramp_len):
        ramp = target_diff
    elif epoch <= delay:
        ramp = 0.0
    elif epoch <= delay + ramp_len:
        ramp = (epoch - 1 - delay) * (target_diff / ramp_len)
    else:
        ramp = target_diff

    return float(min_w + ramp)
