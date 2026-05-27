"""Per-trial signal normalization, ported from ``cgg_selectNormalization.m``.

The MATLAB pipeline supports a long menu of normalization recipes named by
string (e.g. ``'Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered
- Range 0.5'``). Each recipe takes the raw per-trial signal plus a
precomputed ``NormalizationTable`` (per-channel statistics over the training
set) and returns a normalized tensor.

The recipe space is implemented as a **registry** — Milestone 0 fully wires up
the Optimal recipe and the passthrough ``'None'``; the remaining recipes are
registered as ``NotImplementedError`` stubs so callers get clear errors if
they request an unsupported variant. Milestone CC fills in the rest.

The ``NormalizationTable`` is modeled as a :class:`pandas.DataFrame` with the
columns from MATLAB's table: ``Area`` (1-indexed), ``Channel`` (1-indexed),
``Mean``, ``STD``, ``Max``, ``Min``.

Examples
--------
>>> import pandas as pd
>>> table = pd.DataFrame({
...     "Area": [1, 1], "Channel": [1, 2],
...     "Mean": [0.0, 0.0], "STD": [1.0, 1.0],
...     "Min": [-3.0, -3.0], "Max": [3.0, 3.0],
... })
>>> import numpy as np
>>> x = np.zeros((2, 10, 1))  # (channels, samples, areas)
>>> y = select_normalization(x, table, "None")
>>> y.shape
(2, 10, 1)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd

NormalizationFn = Callable[[np.ndarray, pd.DataFrame], np.ndarray]

# Registry mapping recipe-name → implementation. New recipes register via the
# ``@register`` decorator below.
_REGISTRY: dict[str, NormalizationFn] = {}


def register(name: str) -> Callable[[NormalizationFn], NormalizationFn]:
    """Decorator that registers ``fn`` under recipe ``name``.

    Parameters
    ----------
    name
        The exact MATLAB recipe string (e.g. ``'None'``,
        ``'Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered - Range 0.5'``).

    Returns
    -------
    Callable
        A decorator that registers the wrapped function and returns it
        unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """

    def _decorator(fn: NormalizationFn) -> NormalizationFn:
        if name in _REGISTRY:
            raise ValueError(f"Normalization recipe '{name}' is already registered.")
        _REGISTRY[name] = fn
        return fn

    return _decorator


def select_normalization(
    data: np.ndarray,
    normalization_table: pd.DataFrame | None,
    recipe: str,
) -> np.ndarray:
    """Apply a named normalization recipe to a per-trial tensor.

    Mirrors ``cgg_selectNormalization.m``: if ``normalization_table`` is
    ``None`` or contains no rows, the recipe silently degrades to ``'None'``
    (passthrough), regardless of the requested ``recipe``.

    Parameters
    ----------
    data
        Per-trial signal tensor with shape ``(channels, samples, areas)``.
    normalization_table
        DataFrame with columns ``Area``, ``Channel``, ``Mean``, ``STD``,
        ``Max``, ``Min``. ``Area`` and ``Channel`` are 1-indexed to match
        MATLAB. ``None`` is allowed and produces a passthrough.
    recipe
        The recipe name. Must be a key in the registry (see
        :func:`list_recipes`).

    Returns
    -------
    numpy.ndarray
        Normalized signal with the same shape as ``data``.

    Raises
    ------
    KeyError
        If ``recipe`` is not registered.
    NotImplementedError
        If the recipe is registered as a stub that has not been implemented
        yet.
    """
    if normalization_table is None or len(normalization_table) == 0:
        recipe = "None"

    try:
        fn = _REGISTRY[recipe]
    except KeyError as exc:
        registered = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Normalization recipe '{recipe}' is not registered. "
            f"Known recipes: {registered}"
        ) from exc

    return fn(data, normalization_table if normalization_table is not None else pd.DataFrame())


def list_recipes() -> list[str]:
    """Return the list of registered recipe names.

    Returns
    -------
    list of str
        Sorted recipe names.
    """
    return sorted(_REGISTRY.keys())


# ───────────────────────── Registered recipes ─────────────────────────


@register("None")
def _passthrough(data: np.ndarray, _table: pd.DataFrame) -> np.ndarray:
    """Identity transform — used when no normalization is configured."""
    return data


@register("Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered - Range 0.5")
def _channel_zscore_global_minmax_neg1_to_1_zero_centered_range05(
    data: np.ndarray, table: pd.DataFrame
) -> np.ndarray:
    """Apply the Optimal recipe (channel z-score → global min/max → [-1, 1]).

    See ``cgg_procNormalizeChannelZScoreGlobalMinMax.m`` for the MATLAB
    reference. Equivalent call there:

    .. code-block:: matlab

        cgg_procNormalizeChannelZScoreGlobalMinMax(
            Data, NormalizationTable, [-1, 1],
            'WantZeroCentered', true,
            'ExpandedRange_Percent', 0.5)

    Parameters
    ----------
    data
        ``(channels, samples, areas)`` tensor for a single trial.
    table
        Normalization-statistics table; must contain
        ``Area``, ``Channel``, ``Mean``, ``STD``, ``Max``, ``Min`` columns.

    Returns
    -------
    numpy.ndarray
        Normalized signal, same shape as ``data``.
    """
    return _channel_zscore_global_minmax(
        data,
        table,
        limits=(-1.0, 1.0),
        want_zero_centered=True,
        expanded_range_percent=0.5,
    )


# ───────────────────────── Stubs for Milestone CC ─────────────────────────

_CC_STUBS: Final[tuple[str, ...]] = (
    "Channel - MinMax - [-1,1]",
    "Channel - MinMax - [0,1]",
    "Area - MinMax - [-1,1]",
    "Area - MinMax - [0,1]",
    "Global - MinMax - [-1,1]",
    "Global - MinMax - [0,1]",
    "Channel - Z-Score",
    "Area - Z-Score",
    "Global - Z-Score",
    "Channel - Z-Score - Global - MinMax - [0,1]",
    "Channel - Z-Score - Global - MinMax - [-1,1]",
    "Channel - Z-Score - Global - MinMax - [0,1] - Zero Centered",
    "Channel - Z-Score - Global - MinMax - [-1,1] - Zero Centered",
    "Channel - Z-Score - Global - MinMax - [0,1] - Zero Centered - Range 0.5",
)


def _make_stub(name: str) -> NormalizationFn:
    """Build a stub recipe that raises ``NotImplementedError`` when invoked."""

    def _stub(_data: np.ndarray, _table: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            f"Normalization recipe '{name}' is registered but not yet implemented. "
            f"Implementation is scheduled for Milestone CC; only the Optimal recipe "
            f"and 'None' are active in Milestone 0."
        )

    return _stub


for _stub_name in _CC_STUBS:
    register(_stub_name)(_make_stub(_stub_name))


# ───────────────────────── Core kernel ─────────────────────────


def _channel_zscore_global_minmax(
    data: np.ndarray,
    table: pd.DataFrame,
    *,
    limits: tuple[float, float],
    want_zero_centered: bool,
    expanded_range_percent: float | None,
) -> np.ndarray:
    """Compute the channel-z-score + global-min/max + linear-remap kernel.

    Ports ``cgg_procNormalizeChannelZScoreGlobalMinMax.m``. The kernel:

    1. Looks up each ``(channel, area)`` pair's ``Mean``/``STD`` from
       ``table`` and Z-scores ``data`` per channel.
    2. Computes global min/max across the Z-scored per-channel min/max.
    3. If ``expanded_range_percent`` is set, scales by a multiplier derived
       from the global STD and the global min/max range. **Note**: the MATLAB
       reference (line 54 of the source) immediately overwrites the parameter
       with ``ExpandedRange_Percent = 1``, so any nonzero value behaves
       identically to ``1``. This Python implementation preserves that
       behavior for parity.
    4. If ``want_zero_centered``, shifts so the midpoint of the global
       min/max range lies at zero before dividing by the range. Otherwise
       does a standard global min/max scaling to ``[0, 1]``.
    5. Linearly remaps from ``[0, 1]`` (or the centered equivalent) into
       ``limits``.

    Parameters
    ----------
    data
        ``(channels, samples, areas)`` per-trial tensor.
    table
        Normalization statistics with ``Area``, ``Channel``, ``Mean``, ``STD``,
        ``Min``, ``Max`` columns.
    limits
        Target output range, e.g. ``(-1.0, 1.0)``.
    want_zero_centered
        Whether to center around the midpoint of the global min/max range
        before scaling.
    expanded_range_percent
        If non-``None``, applies the MATLAB ``ExpandedRange_Percent``
        multiplier step. The parameter value is overwritten to 1.0 inside
        the kernel to match MATLAB.

    Returns
    -------
    numpy.ndarray
        Normalized tensor, same shape as ``data``.
    """
    if data.ndim != 3:
        raise ValueError(
            f"Expected data with shape (channels, samples, areas); got shape {data.shape}."
        )
    num_channels, num_samples, num_areas = data.shape

    required = {"Area", "Channel", "Mean", "STD", "Min", "Max"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"NormalizationTable is missing required columns: {sorted(missing)}"
        )

    # MATLAB is 1-indexed; convert to 0-indexed for numpy.
    channel_idx = table["Channel"].to_numpy().astype(np.int64) - 1
    area_idx = table["Area"].to_numpy().astype(np.int64) - 1
    mean_per_row = table["Mean"].to_numpy(dtype=np.float64)
    std_per_row = table["STD"].to_numpy(dtype=np.float64)
    min_per_row = table["Min"].to_numpy(dtype=np.float64)
    max_per_row = table["Max"].to_numpy(dtype=np.float64)

    # Build (channels, 1, areas) lookup tensors; cells without a table row
    # remain NaN so they propagate cleanly (matches the MATLAB pattern).
    mean_lookup = np.full((num_channels, 1, num_areas), np.nan, dtype=np.float64)
    std_lookup = np.full((num_channels, 1, num_areas), np.nan, dtype=np.float64)
    mean_lookup[channel_idx, 0, area_idx] = mean_per_row
    std_lookup[channel_idx, 0, area_idx] = std_per_row

    # Step 1: per-channel Z-score.
    normalized = (data - mean_lookup) / std_lookup

    # Step 2: global min/max of the Z-scored per-channel extrema.
    max_zscore_per_row = (max_per_row - mean_per_row) / std_per_row
    min_zscore_per_row = (min_per_row - mean_per_row) / std_per_row
    global_max = float(np.max(max_zscore_per_row))
    global_min = float(np.min(min_zscore_per_row))
    global_range = global_max - global_min

    # Step 3: optional expanded-range scaling.
    if expanded_range_percent is not None:
        # Match MATLAB's quirk: the input parameter is immediately overwritten.
        effective_percent = 1.0
        # cgg_calcGlobalMeanSTDFromNormalizationTable returns the population
        # STD across all (channel, area, sample) cells. We replicate it
        # here from the table.
        std_global = _global_std_from_table(table, num_samples)
        limits_range = limits[1] - limits[0]
        if limits_range == 0:
            raise ValueError("Limits must have nonzero range.")
        expanded_target = effective_percent / 2.0 / limits_range
        multiplier = expanded_target / (std_global / global_range) if global_range else 1.0
        normalized = normalized * multiplier

    # Step 4: global min/max scaling, optionally zero-centered.
    if want_zero_centered:
        mean_global = float(np.nanmean(mean_lookup))
        denom = global_max - global_min
        if denom == 0:
            raise ValueError("Global max equals global min; cannot zero-center.")
        normalized = (normalized - (mean_global - global_range / 2.0)) / denom
    else:
        denom = global_max - global_min
        if denom == 0:
            raise ValueError("Global max equals global min; cannot min/max scale.")
        normalized = (normalized - global_min) / denom

    # Step 5: linear remap to `limits`.
    range_norm = limits[1] - limits[0]
    normalized = normalized * range_norm - (range_norm - limits[1])

    return normalized


def _global_std_from_table(table: pd.DataFrame, num_samples: int) -> float:
    """Compute the pooled global STD from the per-(channel, area) table.

    Mirrors ``cgg_calcGlobalMeanSTDFromNormalizationTable.m`` — combines
    per-row variances weighted by the per-row sample count. We treat every
    ``(channel, area)`` cell as contributing ``num_samples`` observations,
    which matches the MATLAB convention.

    Parameters
    ----------
    table
        Normalization-statistics table.
    num_samples
        Number of time samples per channel-area cell.

    Returns
    -------
    float
        The pooled standard deviation.
    """
    means = table["Mean"].to_numpy(dtype=np.float64)
    stds = table["STD"].to_numpy(dtype=np.float64)
    if len(means) == 0:
        return 0.0

    # Total observations across all cells.
    total_n = float(len(means) * num_samples)
    global_mean = float(means.mean())

    # Sum of squared deviations within each cell plus between-cell mean
    # offsets, all weighted by num_samples.
    within = np.sum(stds**2 * num_samples)
    between = np.sum(((means - global_mean) ** 2) * num_samples)

    return float(np.sqrt((within + between) / total_n))


__all__ = [
    "list_recipes",
    "register",
    "select_normalization",
]
