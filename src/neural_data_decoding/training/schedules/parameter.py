"""``ScheduledParameter`` — one named value that changes over training epochs.

A ``ScheduledParameter`` bundles a static base value with optional
piecewise-anneal waypoints (epoch points + magnitude multipliers). The
``current`` attribute holds the most recently computed per-epoch value;
``update(epoch)`` recomputes it via :func:`piecewise_anneal_value`.

The MATLAB equivalents are the ``Current<Name>`` mirror properties on
``cgg_generateLoadParameters_v2`` / ``cgg_generateLossWeights_v2`` /
``cgg_generateFreezeParameters``. In Python a single dataclass with two
fields (``base`` and ``current``) replaces the prefix-based mirror.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from neural_data_decoding.training.schedules.interpolator import (
    piecewise_anneal_value,
)


@dataclass
class ScheduledParameter:
    """A named scalar whose value changes piecewise-linearly with epoch.

    Parameters
    ----------
    base
        Static base value. Magnitudes are multipliers against this.
    epoch_points
        Monotonically-increasing waypoint epochs (1-indexed to match
        MATLAB). An empty sequence means no schedule — ``current`` stays
        equal to ``base`` for every epoch.
    magnitude_points
        Multipliers at each waypoint; must match the length of
        ``epoch_points``.

    Attributes
    ----------
    current
        The last value produced by :meth:`update` (or ``base`` before
        any update has run). Consumers read this live each epoch (and,
        in the dataset case, live within each ``__getitem__`` call).
    """

    base: float
    epoch_points: Sequence[float] = field(default_factory=tuple)
    magnitude_points: Sequence[float] = field(default_factory=tuple)
    current: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate waypoint lengths and seed ``current`` with the base."""
        if len(self.epoch_points) != len(self.magnitude_points):
            raise ValueError(
                f"epoch_points (len={len(self.epoch_points)}) and "
                f"magnitude_points (len={len(self.magnitude_points)}) "
                "must have the same length."
            )
        self.current = float(self.base)

    def update(self, epoch: int) -> float:
        """Recompute and store :attr:`current` for the given epoch.

        Parameters
        ----------
        epoch
            Current training epoch (1-indexed, matching MATLAB).

        Returns
        -------
        float
            The newly computed value (also stored in :attr:`current`).
        """
        self.current = piecewise_anneal_value(
            self.base, self.epoch_points, self.magnitude_points, epoch
        )
        return self.current
