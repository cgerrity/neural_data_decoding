"""``Schedule`` — a collection of named :class:`ScheduledParameter` values.

A ``Schedule`` is a thin mapping from parameter name to
``ScheduledParameter``, with a bulk ``update(epoch)`` that recomputes
every parameter. Consumers (Dataset, loss orchestrator, freeze applier)
hold a reference to the schedule and read ``current(name)`` live at use
time — never snapshot.

The MATLAB equivalents are ``cgg_generateLoadParameters_v2`` /
``cgg_generateLossWeights_v2`` / ``cgg_generateFreezeParameters``. In
Python we collapse the three subclasses into a single concrete class;
the differences between them (which parameter names, which defaults,
how the YAML is parsed) live in factory functions, not in inheritance.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from neural_data_decoding.training.schedules.parameter import ScheduledParameter


class Schedule:
    """A named collection of :class:`ScheduledParameter` values.

    Parameters
    ----------
    params
        Mapping of parameter name to :class:`ScheduledParameter`. A copy
        is taken; later mutation of the input mapping does not affect
        the schedule.

    Examples
    --------
    >>> from neural_data_decoding.training.schedules import (
    ...     Schedule, ScheduledParameter,
    ... )
    >>> sched = Schedule({
    ...     "kl": ScheduledParameter(base=1.0, epoch_points=[10, 20],
    ...                              magnitude_points=[0.0, 1.0]),
    ... })
    >>> sched.update(15)
    >>> round(sched.current("kl"), 4)
    0.4
    """

    def __init__(self, params: Mapping[str, ScheduledParameter]) -> None:
        """Take a shallow copy of the params mapping."""
        self._params: dict[str, ScheduledParameter] = dict(params)

    def update(self, epoch: int) -> None:
        """Recompute :attr:`ScheduledParameter.current` for every parameter."""
        for p in self._params.values():
            p.update(epoch)

    def current(self, name: str) -> float:
        """Return the live current value of parameter ``name``."""
        return self._params[name].current

    def __getitem__(self, name: str) -> ScheduledParameter:
        """Look up a :class:`ScheduledParameter` by name."""
        return self._params[name]

    def __contains__(self, name: object) -> bool:
        """Membership test for a parameter name."""
        return name in self._params

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names."""
        return iter(self._params)

    def __len__(self) -> int:
        """Number of parameters in the schedule."""
        return len(self._params)

    def names(self) -> list[str]:
        """Return the parameter names as a list."""
        return list(self._params)
