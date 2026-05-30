"""``CurriculumBundle`` — the three schedules with one ``update(epoch)`` call.

Bundles a load, weight, and freeze schedule together so the training
loop can advance the entire curriculum in a single call. Also carries
the optional :class:`KLBaseAnneal` that mirrors the inline ramp on the
KL weight base in ``cgg_trainNetwork.m``::

    WeightKL_Anneal = cgg_annealWeight(Epoch, WeightKL, WeightDelayEpoch, WeightEpochRamp);
    WeightParameters.WeightKL = WeightKL_Anneal;  % overwrites base
    WeightParameters.updateAllParameters(Epoch);  % then dynamic multiply

In Python this is one ``CurriculumBundle.update(epoch)``: the KL base is
adjusted in-place, then the dynamic schedule multiplies.
"""

from __future__ import annotations

from dataclasses import dataclass

from neural_data_decoding.training.schedules.interpolator import (
    piecewise_anneal_value,
)
from neural_data_decoding.training.schedules.schedule import Schedule


@dataclass(frozen=True)
class KLBaseAnneal:
    """Legacy ramp on the KL weight's base value (pre-dynamic-multiply).

    Mirrors ``cgg_annealWeight(Epoch, WeightKL, WeightDelayEpoch, WeightEpochRamp)``:
    ramps the base from 0 to ``initial_weight`` linearly between epoch
    ``delay_epoch`` and ``delay_epoch + epoch_ramp``, with the same
    ``(epoch - 1)`` off-by-one quirk.

    Parameters
    ----------
    initial_weight
        The fully-annealed KL weight base (i.e., the MATLAB ``WeightKL``).
    delay_epoch
        Epoch before which the annealed base is held at 0.
    epoch_ramp
        Number of epochs the ramp takes to reach ``initial_weight``.
    """

    initial_weight: float
    delay_epoch: int
    epoch_ramp: int

    def value_at(self, epoch: int) -> float:
        """Compute the annealed base value at the given epoch."""
        return piecewise_anneal_value(
            self.initial_weight,
            (self.delay_epoch, self.delay_epoch + self.epoch_ramp),
            (0.0, 1.0),
            epoch,
        )


@dataclass
class CurriculumBundle:
    """The full curriculum: load + weight + freeze schedules.

    Parameters
    ----------
    load
        Augmentation-magnitude schedule (consumed by Dataset live).
    weight
        Loss-weight schedule (consumed by loss orchestrator live).
    freeze
        Freeze-factor schedule (consumed at epoch start by the freeze
        applier).
    kl_anneal
        Optional legacy KL base anneal. When present, ``update(epoch)``
        rewrites ``weight["kl"].base`` to the annealed value before
        running the schedule's dynamic multiply — exactly mirroring
        ``cgg_trainNetwork.m``'s two-step KL pipeline.
    """

    load: Schedule
    weight: Schedule
    freeze: Schedule
    kl_anneal: KLBaseAnneal | None = None

    def update(self, epoch: int) -> None:
        """Advance all three schedules (and the KL base anneal if set).

        Order matches ``cgg_trainNetwork.m``: legacy KL base anneal
        first, then ``LoadParameters.updateAllParameters``,
        ``WeightParameters.updateAllParameters``,
        ``FreezeParameters.updateAllParameters``.
        """
        if self.kl_anneal is not None and "kl" in self.weight:
            self.weight["kl"].base = self.kl_anneal.value_at(epoch)
        self.load.update(epoch)
        self.weight.update(epoch)
        self.freeze.update(epoch)
