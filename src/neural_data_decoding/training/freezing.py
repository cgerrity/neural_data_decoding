"""Per-module freezing via optimizer param-group learning-rate scaling.

PyTorch has no direct equivalent of MATLAB's ``setLearnRateFactor``. The
closest semantic match is **per-module param groups**: one ``param_group``
per named submodule (encoder/decoder/classifier), each with its own ``lr``.
At epoch start the freeze schedule writes
``group["lr"] = base_lr * factor`` for each group — factor 0 fully
freezes (no parameter update), factor 1 fully unfreezes, factor 1e-2 is
"mostly frozen but learning slowly" (the value the Soft Three-Stage
regime uses to keep momentum alive on a nominally-frozen network).

This module is opt-in. Single-group optimizers (the legacy
``torch.optim.AdamW(model.parameters(), lr=...)`` form) keep working
unchanged when no freeze schedule is in use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn

from neural_data_decoding.training.schedules import Schedule


# Optimizer factory: typed as a callable to avoid Pylance/torch.Optimizer's
# generic-defaults signature mismatch with concrete subclasses like AdamW.
OptimizerFactory = Callable[..., torch.optim.Optimizer]


def build_optimizer_with_module_groups(
    module_groups: Mapping[str, nn.Module],
    *,
    initial_lr: float,
    weight_decay: float = 0.0,
    optimizer_factory: OptimizerFactory = torch.optim.AdamW,
) -> torch.optim.Optimizer:
    """Build an optimizer with one ``param_group`` per named submodule.

    Parameters
    ----------
    module_groups
        Mapping from group name to the module whose parameters belong in
        that group. The names should match the keys the freeze schedule
        will later use (e.g. ``{"encoder": composite.encoder, ...}``).
    initial_lr
        Initial learning rate applied to every group.
    weight_decay
        L2 weight decay (forwarded to the optimizer).
    optimizer_factory
        Callable that builds the optimizer from ``(param_groups, weight_decay=...)``.
        Defaults to :class:`torch.optim.AdamW`.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer with as many ``param_group`` entries as ``module_groups``
        has non-empty submodules. Each group has a ``"name"`` key the
        freeze applier looks up.

    Raises
    ------
    ValueError
        If no submodule contributes any learnable parameters.

    Notes
    -----
    Parameters shared across submodules are de-duplicated: each parameter
    appears in exactly one group, whichever was iterated first. This
    matches the MATLAB convention (one learnable, one freeze factor).
    """
    seen_ids: set[int] = set()
    groups: list[dict[str, Any]] = []
    for name, module in module_groups.items():
        params: list[torch.nn.Parameter] = []
        for p in module.parameters():
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            params.append(p)
        if not params:
            continue
        groups.append({"params": params, "lr": initial_lr, "name": name})
    if not groups:
        raise ValueError(
            "module_groups produced no learnable parameters; "
            "check that the named submodules are non-empty."
        )
    return optimizer_factory(groups, weight_decay=weight_decay)


def apply_freeze_to_optimizer(
    optimizer: torch.optim.Optimizer,
    freeze_schedule: Schedule,
    *,
    base_lr: float,
) -> None:
    """Update each named param group's lr to ``base_lr * freeze_factor``.

    Mirrors ``cgg_setFrozenNetwork_v2`` which reads ``CurrentFactor<Name>``
    and applies it via ``setLearnRateFactor`` to every learnable.

    Parameters
    ----------
    optimizer
        Optimizer built with :func:`build_optimizer_with_module_groups`
        (or any optimizer whose ``param_groups`` carry a ``"name"`` key).
    freeze_schedule
        Schedule whose parameter names match the optimizer group names.
        Groups whose names are absent from the schedule keep their
        previous learning rate.
    base_lr
        Reference learning rate (typically the config's
        ``initial_learning_rate``). The effective per-group lr is
        ``base_lr * freeze_schedule.current(name)``.
    """
    for group in optimizer.param_groups:
        name = group.get("name")
        if name is None or name not in freeze_schedule:
            continue
        factor = freeze_schedule.current(name)
        group["lr"] = base_lr * factor


# Default SGDM momentum — matches MATLAB cgg_initializeOptimizerVariables.m
# line 10 (which hardcodes 0.9 for the SGD case, where MATLAB's
# sgdmupdate IS SGD with momentum despite the 'SGD' config name).
SGDM_DEFAULT_MOMENTUM: float = 0.9


def resolve_optimizer_factory(name: str) -> OptimizerFactory:
    """Return a parameter-list → optimizer factory for the named optimizer.

    Maps the MATLAB-style config string (``"ADAM"`` or ``"SGDM"``) to a
    callable that wraps the appropriate PyTorch optimizer. The returned
    factory is compatible with both standard call sites
    (``factory(model.parameters(), lr=..., weight_decay=...)``) AND the
    per-module-groups call from
    :func:`build_optimizer_with_module_groups`, where ``lr`` is supplied
    per group inside the param_group dicts.

    Parameters
    ----------
    name
        Case-insensitive optimizer name. ``"ADAM"`` →
        :class:`torch.optim.AdamW`; ``"SGDM"`` →
        :class:`torch.optim.SGD` with ``momentum=0.9`` (matches MATLAB's
        ``sgdmupdate`` default; despite the MATLAB config name ``'SGD'``,
        the implementation is SGD-with-momentum).

    Returns
    -------
    OptimizerFactory
        Callable that accepts ``(params, *, lr=..., weight_decay=...)``
        kwargs and returns a configured optimizer.

    Raises
    ------
    ValueError
        On an unknown optimizer name.
    """
    upper = name.upper()
    if upper == "ADAM":
        # AdamW's lr defaults to 1e-3 in PyTorch — fine for the per-group
        # case where each group brings its own lr.
        return torch.optim.AdamW
    if upper == "SGDM":
        def sgdm_factory(
            params: Any, *,
            lr: float = 0.0,
            weight_decay: float = 0.0,
        ) -> torch.optim.Optimizer:
            return torch.optim.SGD(
                params, lr=lr, momentum=SGDM_DEFAULT_MOMENTUM,
                weight_decay=weight_decay,
            )
        return sgdm_factory
    raise ValueError(
        f"Unknown optimizer name: {name!r}. Expected 'ADAM' or 'SGDM' "
        f"(case-insensitive).",
    )


__all__ = [
    "OptimizerFactory",
    "SGDM_DEFAULT_MOMENTUM",
    "apply_freeze_to_optimizer",
    "build_optimizer_with_module_groups",
    "resolve_optimizer_factory",
]
