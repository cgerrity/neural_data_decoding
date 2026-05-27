"""Architecture-string registry for encoder/classifier builders.

The MATLAB pipeline selects architectures by string identifier — e.g.
``ModelName='GRU'`` or ``ClassifierName='Logistic'`` — and dispatches to
one of 47 named encoder variants or 9 named classifier variants. The
Python pipeline mirrors this with two registries (one per architecture
family) so any of the MATLAB strings can be selected via config.

This module provides:

* :func:`register_encoder`, :func:`build_encoder` — for the
  ``ModelName`` family (47 entries in MATLAB).
* :func:`register_classifier`, :func:`build_classifier` — for the
  ``ClassifierName`` family (9 entries in MATLAB).
* :func:`list_encoders`, :func:`list_classifiers` — introspection.

Each registered builder has the shape ``Callable[[Mapping[str, Any]],
torch.nn.Module]``: takes a config mapping (typically the resolved Hydra
config object), returns the constructed module.

Milestones A/B/C populate only the variants they need; Milestone CC fills
in the rest. See Critical Note #14 in the migration plan.

Examples
--------
>>> from neural_data_decoding.models.registry import (
...     register_classifier, build_classifier, list_classifiers
... )
>>> @register_classifier("Echo")
... def _echo(cfg):
...     import torch.nn as nn
...     return nn.Identity()
>>> _ = list_classifiers()  # at least includes "Echo"
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch.nn as nn

ConfigLike = Mapping[str, Any]
Builder = Callable[[ConfigLike], nn.Module]


# ───────────────────────── Internal storage ─────────────────────────


_ENCODER_REGISTRY: dict[str, Builder] = {}
_CLASSIFIER_REGISTRY: dict[str, Builder] = {}


def _register(
    registry: dict[str, Builder], name: str
) -> Callable[[Builder], Builder]:
    """Build a decorator that adds ``builder`` to ``registry`` under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """

    def _decorator(builder: Builder) -> Builder:
        if name in registry:
            raise ValueError(
                f"Architecture builder '{name}' is already registered. "
                f"Register a different name or remove the duplicate."
            )
        registry[name] = builder
        return builder

    return _decorator


# ───────────────────────── Encoders ─────────────────────────


def register_encoder(name: str) -> Callable[[Builder], Builder]:
    """Decorator: register a builder under a MATLAB ``ModelName`` string.

    Parameters
    ----------
    name
        Exact MATLAB ``ModelName`` value (e.g. ``'GRU'``,
        ``'Variational GRU - Dropout 0.5'``, ``'Logistic Regression'``).

    Returns
    -------
    Callable
        Decorator that registers the wrapped function and returns it unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """
    return _register(_ENCODER_REGISTRY, name)


def build_encoder(name: str, cfg: ConfigLike) -> nn.Module:
    """Construct the encoder registered under ``name``.

    Parameters
    ----------
    name
        Architecture-string key. Must be a previously-registered
        ``ModelName``.
    cfg
        Resolved configuration mapping passed to the builder.

    Returns
    -------
    torch.nn.Module
        The constructed encoder module.

    Raises
    ------
    KeyError
        If ``name`` is not registered. The error message includes the list
        of known names to aid discoverability.
    """
    return _build(_ENCODER_REGISTRY, name, cfg, kind="encoder")


def list_encoders() -> list[str]:
    """Return the list of registered ``ModelName`` strings.

    Returns
    -------
    list of str
        Sorted registered encoder names.
    """
    return sorted(_ENCODER_REGISTRY.keys())


# ───────────────────────── Classifiers ─────────────────────────


def register_classifier(name: str) -> Callable[[Builder], Builder]:
    """Decorator: register a builder under a MATLAB ``ClassifierName`` string.

    Parameters
    ----------
    name
        Exact MATLAB ``ClassifierName`` value (e.g. ``'Deep LSTM - Dropout 0.5'``,
        ``'Logistic'``).

    Returns
    -------
    Callable
        Decorator that registers the wrapped function and returns it unchanged.
    """
    return _register(_CLASSIFIER_REGISTRY, name)


def build_classifier(name: str, cfg: ConfigLike) -> nn.Module:
    """Construct the classifier registered under ``name``.

    Parameters
    ----------
    name
        Classifier-string key. Must be a previously-registered
        ``ClassifierName``.
    cfg
        Resolved configuration mapping passed to the builder.

    Returns
    -------
    torch.nn.Module
        The constructed classifier module.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    return _build(_CLASSIFIER_REGISTRY, name, cfg, kind="classifier")


def list_classifiers() -> list[str]:
    """Return the list of registered ``ClassifierName`` strings.

    Returns
    -------
    list of str
        Sorted registered classifier names.
    """
    return sorted(_CLASSIFIER_REGISTRY.keys())


# ───────────────────────── Shared dispatch ─────────────────────────


def _build(
    registry: dict[str, Builder],
    name: str,
    cfg: ConfigLike,
    *,
    kind: str,
) -> nn.Module:
    """Look up ``name`` in ``registry`` and invoke the builder."""
    try:
        builder = registry[name]
    except KeyError as exc:
        registered = ", ".join(repr(k) for k in sorted(registry))
        raise KeyError(
            f"No {kind} registered under {name!r}. "
            f"Known {kind}s: {registered or '(none)'}"
        ) from exc
    return builder(cfg)


__all__ = [
    "build_classifier",
    "build_encoder",
    "list_classifiers",
    "list_encoders",
    "register_classifier",
    "register_encoder",
]
