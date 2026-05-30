"""Tests for :mod:`neural_data_decoding.models.registry`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch.nn as nn

from neural_data_decoding.models import registry


@pytest.fixture()
def empty_namespace() -> str:
    """Provide a unique architecture name that won't collide with real entries."""
    import uuid

    return f"__test_arch_{uuid.uuid4().hex[:8]}__"


def test_register_and_build_encoder_round_trip(empty_namespace: str) -> None:
    """Registering an encoder makes it discoverable via build_encoder()."""

    @registry.register_encoder(empty_namespace)
    def _factory(_cfg: Mapping[str, Any]) -> nn.Module:
        return nn.Linear(4, 2)

    module = registry.build_encoder(empty_namespace, {})
    assert isinstance(module, nn.Linear)
    assert empty_namespace in registry.list_encoders()


def test_register_and_build_classifier_round_trip(empty_namespace: str) -> None:
    """Same round-trip but for the classifier registry."""

    @registry.register_classifier(empty_namespace)
    def _factory(_cfg: Mapping[str, Any]) -> nn.Module:
        return nn.Identity()

    module = registry.build_classifier(empty_namespace, {})
    assert isinstance(module, nn.Identity)
    assert empty_namespace in registry.list_classifiers()


def test_duplicate_registration_raises(empty_namespace: str) -> None:
    """Registering the same name twice is an error (avoids silent shadowing)."""

    @registry.register_encoder(empty_namespace)
    def _first(_cfg: Mapping[str, Any]) -> nn.Module:
        return nn.Identity()

    with pytest.raises(ValueError, match="already registered"):

        @registry.register_encoder(empty_namespace)
        def _second(_cfg: Mapping[str, Any]) -> nn.Module:
            return nn.Identity()


def test_unknown_encoder_lists_known_options() -> None:
    """Asking for an unregistered encoder produces a helpful error."""
    with pytest.raises(KeyError, match="No encoder registered"):
        registry.build_encoder("__not_a_real_arch__", {})


def test_unknown_classifier_lists_known_options() -> None:
    """Asking for an unregistered classifier produces a helpful error."""
    with pytest.raises(KeyError, match="No classifier registered"):
        registry.build_classifier("__not_a_real_arch__", {})


def test_cfg_is_passed_through_to_builder(empty_namespace: str) -> None:
    """The config object is passed verbatim to the registered builder."""
    received: dict[str, object] = {}

    @registry.register_classifier(empty_namespace)
    def _factory(cfg: Mapping[str, Any]) -> nn.Module:
        received.update(cfg)
        return nn.Identity()

    registry.build_classifier(empty_namespace, {"num_classes": 4, "name": "test"})
    assert received == {"num_classes": 4, "name": "test"}
