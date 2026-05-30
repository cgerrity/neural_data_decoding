"""T2 — Confidence_Beta P-controller parity vs ``cgg_getConfidenceLossInformation``.

Iterates :func:`apply_confidence_routing` across the same sequence of batches
that ``scripts/generate_t2_confidence_beta_fixture.m`` ran through
``cgg_getConfidenceLossInformation``, then asserts the post-batch
``ConfidenceHistory.beta`` matches to ~1e-12.

The test pins **only Beta**, not the EMA values. Reason: MATLAB's
``cgg_getConfidenceLossInformation`` has a first-call special case for
Trial / Task EMAs (the ``else`` branch on lines 89-93 / 99-104) that
leaves them at the initial value of 1.0 after the very first batch,
while the regular EMA update applies from batch 2 onward. The Python
``apply_confidence_routing`` kernel (which matches MATLAB's
``cgg_lossConfidence``, NOT the wrapper) applies the regular EMA update
on every call. Beta is computed directly from the batch mean of
TotalConfidence and does NOT depend on the EMA state, so it is
parity-clean regardless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io
import torch

from neural_data_decoding.training.losses.confidence import (
    ConfidenceHistory,
    apply_confidence_routing,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "golden_weights" / "confidence_beta_t2.mat"
)


@pytest.fixture(scope="module")
def fixture() -> dict:
    """Load the Beta fixture once per module run."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Beta fixture missing: {FIXTURE_PATH}. Regenerate with: "
            "matlab -batch \"run('scripts/generate_t2_confidence_beta_fixture.m')\""
        )
    return scipy.io.loadmat(str(FIXTURE_PATH), struct_as_record=False, squeeze_me=False)


def _scalar(v: Any) -> float:
    return float(np.asarray(v).reshape(-1)[0])


def _cbt_to_btc(x: np.ndarray) -> np.ndarray:
    """Permute MATLAB CBT (channel, batch, time) → PyTorch (batch, time, channel)."""
    return np.transpose(x, (1, 2, 0))


def test_beta_p_controller_tracks_matlab_across_three_batches(fixture: dict) -> None:
    """After each batch the updated Beta matches MATLAB's Confidence_Beta to ~1e-12."""
    num_batches = int(_scalar(fixture["num_batches"]))
    batch_fraction = _scalar(fixture["batch_fraction"])

    # Start from the same initial state MATLAB does.
    history = ConfidenceHistory.initial(dtype=torch.float64, value=1.0)
    assert float(history.beta) == 1.0

    # Use throwaway y/target — Beta does not depend on the classification path.
    K = int(_scalar(fixture["K"]))
    B = int(_scalar(fixture["B"]))
    T = int(_scalar(fixture["T"]))
    y = torch.zeros((B, T, K), dtype=torch.float64)
    target = torch.zeros((B, T, K), dtype=torch.float64)

    for batch_idx in range(1, num_batches + 1):
        inputs = fixture[f"batch_{batch_idx}_inputs"][0, 0]
        expected_state = fixture[f"batch_{batch_idx}_state"][0, 0]

        trial = torch.from_numpy(
            _cbt_to_btc(np.asarray(inputs.trial_in, dtype=np.float64))
        )
        task = torch.from_numpy(
            _cbt_to_btc(np.asarray(inputs.task_in, dtype=np.float64))
        )

        # confidence_dropout=1.0 → mask always False → dropped == undropped
        # (isolates the deterministic math, matches the fixture's setup).
        out = apply_confidence_routing(
            y, target, trial, task,
            history=history,
            batch_fraction=batch_fraction,
            confidence_dropout=1.0,
            want_dataset_confidence=True,
            loss_type="L1",
        )

        expected_beta = _scalar(expected_state.confidence_beta)
        actual_beta = float(out.updated_history.beta)
        np.testing.assert_allclose(
            actual_beta, expected_beta, rtol=1e-12, atol=1e-12,
            err_msg=(
                f"Beta drifted from MATLAB after batch {batch_idx}: "
                f"python={actual_beta}, matlab={expected_beta}"
            ),
        )

        history = out.updated_history


def test_initial_beta_is_one() -> None:
    """ConfidenceHistory.initial sets beta=1.0 (matches MATLAB)."""
    h = ConfidenceHistory.initial(dtype=torch.float32)
    assert float(h.beta) == 1.0


def test_no_confidence_inputs_leaves_beta_untouched() -> None:
    """When both trial and task are None, Beta carries through unchanged."""
    h = ConfidenceHistory.initial(dtype=torch.float64, value=1.0)
    # Manually nudge Beta to a recognizable value so the test would catch
    # an accidental reset.
    h_nudged = ConfidenceHistory(total=h.total, trial=h.trial, task=h.task,
                                  beta=torch.tensor(2.5, dtype=torch.float64))
    y = torch.zeros((1, 1, 1), dtype=torch.float64)
    out = apply_confidence_routing(
        y, y, trial_confidence=None, task_confidence=None,
        history=h_nudged,
        batch_fraction=0.5,
    )
    assert float(out.updated_history.beta) == 2.5


def test_beta_clamps_at_lower_bound() -> None:
    """A sustained high-confidence stream pushes Beta to the 0.1 lower bound."""
    h = ConfidenceHistory.initial(dtype=torch.float64, value=1.0)
    y = torch.zeros((4, 2, 3), dtype=torch.float64)
    # TotalConf mean ≈ 1.0 → diff = -0.5 → Beta *= 0.5 each step → falls to 0.1.
    high_conf = torch.full((4, 2, 1), 1.0, dtype=torch.float64)
    high_task = torch.full((4, 2, 3), 1.0, dtype=torch.float64)
    for _ in range(20):
        out = apply_confidence_routing(
            y, y, high_conf, high_task,
            history=h, batch_fraction=0.5, confidence_dropout=1.0,
        )
        h = out.updated_history
    assert float(h.beta) == pytest.approx(0.1, abs=1e-9)


def test_beta_clamps_at_upper_bound() -> None:
    """A sustained low-confidence stream pushes Beta to the 10.0 upper bound."""
    h = ConfidenceHistory.initial(dtype=torch.float64, value=1.0)
    y = torch.zeros((4, 2, 3), dtype=torch.float64)
    # TotalConf mean ≈ 0 → diff = +0.5 → Beta *= 1.5 each step → rises to 10.
    low_conf = torch.full((4, 2, 1), 0.0, dtype=torch.float64)
    low_task = torch.full((4, 2, 3), 0.0, dtype=torch.float64)
    for _ in range(20):
        out = apply_confidence_routing(
            y, y, low_conf, low_task,
            history=h, batch_fraction=0.5, confidence_dropout=1.0,
        )
        h = out.updated_history
    assert float(h.beta) == pytest.approx(10.0, abs=1e-9)
