"""Unit tests for the confidence-routing kernel.

Covers the two Critical Note #29 subtleties that don't need a MATLAB
fixture (they're either deterministic or autograd-internal):

* **Subtlety #2 — ConfidenceDropout**: with an explicit mask, the
  dropped tensor has ``1`` at masked positions and the original value
  elsewhere. With ``confidence_dropout=1.0`` the mask is always False
  (no resets); with ``confidence_dropout=0.0`` the mask is always True
  (all entries reset to 1).
* **Subtlety #4 — Stop-gradient on historical EMA**: when
  ``ConfidenceHistory`` fields require grad, no gradient flows back to
  them through the updated EMA.

Plus shape / interface tests for the kernel as a whole.
"""

from __future__ import annotations

import torch

from neural_data_decoding.training.losses.confidence import (
    ConfidenceHistory,
    apply_confidence_routing,
)


# ───────────────────────── Subtlety #2: ConfidenceDropout ─────────────────────────


def test_dropout_one_threshold_keeps_original() -> None:
    """``confidence_dropout=1.0`` → no resets (mask always False), Y' uses orig."""
    torch.manual_seed(0)
    y       = torch.randn(2, 3, 4)
    target  = torch.randn(2, 3, 4)
    trial   = torch.rand(2, 3, 1) * 0.5 + 0.3   # [0.3, 0.8]
    task    = torch.rand(2, 3, 4) * 0.5 + 0.3
    out = apply_confidence_routing(
        y, target, trial, task,
        history=ConfidenceHistory.initial(),
        confidence_dropout=1.0,                  # nothing should be reset
        want_dataset_confidence=False,
    )
    # With no resets, Y' = trial_last*task_last (broadcast) * y + (1-...)*target.
    trial_last = trial[:, -1, :]                 # (B, 1)
    task_last  = task[:, -1, :]                  # (B, K)
    c = (trial_last * task_last).unsqueeze(1)    # (B, 1, K)
    expected = c * y + (1 - c) * target
    torch.testing.assert_close(out.y_interpolated, expected)


def test_dropout_zero_threshold_resets_everything() -> None:
    """``confidence_dropout=0.0`` → mask always True; dropped == 1 everywhere."""
    torch.manual_seed(0)
    y       = torch.randn(2, 3, 4)
    target  = torch.randn(2, 3, 4)
    trial   = torch.rand(2, 3, 1) * 0.5 + 0.3
    task    = torch.rand(2, 3, 4) * 0.5 + 0.3
    out = apply_confidence_routing(
        y, target, trial, task,
        history=ConfidenceHistory.initial(),
        confidence_dropout=0.0,
        want_dataset_confidence=False,
    )
    # All confidence reset to 1 → interpolation pulls Y' all the way toward Y.
    torch.testing.assert_close(out.y_interpolated, y)


def test_dropout_explicit_mask_resets_only_masked_positions() -> None:
    """Explicit mask: True positions become 1; False positions keep orig conf."""
    y      = torch.zeros(2, 3, 4)
    target = torch.ones(2, 3, 4)
    trial  = torch.full((2, 3, 1), 0.4)
    task   = torch.full((2, 3, 4), 0.5)
    # Trial mask: True for batch 0, False for batch 1.
    trial_mask = torch.tensor([[True], [False]])
    # Task mask: True for batch 0 only.
    task_mask = torch.tensor([
        [True,  True,  True,  True],
        [False, False, False, False],
    ])
    out = apply_confidence_routing(
        y, target, trial, task,
        history=ConfidenceHistory.initial(),
        confidence_dropout=0.5,      # value doesn't matter; mask overrides
        explicit_trial_dropout_mask=trial_mask,
        explicit_task_dropout_mask=task_mask,
        want_dataset_confidence=False,
    )
    # Batch 0: dropped (trial, task) = (1, 1) → c = 1 → Y' = y = 0.
    # Batch 1: dropped (trial, task) = (0.4, 0.5) → c = 0.2 → Y' = 0.2*0 + 0.8*1 = 0.8.
    torch.testing.assert_close(out.y_interpolated[0], torch.zeros(3, 4))
    torch.testing.assert_close(out.y_interpolated[1], torch.full((3, 4), 0.8))


# ───────────────────────── Subtlety #4: stop-gradient on history ─────────────────────────


def test_no_gradient_flows_back_to_historical_ema() -> None:
    """history.trial.requires_grad=True; backward gives no grad to history."""
    torch.manual_seed(0)
    history = ConfidenceHistory(
        total=torch.tensor(0.7, requires_grad=True),
        trial=torch.tensor(0.7, requires_grad=True),
        task=torch.tensor(0.7,  requires_grad=True),
    )
    y      = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    # .detach().requires_grad_(True) to ensure these are LEAF tensors so .grad
    # is populated. Plain `requires_grad=True` followed by arithmetic produces
    # a non-leaf that won't receive .grad after backward.
    trial  = (torch.rand(2, 3, 1) * 0.5 + 0.3).detach().requires_grad_(True)
    task   = (torch.rand(2, 3, 4) * 0.5 + 0.3).detach().requires_grad_(True)

    out = apply_confidence_routing(
        y, target, trial, task,
        history=history,
        batch_fraction=0.3,
        confidence_dropout=1.0,
        want_dataset_confidence=True,
        loss_type="L1",
    )
    (out.total_loss + out.trial_loss + out.task_loss).backward()

    # Historical EMA must not receive gradient (detached inside the kernel).
    assert history.total.grad is None or torch.equal(
        history.total.grad, torch.zeros_like(history.total)
    )
    assert history.trial.grad is None or torch.equal(
        history.trial.grad, torch.zeros_like(history.trial)
    )
    assert history.task.grad is None or torch.equal(
        history.task.grad, torch.zeros_like(history.task)
    )
    # But the CURRENT batch's confidences DO receive gradient.
    assert trial.grad is not None and trial.grad.abs().sum() > 0
    assert task.grad is not None and task.grad.abs().sum() > 0


def test_updated_history_is_detached() -> None:
    """The returned ``updated_history`` fields have no autograd graph."""
    torch.manual_seed(0)
    y      = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    trial  = (torch.rand(2, 3, 1) * 0.5 + 0.3).detach().requires_grad_(True)
    task   = (torch.rand(2, 3, 4) * 0.5 + 0.3).detach().requires_grad_(True)

    out = apply_confidence_routing(
        y, target, trial, task,
        history=ConfidenceHistory.initial(),
        batch_fraction=0.5,
        confidence_dropout=1.0,
        want_dataset_confidence=True,
    )
    assert not out.updated_history.total.requires_grad
    assert not out.updated_history.trial.requires_grad
    assert not out.updated_history.task.requires_grad


# ───────────────────────── Branch availability ─────────────────────────


def test_no_confidence_returns_y_unchanged_and_zero_losses() -> None:
    """With both confidences None, kernel is a pass-through."""
    y      = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    out = apply_confidence_routing(
        y, target, trial_confidence=None, task_confidence=None,
        history=ConfidenceHistory.initial(),
    )
    assert torch.equal(out.y_interpolated, y)
    assert float(out.total_loss) == 0.0
    assert float(out.trial_loss) == 0.0
    assert float(out.task_loss)  == 0.0


def test_task_only_skips_conjunction() -> None:
    """With trial=None, total == task (no multiplication)."""
    torch.manual_seed(0)
    y      = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    task   = torch.rand(2, 3, 4) * 0.5 + 0.3
    out = apply_confidence_routing(
        y, target, trial_confidence=None, task_confidence=task,
        history=ConfidenceHistory.initial(),
        confidence_dropout=1.0,
        want_dataset_confidence=False,
    )
    # Total = Task (no Trial to conjunct with) → trial_loss = 0.
    assert float(out.trial_loss) == 0.0
    assert float(out.task_loss) > 0.0
