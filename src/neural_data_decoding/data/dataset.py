"""PyTorch :class:`~torch.utils.data.Dataset` implementations.

For Milestone A, we implement :class:`SyntheticTrialDataset` — an
in-memory dataset that generates classification-friendly trial-shaped
tensors with controllable signal-to-noise so the tracer bullet exercises
the full training loop without needing the real ``.mat`` data on disk.
A future milestone will add :class:`MatFileTrialDataset` that pulls
trials from ``.mat`` files via :mod:`neural_data_decoding.data.mat_files`.

Both datasets emit the same triple per ``__getitem__`` call:

* ``x`` — feature tensor of shape ``(num_samples, num_features)``
* ``targets`` — integer-class label per output dimension (shape ``(num_dimensions,)``)
* ``metadata`` — dict with at least ``session_id`` (used by the
  :class:`~neural_data_decoding.data.samplers.SingleSessionBatchSampler`)
  and ``trial_id``

The dataset returns the **NaN-zeroed input** as ``x`` (matching the
encoder-input convention; see Critical Note #38). For a future
reconstruction-loss path we'll also surface a NaN-preserving ``target``,
but Milestone A is classifier-only so we don't need the second variant.

Examples
--------
>>> ds = SyntheticTrialDataset(
...     num_sessions=2, trials_per_session=8, num_samples=20,
...     num_features=4, num_classes_per_dim=[3], seed=0,
... )
>>> x, t, meta = ds[0]
>>> x.shape, t.shape, meta["session_id"]
(torch.Size([20, 4]), torch.Size([1]), 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class TrialSample:
    """A single trial returned by a dataset.

    Attributes
    ----------
    x
        Feature tensor for the trial. Shape ``(num_samples, num_features)``.
    targets
        Per-output-dimension integer class label. Shape ``(num_dimensions,)``.
    metadata
        Free-form per-trial metadata. Always contains at least
        ``"session_id"`` and ``"trial_id"`` keys.
    """

    x: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, Any]


class SyntheticTrialDataset(Dataset):
    """Trial-shaped synthetic dataset for end-to-end pipeline testing.

    Generates a dataset where each output dimension's class label is
    causally encoded into the feature tensor — so a Logistic Regression
    classifier *should* be able to achieve well-above-chance accuracy
    given enough samples. This makes the dataset useful for the tracer
    bullet: training should converge to demonstrate the loop is wired
    up correctly.

    Trials are evenly distributed across ``num_sessions`` sessions. The
    underlying signal is:

        x[t, c] = label_signal[c] + N(0, noise_std)

    where ``label_signal`` is a per-class linear code with the active
    class's signal raised by ``signal_strength``.

    Parameters
    ----------
    num_sessions
        Number of distinct sessions (used by the single-session sampler).
    trials_per_session
        Number of trials per session. Total trials = sessions × trials.
    num_samples
        Time-dimension length (samples per trial).
    num_features
        Feature-dimension length (channels per sample).
    num_classes_per_dim
        Class count per output dimension. Length = number of output
        dimensions. Each dimension is treated independently — the per-
        dimension class is drawn uniformly.
    signal_strength
        Per-class signal magnitude added to the active class's feature
        channels. Higher values → easier classification.
    noise_std
        Standard deviation of the Gaussian noise added to every sample.
    seed
        RNG seed for reproducible dataset generation.
    """

    def __init__(
        self,
        *,
        num_sessions: int,
        trials_per_session: int,
        num_samples: int,
        num_features: int,
        num_classes_per_dim: list[int],
        signal_strength: float = 1.0,
        noise_std: float = 0.5,
        seed: int = 0,
    ) -> None:
        if num_sessions <= 0 or trials_per_session <= 0:
            raise ValueError("num_sessions and trials_per_session must be > 0.")
        if num_samples <= 0 or num_features <= 0:
            raise ValueError("num_samples and num_features must be > 0.")
        if not num_classes_per_dim or any(k <= 0 for k in num_classes_per_dim):
            raise ValueError(
                "num_classes_per_dim must be a non-empty list of positive ints."
            )

        rng = np.random.default_rng(seed)

        self.num_sessions = num_sessions
        self.trials_per_session = trials_per_session
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes_per_dim = list(num_classes_per_dim)
        self.signal_strength = float(signal_strength)
        self.noise_std = float(noise_std)

        total_trials = num_sessions * trials_per_session
        num_dims = len(num_classes_per_dim)

        # Per-class signal centers in feature space. Each (dim, class)
        # has a random unit-norm-ish direction; the active class's
        # direction is added to every sample of the trial.
        class_signals: list[np.ndarray] = []
        for k in num_classes_per_dim:
            signals = rng.standard_normal((k, num_features))
            signals /= np.linalg.norm(signals, axis=1, keepdims=True) + 1e-9
            class_signals.append(signals)
        self._class_signals = class_signals

        # Per-trial class labels: shape (total_trials, num_dims).
        self._labels = np.zeros((total_trials, num_dims), dtype=np.int64)
        for d, k in enumerate(num_classes_per_dim):
            self._labels[:, d] = rng.integers(low=0, high=k, size=total_trials)

        # Per-trial features: precomputed once for determinism.
        # Shape (total_trials, num_samples, num_features).
        features = rng.standard_normal(
            (total_trials, num_samples, num_features)
        ).astype(np.float32) * self.noise_std

        for d in range(num_dims):
            for trial_idx in range(total_trials):
                cls = self._labels[trial_idx, d]
                features[trial_idx] += (
                    self.signal_strength * class_signals[d][cls].astype(np.float32)
                )
        self._features = features

        # Session assignment: trials_per_session contiguous trials per session.
        self._session_ids = np.repeat(
            np.arange(num_sessions, dtype=np.int64), trials_per_session
        )

    @property
    def session_ids(self) -> np.ndarray:
        """Per-trial session identifier — consumed by SingleSessionBatchSampler."""
        return self._session_ids

    @property
    def num_dimensions(self) -> int:
        """Number of classification output dimensions."""
        return len(self.num_classes_per_dim)

    def __len__(self) -> int:
        """Total number of trials in the dataset."""
        return self._features.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return ``(features, targets, metadata)`` for trial ``idx``.

        Parameters
        ----------
        idx
            Trial index in ``[0, len(self))``.

        Returns
        -------
        features : torch.Tensor
            ``(num_samples, num_features)`` ``float32`` tensor.
        targets : torch.Tensor
            ``(num_dimensions,)`` ``int64`` class labels (one per output
            dimension).
        metadata : dict
            ``{"session_id": int, "trial_id": int}``.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Trial index {idx} out of range [0, {len(self)}).")

        features = torch.from_numpy(self._features[idx]).float()
        targets = torch.from_numpy(self._labels[idx]).long()
        metadata = {
            "session_id": int(self._session_ids[idx]),
            "trial_id": int(idx),
        }
        return features, targets, metadata


def collate_trials(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
) -> dict[str, Any]:
    """Collate a list of trial tuples into batched tensors + a metadata list.

    This is the ``collate_fn`` to pass to :class:`torch.utils.data.DataLoader`.

    Parameters
    ----------
    batch
        List of ``(features, targets, metadata)`` triples as produced by a
        Dataset's ``__getitem__``.

    Returns
    -------
    dict
        Keys:

        * ``"x"`` — stacked features, shape ``(batch, num_samples, num_features)``
        * ``"targets"`` — stacked targets, shape ``(batch, num_dimensions)``
        * ``"metadata"`` — list of per-trial metadata dicts, in input order
    """
    x = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return {"x": x, "targets": targets, "metadata": metadata}


__all__ = [
    "SyntheticTrialDataset",
    "TrialSample",
    "collate_trials",
]
