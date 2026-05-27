"""Custom :class:`torch.utils.data.Sampler` implementations.

This module owns the **single-session minibatch contract** specified by
Critical Note #9 in the migration plan:

    Every minibatch contains trials from exactly one session.

This is the *opposite* of cross-session shuffling. The MATLAB pipeline
groups trials by session, partitions each session into minibatches, then
interleaves the per-session minibatches into the training schedule. Future
session-specific stitching layers will require this guarantee — they
apply a per-session transform that must operate on a uniformly-sessioned
batch.

See ``cgg_procAllSessionMiniBatchTable.m`` and
``cgg_procSplitSingleSessionDataStoreByMiniBatchSize.m`` for the MATLAB
reference.

Examples
--------
>>> import numpy as np
>>> sessions = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])  # per-trial session id
>>> sampler = SingleSessionBatchSampler(
...     session_ids=sessions, batch_size=2, drop_last=False, seed=0,
... )
>>> batches = list(sampler)
>>> all(len({sessions[i] for i in b}) == 1 for b in batches)
True
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Union

import numpy as np

# Accept anything array-like with a known length; we coerce to numpy.
SessionIdsLike = Union[np.ndarray, Sequence[int]]


class SingleSessionBatchSampler:
    """Yield index batches where every batch is drawn from one session.

    The sampler partitions trial indices by session, splits each
    session's index list into batches of ``batch_size``, then shuffles
    the resulting batches together. This produces a training schedule
    where successive batches may come from different sessions but no
    individual batch mixes sessions.

    Parameters
    ----------
    session_ids
        Per-trial session identifier. Length equals the number of trials
        in the dataset; values can be integers or strings — only equality
        matters. Trial order is preserved within session ID.
    batch_size
        Number of trials per minibatch. Each session's trial list is
        chopped into chunks of this size.
    drop_last
        If True (matches MATLAB's ``WantFullBatch=true`` behavior in
        ``cgg_procSplitSingleSessionDataStoreByMiniBatchSize``), any
        partial trailing chunk in a session is discarded. If False the
        partial chunk is kept as a smaller batch.
    seed
        RNG seed used to shuffle per-session trial order and the final
        batch order. Pass a different seed each epoch (via ``set_epoch``)
        if you want different orderings across epochs.

    Attributes
    ----------
    session_ids
        The session-id array stored as a :class:`numpy.ndarray`.
    batch_size, drop_last
        As passed to the constructor.

    Notes
    -----
    This sampler is intended to be passed as ``batch_sampler`` to a
    PyTorch :class:`torch.utils.data.DataLoader`. It yields **lists of
    integer indices**, not Tensors. Wrap with ``BatchSampler`` semantics:
    do not also set ``batch_size`` / ``shuffle`` on the DataLoader.
    """

    def __init__(
        self,
        session_ids: SessionIdsLike,
        *,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {batch_size}.")

        self.session_ids = np.asarray(session_ids)
        if self.session_ids.ndim != 1:
            raise ValueError(
                f"session_ids must be 1-D; got shape {self.session_ids.shape}."
            )
        if len(self.session_ids) == 0:
            raise ValueError("session_ids must be non-empty.")

        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self._seed = int(seed)
        self._epoch = 0
        self._session_to_indices = self._group_by_session()

    def _group_by_session(self) -> dict[object, np.ndarray]:
        """Return a mapping from session-id to the trial indices in that session.

        Preserves the original within-session ordering (i.e. the trial
        indices for a given session are sorted ascending by position in
        the dataset).
        """
        groups: dict[object, list[int]] = {}
        for trial_idx, sid in enumerate(self.session_ids):
            key = sid.item() if hasattr(sid, "item") else sid
            groups.setdefault(key, []).append(trial_idx)
        return {sid: np.array(idxs, dtype=np.int64) for sid, idxs in groups.items()}

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch, which influences the per-call RNG seed.

        Mirrors the ``DistributedSampler.set_epoch`` pattern: each epoch
        gets a different shuffle. Callers should invoke this at the start
        of each epoch to ensure unique orderings.
        """
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of trial indices, one batch per call.

        The order of trials within each session and the order of batches
        across sessions are both randomized using a seed derived from
        ``self._seed + self._epoch``.
        """
        rng = np.random.default_rng(self._seed + self._epoch)

        all_batches: list[list[int]] = []
        for indices in self._session_to_indices.values():
            shuffled = indices.copy()
            rng.shuffle(shuffled)

            num_full = len(shuffled) // self.batch_size
            for chunk_idx in range(num_full):
                start = chunk_idx * self.batch_size
                end = start + self.batch_size
                all_batches.append(shuffled[start:end].tolist())

            remainder = len(shuffled) % self.batch_size
            if remainder > 0 and not self.drop_last:
                all_batches.append(shuffled[-remainder:].tolist())

        # Shuffle the batch order so successive batches come from
        # different sessions (when there are multiple).
        rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        """Return the total number of batches per epoch.

        Computed deterministically from session sizes — does not depend on
        the RNG state, so this value is stable across epochs.
        """
        total = 0
        for indices in self._session_to_indices.values():
            full = len(indices) // self.batch_size
            total += full
            if not self.drop_last and len(indices) % self.batch_size:
                total += 1
        return total


__all__ = ["SingleSessionBatchSampler"]
