"""Frozen PCA encode/decode layers — port of ``cgg_PCAEncodingLayer.m``.

The MATLAB pipeline uses a frozen, pre-fit PCA layer as one of the
encoder transforms (``ModelName='PCA'``). PCA components are computed
once per fold on the training data, then injected as fixed weights
with no gradient flow. The Python port mirrors this with ``nn.Module``
classes that store the components and mean as buffers (not parameters)
and expose a :meth:`PCAEncodingLayer.fit` method that delegates to
:class:`sklearn.decomposition.PCA`.

Layout
------
The encoder receives the composite's standard 3-D ``(B, W, F)`` input
(post-flatten) and projects it to ``(B, W, n_components)`` via the
per-timestep linear transform:

    z = (x - mean) @ components.T

The decoder reverses the projection:

    x_hat = z @ components + mean

Both layers preserve the batch (``B``) and window (``W``) axes; PCA
operates per-window on the feature axis.

Pythonic deviations from MATLAB
-------------------------------
* MATLAB's ``ApplyPerTimePoint`` flag (per-window PCA vs. PCA across
  all windows) is implemented here only in the **across-all-windows**
  mode (``ApplyPerTimePoint=false``) — the default in production
  configs. The per-time-point variant would be a list of per-W PCAs;
  add when needed.
* MATLAB stores ``PCCoefficients`` as ``(in_features, out_features)``
  and right-multiplies; our convention is the sklearn / PyTorch
  ``(out_features, in_features)`` shape (components rows are
  principal axes) with ``x @ components.T``. Mathematically
  equivalent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class PCAEncodingLayer(nn.Module):
    """Frozen PCA encoder.

    Holds ``components`` (shape ``(n_components, in_features)``) and
    ``mean`` (shape ``(in_features,)``) as buffers — no gradient flows
    through them. Call :meth:`fit` once per fold on the training data
    before training starts.

    Parameters
    ----------
    in_features
        Number of input features (per-window flat dim, matches the
        composite's encoder contract).
    n_components
        Number of PCA components to retain — the layer's
        ``out_features``.

    Attributes
    ----------
    components : torch.Tensor
        Buffer of shape ``(n_components, in_features)``.
    mean : torch.Tensor
        Buffer of shape ``(in_features,)``.
    out_features : int
        ``n_components``.
    """

    def __init__(self, *, in_features: int, n_components: int) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1; got {in_features}.")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1; got {n_components}.")
        if n_components > in_features:
            raise ValueError(
                f"n_components ({n_components}) must be <= in_features "
                f"({in_features}).",
            )
        self.in_features = in_features
        self.n_components = n_components
        self.out_features = n_components
        self.register_buffer(
            "components", torch.zeros(n_components, in_features),
        )
        self.register_buffer("mean", torch.zeros(in_features))
        self._fitted = False

    def fit(self, data: np.ndarray | torch.Tensor) -> None:
        """Fit PCA on ``(N, in_features)`` samples and freeze the components.

        Uses :class:`sklearn.decomposition.PCA`. Call once before
        training; subsequent forward passes use the frozen components.

        Parameters
        ----------
        data
            Training data, shape ``(N, in_features)``. Either ``np.ndarray``
            or 2-D ``torch.Tensor`` (any dtype — converted to float32).

        Raises
        ------
        ValueError
            If ``data`` is not 2-D or its trailing axis doesn't match
            ``in_features``.
        """
        from sklearn.decomposition import PCA  # local import — soft dep at fit time

        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data)
        if data_np.ndim != 2 or data_np.shape[1] != self.in_features:
            raise ValueError(
                f"PCAEncodingLayer.fit expects (N, {self.in_features}); "
                f"got shape {tuple(data_np.shape)}.",
            )
        pca = PCA(n_components=self.n_components)
        pca.fit(data_np.astype(np.float32, copy=False))
        # sklearn's components_ is (n_components, in_features) — matches
        # our buffer shape exactly. mean_ is (in_features,). The
        # explicit asserts narrow Optional[ndarray] to ndarray.
        components_ = pca.components_
        mean_ = pca.mean_
        assert components_ is not None and mean_ is not None
        self.components.copy_(torch.from_numpy(components_.astype(np.float32)))
        self.mean.copy_(torch.from_numpy(mean_.astype(np.float32)))
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called (else forward is undefined)."""
        return self._fitted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, W, in_features) → (B, W, n_components)``.

        Subtracts the fitted mean per-feature, then projects via the
        components. No gradient flows through ``components`` or ``mean``
        (they're buffers).

        Parameters
        ----------
        x
            Tensor of shape ``(B, W, in_features)`` or any leading
            batch shape ending in ``in_features``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(*batch, n_components)``.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"PCAEncodingLayer expects trailing axis "
                f"{self.in_features}; got {x.shape[-1]}.",
            )
        centered = x - self.mean
        return centered @ self.components.T


class PCADecodingLayer(nn.Module):
    """Frozen PCA decoder — inverse of :class:`PCAEncodingLayer`.

    Shares the same ``components`` / ``mean`` buffers via either
    construction-time injection or :meth:`fit`. Forward computes
    ``z @ components + mean`` to reconstruct the original feature
    space from the PCA-encoded representation.

    Parameters
    ----------
    in_features
        Output feature count (matches the encoder's ``in_features``).
    n_components
        Number of PCA components (matches the encoder's ``out_features``).
    """

    def __init__(self, *, in_features: int, n_components: int) -> None:
        super().__init__()
        if in_features < 1 or n_components < 1:
            raise ValueError(
                f"in_features ({in_features}) and n_components "
                f"({n_components}) must both be >= 1.",
            )
        self.in_features = in_features
        self.n_components = n_components
        self.out_features = in_features
        self.register_buffer(
            "components", torch.zeros(n_components, in_features),
        )
        self.register_buffer("mean", torch.zeros(in_features))
        self._fitted = False

    def load_from(self, encoder: PCAEncodingLayer) -> None:
        """Copy ``components`` and ``mean`` from a fitted encoder.

        Convenience for the common pattern of fitting PCA once and
        sharing the buffers between encoder and decoder.

        Raises
        ------
        ValueError
            If shapes don't match.
        """
        if encoder.in_features != self.in_features:
            raise ValueError(
                f"encoder.in_features ({encoder.in_features}) does not "
                f"match decoder.in_features ({self.in_features}).",
            )
        if encoder.n_components != self.n_components:
            raise ValueError(
                f"encoder.n_components ({encoder.n_components}) does not "
                f"match decoder.n_components ({self.n_components}).",
            )
        self.components.copy_(encoder.components)
        self.mean.copy_(encoder.mean)
        self._fitted = encoder.is_fitted

    @property
    def is_fitted(self) -> bool:
        """``True`` after ``components`` and ``mean`` buffers are populated."""
        return self._fitted

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(B, W, n_components) → (B, W, in_features)``."""
        if z.shape[-1] != self.n_components:
            raise ValueError(
                f"PCADecodingLayer expects trailing axis "
                f"{self.n_components}; got {z.shape[-1]}.",
            )
        return z @ self.components + self.mean


def fit_pca_encoder_decoder(
    train_features: np.ndarray | torch.Tensor,
    *,
    n_components: int,
) -> tuple[PCAEncodingLayer, PCADecodingLayer]:
    """Fit a paired encoder/decoder from training data in one call.

    Convenience for the typical CLI flow: gather train_features
    ``(N_trials * N_windows, in_features)`` from the training set,
    pass through this helper, store the returned modules in the
    composite. Both modules share the same components/mean.

    Parameters
    ----------
    train_features
        Training data, shape ``(N, in_features)``.
    n_components
        Number of principal components.

    Returns
    -------
    tuple[PCAEncodingLayer, PCADecodingLayer]
        Paired, fitted modules.
    """
    if isinstance(train_features, torch.Tensor):
        data_np = train_features.detach().cpu().numpy()
    else:
        data_np = np.asarray(train_features)
    if data_np.ndim != 2:
        raise ValueError(
            f"train_features must be 2-D (N, F); got shape "
            f"{tuple(data_np.shape)}.",
        )
    in_features = data_np.shape[1]
    enc = PCAEncodingLayer(in_features=in_features, n_components=n_components)
    enc.fit(data_np)
    dec = PCADecodingLayer(in_features=in_features, n_components=n_components)
    dec.load_from(enc)
    return enc, dec


class PCAEncoder(nn.Module):
    """Per-window PCA encoder — registry-facing wrapper of :class:`PCAEncodingLayer`.

    Implements the MATLAB ``ModelName='PCA'`` encoder
    (``cgg_constructNetworkArchitecture.m`` lines 111-120 +
    ``cgg_PCAEncodingLayer.m``). Per the project directive, fits PCA
    components once per fold on the training set, then projects every
    window through the frozen transform.

    Composite contract: accepts 3-D ``(B, W, in_features)`` input,
    returns 3-D ``(B, W, n_components)``. Forward raises if
    :meth:`fit` has not been called yet.

    Parameters
    ----------
    in_features
        Flat feature count per window (``T * A * C``).
    n_components
        PCA output dimension — the encoder's ``out_features``.

    Attributes
    ----------
    out_features : int
        ``n_components``.
    """

    def __init__(self, *, in_features: int, n_components: int) -> None:
        super().__init__()
        self.pca = PCAEncodingLayer(
            in_features=in_features, n_components=n_components,
        )
        self.in_features = in_features
        self.out_features = n_components

    def fit(self, data: np.ndarray | torch.Tensor) -> None:
        """Fit the PCA on flat training features ``(N, in_features)``.

        See :meth:`PCAEncodingLayer.fit`.
        """
        self.pca.fit(data)

    def fit_from_dataloader(
        self,
        loader: "DataLoader",
    ) -> None:
        """Convenience: gather flat features from a DataLoader and fit.

        Iterates the loader, collecting each batch's flat per-window
        feature matrix ``(B*W, in_features)``, concatenates, and fits.
        Use this from the CLI's train-setup path after the data loader
        is built but before training starts.
        """
        chunks: list[np.ndarray] = []
        for batch in loader:
            x = batch["x"] if isinstance(batch, dict) else batch[0]
            # Accept either 5-D (B, W, T, A, C) or 3-D (B, W, F); flatten
            # the per-window dims to (B*W, in_features).
            if x.ndim == 5:
                b, w = x.shape[0], x.shape[1]
                flat = x.reshape(b * w, -1)
            elif x.ndim == 3:
                b, w = x.shape[0], x.shape[1]
                flat = x.reshape(b * w, x.shape[-1])
            else:
                raise ValueError(
                    "fit_from_dataloader expects each batch tensor to be "
                    f"3-D (B, W, F) or 5-D (B, W, T, A, C); got shape "
                    f"{tuple(x.shape)}.",
                )
            chunks.append(flat.detach().cpu().numpy())
        if not chunks:
            raise ValueError("Loader yielded zero batches — nothing to fit.")
        self.fit(np.concatenate(chunks, axis=0))

    @property
    def is_fitted(self) -> bool:
        """``True`` once :meth:`fit` (or :meth:`fit_from_dataloader`) has been called."""
        return self.pca.is_fitted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, W, in_features) → (B, W, n_components)``."""
        if not self.is_fitted:
            raise RuntimeError(
                "PCAEncoder.forward called before .fit() — PCA components "
                "are still zero. Call fit() or fit_from_dataloader() "
                "before training starts.",
            )
        return self.pca(x)


__all__ = [
    "PCADecodingLayer",
    "PCAEncoder",
    "PCAEncodingLayer",
    "fit_pca_encoder_decoder",
]
