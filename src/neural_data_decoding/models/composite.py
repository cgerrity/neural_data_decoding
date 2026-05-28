"""Encoder + Bottleneck + Classifier glue module.

Milestone B adds the encoder pathway, which means the training loop now
operates on a **composed** model:

    raw_input → Encoder → Bottleneck → Classifier → per-dim logits

This module wraps the three sub-modules in a single :class:`nn.Module`
exposing the same ``forward(x) -> list[Tensor]`` contract as the
classifier-only Milestone A path. The training loop in
:mod:`neural_data_decoding.training.loop` thus doesn't need to know
whether the model has an encoder.

Milestone C will extend this composite to optionally include a decoder
(for the VAE / reconstruction loss path) — that change is additive: the
classifier branch keeps the same interface; the decoder is exposed via a
separate ``forward_with_reconstruction`` method.

Examples
--------
>>> import torch
>>> from neural_data_decoding.models.composite import EncoderClassifierComposite
>>> from neural_data_decoding.models.encoder import SimpleSequenceEncoder
>>> from neural_data_decoding.models.bottleneck import PassthroughBottleneck
>>> from neural_data_decoding.models.classifier import MultiHeadClassifier
>>> encoder = SimpleSequenceEncoder(in_features=8, hidden_sizes=[4], transform="GRU")
>>> bottleneck = PassthroughBottleneck(in_features=4)
>>> classifier = MultiHeadClassifier(in_features=4, num_classes_per_dim=[3])
>>> model = EncoderClassifierComposite(encoder=encoder, bottleneck=bottleneck,
...                                    classifier=classifier)
>>> outs = model(torch.zeros(2, 5, 8))
>>> [o.shape for o in outs]
[torch.Size([2, 5, 3])]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from neural_data_decoding.models.bottleneck import LinearBottleneck
from neural_data_decoding.models.classifier import DeepLSTMClassifier
from neural_data_decoding.models.decoder import NoopDecoder, build_decoder
from neural_data_decoding.models.encoder import SimpleSequenceEncoder
from neural_data_decoding.models.layers.nan_to_zero import NaNToZero
from neural_data_decoding.models.layers.sampling import SamplingLayer


class EncoderClassifierComposite(nn.Module):
    """Compose Encoder → Bottleneck → Classifier into a single trainable module.

    The composite's ``forward(x)`` returns the per-dim logits list produced
    by the classifier head — same shape contract as
    :class:`~neural_data_decoding.models.classifier.MultiHeadClassifier` so
    the training loop is unchanged across Milestones A and B.

    Parameters
    ----------
    encoder
        Module mapping ``(batch, time, in_features) → (batch, time, enc_out)``.
        Use :class:`~neural_data_decoding.models.encoder.SimpleSequenceEncoder`
        for the Simple GRU/LSTM/Feedforward branch.
    bottleneck
        Module mapping the encoder's output to the classifier's input. Use
        :class:`~neural_data_decoding.models.bottleneck.PassthroughBottleneck`
        when no transform is needed, or
        :class:`~neural_data_decoding.models.bottleneck.LinearBottleneck`
        for a single FC step (the Milestone C+ stack will be added here).
    classifier
        Module returning a list of per-dim logit tensors. Use
        :class:`~neural_data_decoding.models.classifier.MultiHeadClassifier`
        (Logistic) or
        :class:`~neural_data_decoding.models.classifier.DeepLSTMClassifier`
        (Deep LSTM).
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        bottleneck: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Apply Encoder → Bottleneck → Classifier and return per-dim logits."""
        encoded = self.encoder(x)
        bottle = self.bottleneck(encoded)
        return self.classifier(bottle)


@dataclass(slots=True)
class VariationalOutput:
    """Structured forward output of :class:`VariationalComposite`.

    Attributes
    ----------
    logits
        Per-output-dimension classification logits (the classifier head's
        output) — same contract as the non-variational composite.
    reconstruction
        Decoder output reconstructing the encoder input, shape
        ``(batch, time, input_features)``. ``None`` when the decoder is a
        :class:`~neural_data_decoding.models.decoder.NoopDecoder` (no
        reconstruction term active).
    mu
        Latent mean from the sampling layer.
    logvar
        Latent log-variance from the sampling layer.
    """

    logits: list[torch.Tensor]
    reconstruction: Optional[torch.Tensor]
    mu: torch.Tensor
    logvar: torch.Tensor


class VariationalComposite(nn.Module):
    """Full Stochastic VAE composite: encoder → sample → {decoder, classifier}.

    Wires the active Milestone C "Optimal" topology
    (``IsVariational=true``, ``EncoderOutputType='Stochastic'``):

    1. :class:`~neural_data_decoding.models.layers.nan_to_zero.NaNToZero` —
       removed-channel ``NaN`` → 0 at the input (Critical Note #38a).
    2. **Encoder** — the Simple GRU/LSTM stack.
    3. **Bottleneck** — a :class:`~neural_data_decoding.models.bottleneck.LinearBottleneck`
       that outputs ``2 * latent`` channels (mu | logvar concatenated).
    4. :class:`~neural_data_decoding.models.layers.sampling.SamplingLayer` —
       splits the statistics and draws ``Z`` (deterministic ``Z = mu`` at
       eval; reparameterized at train).
    5. **Decoder** — reconstructs the input from ``Z`` (or a
       :class:`~neural_data_decoding.models.decoder.NoopDecoder` when no
       reconstruction term is active).
    6. **Classifier** — per-dim logits from ``Z``.

    The "Stochastic" choice means the sampling lives in the *encoder* path,
    so **both** the decoder and classifier consume the sampled ``Z`` — the
    cleaner single-module expression of MATLAB's two-graph topology
    (Critical Note #13). The constituent pieces (GRU, sampling, FC, ELBO)
    are each independently parity-verified against MATLAB.

    Parameters
    ----------
    encoder
        Sequence encoder mapping ``(batch, time, in_features)`` →
        ``(batch, time, enc_out)``.
    bottleneck
        :class:`~neural_data_decoding.models.bottleneck.LinearBottleneck`
        whose ``out_features`` is ``2 * latent`` (the concatenated
        statistics). A non-linear bottleneck would also work as long as it
        emits ``2 * latent`` channels.
    sampling
        :class:`~neural_data_decoding.models.layers.sampling.SamplingLayer`
        splitting the bottleneck output on the channel axis.
    decoder
        Reconstruction decoder consuming ``Z``, or a ``NoopDecoder``.
    classifier
        Per-dim classifier consuming ``Z``.
    nan_to_zero
        Optional leading NaN→0 transform. Defaults to a fresh
        :class:`~neural_data_decoding.models.layers.nan_to_zero.NaNToZero`.

    Attributes
    ----------
    has_reconstruction : bool
        ``False`` when ``decoder`` is a ``NoopDecoder`` — the forward output's
        ``reconstruction`` field is ``None`` in that case.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        bottleneck: nn.Module,
        sampling: SamplingLayer,
        decoder: nn.Module,
        classifier: nn.Module,
        nan_to_zero: Optional[NaNToZero] = None,
    ) -> None:
        super().__init__()
        self.nan_to_zero = nan_to_zero if nan_to_zero is not None else NaNToZero()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.sampling = sampling
        self.decoder = decoder
        self.classifier = classifier
        self.has_reconstruction = not isinstance(decoder, NoopDecoder)

    def forward(self, x: torch.Tensor) -> VariationalOutput:
        """Run the full variational forward pass.

        Parameters
        ----------
        x
            Input tensor ``(batch, time, in_features)``. May contain
            ``NaN`` at removed-channel positions; they are zeroed before
            the encoder.

        Returns
        -------
        VariationalOutput
            ``logits`` (per-dim), ``reconstruction`` (or ``None``), ``mu``,
            ``logvar``.
        """
        x0 = self.nan_to_zero(x)
        encoded = self.encoder(x0)
        stats = self.bottleneck(encoded)
        z, mu, logvar = self.sampling(stats)

        reconstruction: Optional[torch.Tensor] = None
        if self.has_reconstruction:
            reconstruction = self.decoder(z)

        logits = self.classifier(z)
        return VariationalOutput(
            logits=logits, reconstruction=reconstruction, mu=mu, logvar=logvar
        )


def build_variational_composite(cfg: Mapping[str, Any]) -> VariationalComposite:
    """Assemble a Stochastic VAE composite from a resolved config.

    Follows MATLAB's ``HiddenSize`` convention: the **last** entry of
    ``hidden_sizes`` is the latent/bottleneck size; the rest are the
    encoder stack. The bottleneck emits ``2 * latent`` channels (mu |
    logvar), which the sampling layer splits.

    Recognized config keys
    ----------------------
    ``in_features`` (required)
        Input feature count (channels).
    ``hidden_sizes`` (required)
        Encoder hidden sizes with the latent size last, e.g. ``[16, 8, 4]``
        → encoder stack ``[16, 8]``, latent ``4``.
    ``num_classes_per_dim`` (required)
        Per-output-dimension class counts.
    ``classifier_hidden_size`` (required)
        Deep LSTM classifier hidden sizes.
    ``transform``
        Encoder/decoder transform (``'GRU'`` | ``'LSTM'`` |
        ``'Feedforward'``). Defaults to ``'GRU'``.
    ``classifier_dropout``
        Dropout for the Deep LSTM classifier. Defaults to ``0.5``.
    ``dropout`` / ``want_normalization`` / ``activation``
        Encoder/decoder block knobs. Defaults: ``0.0`` / ``False`` / ``''``.
    ``loss_type_decoder``
        ``"None"`` builds a
        :class:`~neural_data_decoding.models.decoder.NoopDecoder` (no
        reconstruction); anything else builds the real decoder.

    Returns
    -------
    VariationalComposite

    Raises
    ------
    KeyError
        If a required key is missing.
    ValueError
        If ``hidden_sizes`` has fewer than 2 entries (need at least one
        encoder layer + a latent size).
    """
    try:
        in_features = int(cfg["in_features"])
        hidden_sizes = [int(h) for h in cfg["hidden_sizes"]]
        num_classes_per_dim = list(cfg["num_classes_per_dim"])
        classifier_hidden = [int(h) for h in cfg["classifier_hidden_size"]]
    except KeyError as exc:
        raise KeyError(
            f"build_variational_composite: missing required cfg key {exc}"
        ) from exc

    if len(hidden_sizes) < 2:
        raise ValueError(
            "hidden_sizes needs at least 2 entries (>=1 encoder layer + a "
            f"trailing latent size); got {hidden_sizes}."
        )

    encoder_hidden = hidden_sizes[:-1]
    latent = hidden_sizes[-1]
    transform = str(cfg.get("transform", "GRU"))
    dropout = float(cfg.get("dropout", 0.0))
    want_normalization = bool(cfg.get("want_normalization", False))
    activation = str(cfg.get("activation", ""))

    encoder = SimpleSequenceEncoder(
        in_features=in_features,
        hidden_sizes=encoder_hidden,
        transform=transform,
        dropout=dropout,
        want_normalization=want_normalization,
        activation=activation,
    )
    # Bottleneck emits 2*latent channels (mu | logvar concatenated).
    bottleneck = LinearBottleneck(
        in_features=encoder.out_features, hidden_size=2 * latent
    )
    sampling = SamplingLayer(channel_dim=-1)

    decoder = build_decoder(
        {
            "loss_type_decoder": str(cfg.get("loss_type_decoder", "None")),
            "latent_size": latent,
            "encoder_hidden_sizes": encoder_hidden,
            "output_features": in_features,
            "transform": transform,
            "dropout": dropout,
            "want_normalization": want_normalization,
            "activation": activation,
        }
    )

    classifier = DeepLSTMClassifier(
        in_features=latent,
        num_classes_per_dim=num_classes_per_dim,
        hidden_sizes=classifier_hidden,
        dropout=float(cfg.get("classifier_dropout", 0.5)),
    )

    return VariationalComposite(
        encoder=encoder,
        bottleneck=bottleneck,
        sampling=sampling,
        decoder=decoder,
        classifier=classifier,
    )


__all__ = [
    "EncoderClassifierComposite",
    "VariationalComposite",
    "VariationalOutput",
    "build_variational_composite",
]

