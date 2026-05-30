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
from neural_data_decoding.models.confidence_heads import (
    TaskConfidenceHead,
    TrialConfidenceHead,
)
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
    trial_confidence
        Per-trial confidence ``(batch, time, 1)`` from
        :class:`~neural_data_decoding.models.confidence_heads.TrialConfidenceHead`.
        ``None`` when no Trial head is configured (``confidence_type``
        omits ``'Trial'``).
    task_confidence
        Per-output-dim confidence ``(batch, time, num_dims)`` from
        :class:`~neural_data_decoding.models.confidence_heads.TaskConfidenceHead`.
        ``None`` when no Task head is configured.
    """

    logits: list[torch.Tensor]
    reconstruction: Optional[torch.Tensor]
    mu: torch.Tensor
    logvar: torch.Tensor
    trial_confidence: Optional[torch.Tensor] = None
    task_confidence: Optional[torch.Tensor] = None


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
        trial_confidence_head: Optional[TrialConfidenceHead] = None,
        task_confidence_head: Optional[TaskConfidenceHead] = None,
    ) -> None:
        super().__init__()
        self.nan_to_zero = nan_to_zero if nan_to_zero is not None else NaNToZero()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.sampling = sampling
        self.decoder = decoder
        self.classifier = classifier
        self.trial_confidence_head = trial_confidence_head
        self.task_confidence_head = task_confidence_head
        self.has_reconstruction = not isinstance(decoder, NoopDecoder)
        # Task confidence requires the classifier to expose penultimate
        # features. DeepLSTMClassifier does; MultiHeadClassifier does not.
        # Detected lazily in forward() so a Task head built against a
        # classifier that supports it just works.
        self.has_trial_confidence = trial_confidence_head is not None
        self.has_task_confidence = task_confidence_head is not None

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
            ``logvar``, ``trial_confidence`` (or ``None``),
            ``task_confidence`` (or ``None``).
        """
        x0 = self.nan_to_zero(x)
        encoded = self.encoder(x0)
        stats = self.bottleneck(encoded)
        z, mu, logvar = self.sampling(stats)

        reconstruction: Optional[torch.Tensor] = None
        if self.has_reconstruction:
            reconstruction = self.decoder(z)

        # Classifier path: when a Task confidence head is present, ask the
        # classifier to also surface its penultimate features (one tensor
        # per output dim) so the Task head can tap them in parallel to
        # the classification FC. Otherwise stick with the plain forward.
        task_confidence: Optional[torch.Tensor] = None
        if self.task_confidence_head is not None:
            if not hasattr(self.classifier, "forward_with_features"):
                raise TypeError(
                    "task_confidence_head requires a classifier that exposes "
                    "`forward_with_features` (e.g. DeepLSTMClassifier). "
                    f"Got {type(self.classifier).__name__}."
                )
            features_per_dim, logits = self.classifier.forward_with_features(z)
            task_confidence = self.task_confidence_head(features_per_dim)
        else:
            logits = self.classifier(z)

        trial_confidence: Optional[torch.Tensor] = None
        if self.trial_confidence_head is not None:
            trial_confidence = self.trial_confidence_head(z)

        return VariationalOutput(
            logits=logits, reconstruction=reconstruction,
            mu=mu, logvar=logvar,
            trial_confidence=trial_confidence,
            task_confidence=task_confidence,
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
        num_classes_per_dim = list(cfg["num_classes_per_dim"])
        classifier_hidden = [int(h) for h in cfg["classifier_hidden_size"]]
    except KeyError as exc:
        raise KeyError(
            f"build_variational_composite: missing required cfg key {exc}"
        ) from exc

    encoder, bottleneck, sampling, decoder = _build_ae_core(cfg)
    latent = int(cfg["hidden_sizes"][-1])

    classifier = DeepLSTMClassifier(
        in_features=latent,
        num_classes_per_dim=num_classes_per_dim,
        hidden_sizes=classifier_hidden,
        dropout=float(cfg.get("classifier_dropout", 0.5)),
    )

    # Confidence heads (Milestone C #7) — built conditionally based on
    # cfg.confidence_type. Recognized entries (case-insensitive,
    # whitespace-tolerant): "Trial" and "Task".
    confidence_types = _normalize_confidence_types(cfg.get("confidence_type", []))
    trial_head: Optional[TrialConfidenceHead] = None
    task_head: Optional[TaskConfidenceHead] = None
    if "trial" in confidence_types:
        trial_head = TrialConfidenceHead(in_features=latent)
    if "task" in confidence_types:
        # All DeepLSTMClassifier stacks share the same final hidden size
        # (last entry of classifier_hidden_size).
        last_hidden = classifier_hidden[-1] if classifier_hidden else latent
        task_head = TaskConfidenceHead(
            in_features_per_dim=[last_hidden] * len(num_classes_per_dim),
        )

    return VariationalComposite(
        encoder=encoder,
        bottleneck=bottleneck,
        sampling=sampling,
        decoder=decoder,
        classifier=classifier,
        trial_confidence_head=trial_head,
        task_confidence_head=task_head,
    )


def _normalize_confidence_types(raw: Any) -> set[str]:
    """Lower-case the confidence_type list for case-insensitive containment checks.

    Accepts a list, tuple, or single string (MATLAB sometimes passes a
    bare string when only one type is active).
    """
    if isinstance(raw, str):
        items = [raw]
    elif raw is None:
        items = []
    else:
        items = list(raw)
    return {str(it).strip().lower() for it in items if str(it).strip()}


@dataclass(slots=True)
class AutoencoderOutput:
    """Structured forward output of :class:`VariationalAutoencoder` (Stage 1).

    Mirrors :class:`VariationalOutput` but **lacks the ``logits`` field**:
    Stage 1 unsupervised pre-training literally cannot produce classification
    logits because the model has no classifier head. The training kernel
    that consumes this type can therefore not accidentally compute a
    classification loss.

    Attributes
    ----------
    reconstruction
        Decoder output reconstructing the encoder input. Always present in
        Stage 1 (a Stage 1 model with ``NoopDecoder`` would have nothing
        to optimize).
    mu
        Latent mean from the sampling layer.
    logvar
        Latent log-variance from the sampling layer.
    """

    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class VariationalAutoencoder(nn.Module):
    """Stage 1 unsupervised pre-training model: encoder → sample → decoder.

    The Stage 1 counterpart of :class:`VariationalComposite`. Identical
    construction except there is **no classifier head** — the forward
    output is an :class:`AutoencoderOutput` carrying ``reconstruction``,
    ``mu``, ``logvar``, with no ``logits`` field.

    Mirrors MATLAB's Stage 1 path in ``cgg_trainAllAutoEncoder_v2.m``
    (line 232's ``cgg_trainNetwork`` call **without** a ``Classifier``
    argument → ``HasClassifier=false`` → all classification math
    skipped).

    Parameters
    ----------
    encoder
        Sequence encoder.
    bottleneck
        :class:`~neural_data_decoding.models.bottleneck.LinearBottleneck`
        emitting ``2 * latent`` channels (mu | logvar).
    sampling
        :class:`~neural_data_decoding.models.layers.sampling.SamplingLayer`.
    decoder
        Reconstruction decoder consuming ``Z``.
    nan_to_zero
        Optional leading NaN→0 transform. Defaults to a fresh
        :class:`~neural_data_decoding.models.layers.nan_to_zero.NaNToZero`.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        bottleneck: nn.Module,
        sampling: SamplingLayer,
        decoder: nn.Module,
        nan_to_zero: Optional[NaNToZero] = None,
    ) -> None:
        super().__init__()
        self.nan_to_zero = nan_to_zero if nan_to_zero is not None else NaNToZero()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.sampling = sampling
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        """Run the Stage 1 forward pass: encode → sample → decode."""
        x0 = self.nan_to_zero(x)
        encoded = self.encoder(x0)
        stats = self.bottleneck(encoded)
        z, mu, logvar = self.sampling(stats)
        reconstruction = self.decoder(z)
        return AutoencoderOutput(reconstruction=reconstruction, mu=mu, logvar=logvar)


def build_variational_autoencoder(cfg: Mapping[str, Any]) -> VariationalAutoencoder:
    """Assemble a Stage 1 :class:`VariationalAutoencoder` from a resolved config.

    Same config schema as :func:`build_variational_composite` — the
    ``classifier_*`` keys are simply not consumed because there is no
    classifier head. This lets the CLI pass the same config dict to both
    builders without conditional pre-processing.

    A ``NoopDecoder`` is disallowed in Stage 1: there has to be a
    real reconstruction objective for the unsupervised stage to do any
    work. Any ``loss_type_decoder`` other than ``"None"`` is accepted.

    Returns
    -------
    VariationalAutoencoder

    Raises
    ------
    KeyError
        If a required key is missing.
    ValueError
        If ``hidden_sizes`` has fewer than 2 entries, or
        ``loss_type_decoder`` is ``"None"`` (no reconstruction).
    """
    encoder, bottleneck, sampling, decoder = _build_ae_core(cfg)
    if isinstance(decoder, NoopDecoder):
        raise ValueError(
            "Stage 1 (autoencoder) requires a real decoder for the "
            "reconstruction objective; cfg.loss_type_decoder='None' "
            "produced a NoopDecoder. Either enable reconstruction or "
            "skip Stage 1 by setting num_epochs_autoencoder=0."
        )
    return VariationalAutoencoder(
        encoder=encoder, bottleneck=bottleneck, sampling=sampling, decoder=decoder,
    )


def copy_autoencoder_weights(
    src: VariationalAutoencoder, dst: VariationalComposite,
) -> None:
    """Copy Stage 1 encoder/bottleneck/sampling/decoder weights into Stage 2.

    Used at the Stage 1 → Stage 2 handoff: the Optimal autoencoder weights
    bootstrap the corresponding submodules of the full composite, then
    Stage 2 training fine-tunes them alongside the freshly-initialized
    classifier head.

    The two architectures must agree on the encoder / bottleneck / sampling
    / decoder shapes (this is the caller's responsibility — both should
    be built from the same cfg).

    Raises
    ------
    RuntimeError
        If any of the four submodules' state_dicts have mismatched keys
        or shapes.
    """
    dst.encoder.load_state_dict(src.encoder.state_dict())
    dst.bottleneck.load_state_dict(src.bottleneck.state_dict())
    dst.sampling.load_state_dict(src.sampling.state_dict())
    dst.decoder.load_state_dict(src.decoder.state_dict())
    # nan_to_zero is stateless (no learnables) — nothing to copy.


def _build_ae_core(
    cfg: Mapping[str, Any],
) -> tuple[SimpleSequenceEncoder, LinearBottleneck, SamplingLayer, nn.Module]:
    """Shared construction for the autoencoder core (encoder + bottleneck + sampling + decoder).

    Extracted so :func:`build_variational_composite` and
    :func:`build_variational_autoencoder` stay DRY without inheriting from
    each other.
    """
    try:
        in_features = int(cfg["in_features"])
        hidden_sizes = [int(h) for h in cfg["hidden_sizes"]]
    except KeyError as exc:
        raise KeyError(
            f"_build_ae_core: missing required cfg key {exc}"
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
    bottleneck = LinearBottleneck(
        in_features=encoder.out_features, hidden_size=2 * latent,
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
    return encoder, bottleneck, sampling, decoder


__all__ = [
    "AutoencoderOutput",
    "EncoderClassifierComposite",
    "VariationalAutoencoder",
    "VariationalComposite",
    "VariationalOutput",
    "build_variational_autoencoder",
    "build_variational_composite",
    "copy_autoencoder_weights",
]


# Re-export TaskConfidenceHead so callers don't have to dig through
# models/confidence_heads.py. (TrialConfidenceHead and TaskConfidenceHead
# are already exported from confidence_heads.py for direct use.)

