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

import torch
import torch.nn as nn


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


__all__ = ["EncoderClassifierComposite"]
