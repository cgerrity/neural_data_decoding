# `models/` — architecture building blocks

This subpackage holds the neural-network pieces of the pipeline: the
**encoders**, the **classifier heads**, the **composite** models that wire
them together, the **custom layers** the VAE path needs, and the two
**registries** that let a MATLAB config string select an architecture.
Everything here is a `torch.nn.Module` (or a builder that returns one); the
data pipeline, training loop, and loss code consume these modules but do not
define them.

Importing the package runs side-effect registration: each submodule adds its
architectures to the registry under their exact MATLAB `ModelName` /
`ClassifierName` strings.

## What's inside

- **Encoders** — `SimpleSequenceEncoder` (GRU / LSTM / Feedforward, in
  `encoder.py`), the `ConvolutionalEncoder` / `MultiFilterConvolutionalEncoder`
  (`conv_encoder.py`), and the frozen `PCAEncoder` (`layers/pca.py`).
  Seven `ModelName` strings are registered today: `GRU`, `LSTM`,
  `Feedforward`, `Convolutional`, `Resnet`, `Multi-Filter Convolutional`,
  `PCA`.
- **Classifier** — `MultiHeadClassifier` and `DeepLSTMClassifier`
  (`classifier.py`). Heads return a **list of per-dim logit tensors**
  (pre-softmax); softmax/cross-entropy is applied by the loss code, not here.
  Three `ClassifierName` strings are registered: `Logistic`,
  `Deep LSTM - Dropout 0.5`, `Deep LSTM - Dropout 0.25`.
- **Composite** (`composite.py`) — `EncoderClassifierComposite` (encoder →
  bottleneck → classifier) and `VariationalComposite` (the production VAE:
  NaN→0 → encoder → bottleneck → sampling → {decoder, classifier}).
- **Custom layers** (`layers/`) — `SamplingLayer` (VAE reparameterization),
  `NaNToZero` (removed-channel NaN → 0 at the encoder input), and
  `MILSoftmaxLayer` (softmax normalized jointly over Space-Channel-Time).
- **Stitching + fusion** (`stitching_fusion/`) — cross-area bridges that sit
  before the encoder / after the decoder.
- **Registries** — see below.

## Key entry points

- **`registry.build_encoder(name, cfg)` / `build_classifier(name, cfg)`** —
  look up a MATLAB architecture string and construct the module. Paired with
  the `@register_encoder` / `@register_classifier` decorators that populate
  the registries. (MATLAB defines 47 encoder / 9 classifier names; Python
  registers the SLURM-sweep + production subset listed above, not all of them.)
- **`architecture_registry.resolve_architecture(name) -> ArchitectureSpec`** —
  maps a `ModelName` string to a frozen flag bundle (`is_variational`,
  `transform`, `dropout`, conv-only fields, …), the port of
  `cgg_constructNetworkArchitecture.m`. This is the *spec* registry; the
  builders above are the *module* registry. Eight specs are registered.
- **`composite.build_variational_composite(cfg) -> VariationalComposite`** —
  assembles the active "Optimal" VAE topology from a resolved config.

## Design decisions (ADRs)

These record where the implementation intentionally diverges from
conventional PyTorch. Read them before changing block wiring, init, or the
sampling path.

- [ADR 018 — layer block order](../../../docs/narrative/adrs/018_layer_block_order_dropout_before_norm.md):
  each encoder block runs **Transform → Dropout → Norm → Activation** —
  dropout *before* norm, mirroring `cgg_generateSimpleBlock.m`, not the
  conventional Norm→Activation→Dropout.
- [ADR 022 — explicit He init](../../../docs/narrative/adrs/022_he_initialization_explicit.md):
  explicit Kaiming-*normal* init (with zeroed bias) is applied **only to
  Linear/FC layers** — the Feedforward encoder blocks and `LinearBottleneck`.
  GRU/LSTM layers keep PyTorch's default init, matching MATLAB, which sets
  `'he'` only on `fullyConnectedLayer`.
- [ADR 024 — sampling deterministic at inference](../../../docs/narrative/adrs/024_sampling_layer_deterministic_at_inference.md):
  `SamplingLayer` reparameterizes (`Z = mu + eps·sigma`) in `train` mode but
  returns `Z = mu` in `eval` mode. This is *not* the textbook VAE (which
  samples in both modes); the `self.training` branch is deliberate — don't
  "simplify" it to always sample.

One more naming trap to know: the `'SoftSign'` activation string actually
instantiates `nn.Softplus`, faithfully reproducing a long-standing MATLAB
naming bug (Critical Note #37). Use `''` in new configs.

## See also

- Concept page: [`docs/narrative/concepts/vae_sampling.md`](../../../docs/narrative/concepts/vae_sampling.md)
- Cookbook: [`docs/narrative/cookbook/add_a_new_architecture.md`](../../../docs/narrative/cookbook/add_a_new_architecture.md)
- Notebooks: [`notebooks/04_architecture/`](../../../notebooks/04_architecture/)
  (string dispatcher, simple/RNN/conv encoders, bottleneck, multi-head
  classifier, He-vs-default init).
