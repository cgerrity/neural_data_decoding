neural_data_decoding — API reference
=====================================

This is the auto-generated API reference, built from the package's NumPy-style
docstrings. The narrative documentation — quickstart, concepts, cookbook,
deployment guides — lives in the MkDocs site under ``docs/narrative/``, and the
step-by-step teaching material is in the notebooks under ``notebooks/``.

Each page below documents a subpackage's public API (the names re-exported from
its ``__init__``). The five subpackages:

- :mod:`neural_data_decoding.data` — datasets, ``.mat`` loading, normalization,
  augmentation, collation.
- :mod:`neural_data_decoding.models` — encoders, classifiers, the composite
  model, custom layers, and the architecture registries.
- :mod:`neural_data_decoding.training` — the training loop and lifecycle, loss
  kernels, the multi-objective aggregator, curriculum schedules, freezing,
  checkpointing.
- :mod:`neural_data_decoding.interop` — the MATLAB bridge: folder hierarchy,
  ``CM_Table`` writer, the MATLAB subprocess runner, weight/parameter
  converters.
- :mod:`neural_data_decoding.sweeps` — the sweep dispatcher, SLURM template
  generator, and CLI helpers.

.. toctree::
   :maxdepth: 2
   :caption: Subpackages

   data
   models
   training
   interop
   sweeps
