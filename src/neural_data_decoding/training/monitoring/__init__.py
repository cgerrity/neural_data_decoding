"""Training telemetry package — currently a placeholder.

Reserved for consolidated run telemetry: a Weights & Biases logger (``wandb``
is a declared dependency but not yet wired — attach one via the ``EpochCallback``
hook on :func:`neural_data_decoding.training.lifecycle.fit_supervised`) and a
proactive OOM memory probe (Critical Note #19). Neither is implemented yet, so
this package is intentionally empty.

Note: the ``CM_Table.mat`` writer lives in
:mod:`neural_data_decoding.interop.cm_table_format`, not here; model-selection
telemetry is driven today by the ``on_optimal_callback`` in the training loop.
"""
