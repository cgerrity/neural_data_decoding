# Add a new architecture

Architectures are registered, not hardcoded — you add one without editing any
core dispatcher. There are two registries, depending on whether your
architecture is genuinely new code or a parameter variation of an existing one.

## New builder: `@register_encoder`

For a genuinely new encoder, decorate a builder function:

```python
import torch.nn as nn
from neural_data_decoding.models import registry

@registry.register_encoder("MyEncoder")          # the MATLAB ModelName string
def build_my_encoder(cfg):
    return nn.Sequential(
        nn.Linear(cfg.get("in_dim"), 128),
        nn.ReLU(),
        nn.Linear(128, cfg.get("out_dim")),
    )
```

The builder takes the config mapping and returns an `nn.Module`. Once
registered, `registry.build_encoder("MyEncoder", cfg)` works like any built-in,
and `model_name: MyEncoder` in a config selects it.

Classifiers use the parallel `register_classifier` / `build_classifier`.

### Make the registration run

Registration is an **import side-effect**. Put your builder in a module and
ensure that module is imported at startup — e.g. add it to the imports in
`neural_data_decoding/models/__init__.py`, which is imported before any build.
If the module is never imported, the registry never learns the name.

## Parameter variation: `ArchitectureSpec`

If your "new" architecture is really an existing builder with different flags
(a common case), add an entry to the flag-bundle registry in
`models/architecture_registry.py` (`ArchitectureSpec` / `_ARCH_SPECS`) instead
of writing a new builder. `resolve_architecture(name)` returns the flag bundle.

## Verify

- A duplicate name raises `ValueError` at import — pick a unique name.
- `registry.build_encoder("Unknown", cfg)` lists valid options in its error.
- The non-crash gating tests live in
  `tests/integration/test_slurm_sweep_coverage.py`; add your architecture to the
  parametrization to confirm it trains end-to-end.

## Related

- [The dynamic curriculum](../concepts/dynamic_curriculum.md) applies per-module
  freeze factors — make sure your model exposes `encoder`/`decoder`/`classifier`
  submodules if you want per-module freezing.
- Notebook `notebooks/09_production_deployment/09.6_extending_the_pipeline.ipynb`.
