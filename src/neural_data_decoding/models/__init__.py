"""Architecture: encoder / decoder / classifier builders, custom layers, model registry.

Importing this package triggers side-effect registration of every
architecture variant defined in its submodules. Downstream code can then
look up architectures by their MATLAB-string identifier via
:func:`neural_data_decoding.models.registry.build_encoder` and
:func:`neural_data_decoding.models.registry.build_classifier`.
"""

# Import the registry first so submodules can decorate against it.
from . import registry  # noqa: F401

# Side-effect imports: each submodule registers one or more architectures
# under their MATLAB names when imported. Listed in milestone-order:
# Milestone A registers Logistic; later milestones expand.
from . import classifier  # noqa: F401
