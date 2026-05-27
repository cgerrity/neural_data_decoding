"""Entry point for ``python -m neural_data_decoding``.

The ``if __name__ == "__main__"`` guard is essential here: Python executes
``__main__.py`` not only when the package is invoked via ``-m``, but also
when documentation tools (Sphinx autosummary, pdoc, etc.) import it for
introspection. Without the guard, those imports would parse the tool's own
``sys.argv`` and crash on unrecognized arguments.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
