"""Sphinx configuration for the API reference build.

The narrative documentation is built by MkDocs (see ``../mkdocs.yml``).
This Sphinx build covers only the auto-generated API reference.

Build with: ``sphinx-build -W -b html . ../build/api``
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable for autodoc without requiring `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ───────────────────────── Project info ─────────────────────────
project = "neural_data_decoding"
author = "Charles Gerrity"
copyright = f"2026, {author}"

try:
    from neural_data_decoding import __version__ as version
except ImportError:  # pragma: no cover
    version = "0.0.1"

release = version

# ───────────────────────── Extensions ─────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # NumPy / Google docstring parsing
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # NOTE: sphinx_autodoc_typehints is intentionally NOT enabled — it is
    # incompatible with Sphinx 9+ (emits RemovedInSphinx10Warning, which fails
    # the ``-W`` build) and is redundant with ``autodoc_typehints = "description"``
    # below, which native autodoc + napoleon already handle.
    "myst_parser",               # accept Markdown sources alongside RST
]

# Disable recursive auto-generation; each subpackage `.rst` file documents
# its own module via :automodule:. That keeps the structure flat and avoids
# duplicate object descriptions between the auto-generated pages and the
# hand-written subpackage pages.
autosummary_generate = False

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,      # interrogate is the source of truth for coverage
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
# Render a class's ``Attributes`` docstring section as inline ``:ivar:`` fields
# rather than separate ``.. py:attribute::`` objects. Without this, dataclass
# fields are documented twice (once by autodoc's ``:members:``, once by the
# napoleon Attributes section) — the "duplicate object description" warnings.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ───────────────────────── HTML theme ─────────────────────────
html_theme = "furo"
html_title = f"{project} {release}"
html_static_path: list[str] = []  # empty until we add custom CSS
