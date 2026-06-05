"""Helper for building curriculum notebooks programmatically.

Each curriculum notebook follows the same 6-section template. Rather
than hand-rolling JSON each time (with the cell-id boilerplate jupyter
warns about), modules import :func:`save_notebook` and pass a list of
``(cell_type, source)`` tuples.

Usage::

    from notebooks._build_notebook import md, code, save_notebook

    save_notebook(
        path="01_python_for_matlab_users/01.1_syntax_basics.ipynb",
        cells=[
            md("# 01.1 — Syntax basics\\n\\n..."),
            code("x = 5\\nprint(x)"),
            ...
        ],
    )
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def md(source: str) -> dict[str, Any]:
    """Build a markdown cell from a source string."""
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str, *, raises: bool = False) -> dict[str, Any]:
    """Build a code cell from a source string.

    Parameters
    ----------
    source
        The cell source.
    raises
        When ``True``, tags the cell with ``raises-exception`` so
        ``jupyter nbconvert --execute`` keeps going past it. Use this
        for cells that intentionally trigger an error (e.g. demoing
        what a traceback looks like).
    """
    metadata: dict[str, Any] = {}
    if raises:
        metadata["tags"] = ["raises-exception"]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def save_notebook(*, path: str | Path, cells: list[dict[str, Any]]) -> Path:
    """Write a notebook to ``notebooks/<path>`` and return the absolute path.

    The kernelspec block is identical across the curriculum so callers
    don't need to repeat it.
    """
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    abs_path = Path(__file__).parent / path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(json.dumps(nb, indent=1))
    return abs_path


__all__ = ["code", "md", "save_notebook"]
