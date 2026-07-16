# Contributing

Conventions and checks for working on the codebase.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
nbstripout --install          # per clone: strip notebook outputs at commit
```

## The checks that gate a change

| Check | Command | Bar |
|---|---|---|
| Tests | `python -m pytest` | the default suite passes (~8s) |
| Static types | `pyright` | zero errors project-wide |
| Docstring coverage | `interrogate` | 100% |
| MATLAB-gated parity | `python -m pytest -m needs_matlab` | requires MATLAB + source root |

The default test run **deselects** `needs_matlab` (MATLAB-spawning parity
tests), so it runs anywhere. Run those explicitly when you have MATLAB.

## Style

- **NumPy-style docstrings.** The class docstring carries `Parameters` (not
  `__init__`). Match the surrounding code's comment density and idioms.
- **Cite the MATLAB source** a function ports, and the relevant Critical Note
  number, in the docstring.
- **Don't fabricate parity values.** Verify empirically against MATLAB fixtures —
  the migration plan's example numbers have been wrong before (the reconstruction
  normalization was caught by an empirical probe, not by reading the plan).

## Extending vs modifying

Prefer **registering** over editing core dispatchers (the open/closed
principle). See the cookbook:

- [Add a new architecture](cookbook/add_a_new_architecture.md)
- [Add a new loss component](cookbook/add_a_new_loss_component.md)
- [Add a new curriculum schedule](cookbook/add_a_new_curriculum_schedule.md)
- [Add a new target task](cookbook/add_a_new_target_task.md)

## Notebooks

The curriculum notebooks under `notebooks/` are a first-class deliverable. They
are authored via a helper (`notebooks/_build_notebook.py`), must **execute clean**
via `jupyter nbconvert --to notebook --execute` with verified outputs, and follow
a 6-section template. `nbstripout` strips outputs at commit, so diffs stay small;
cell IDs are committed.

## Decision records

Architectural decisions are recorded as ADRs under `adrs/` — see
[the first ADR](adrs/001_tiered_parity_not_bit_exact.md) for the format. Add one
when you make a decision future contributors would otherwise have to reverse-
engineer.

## Docs

This site is MkDocs Material (`docs/mkdocs.yml`); the API reference is
Sphinx-generated (`docs/api/`). Build both with `scripts/build_docs.sh`. The
narrative build runs with `--strict`, so internal links must resolve.
