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
Sphinx-generated (`docs/api/`). Build both with `bash scripts/build_docs.sh both`
(or `narrative` / `api` for one). The narrative build runs with `--strict` and
the API build with `-W`, so **broken internal links and missing docstrings fail
the build** — fix them before pushing.

## Continuous integration

Two GitHub Actions workflows run on every push to `main` and every PR:

- **`.github/workflows/ci.yml`** — `pytest` (the default suite; MATLAB-gated
  tests are deselected) and execution of every curriculum notebook via
  `nbconvert`.
- **`.github/workflows/docs.yml`** — `bash scripts/build_docs.sh both`
  (`mkdocs --strict` + `sphinx -W`) plus `interrogate --fail-under=100`.

Run the same gates locally before pushing: `python -m pytest`,
`bash scripts/build_docs.sh both`, `interrogate --fail-under=100 src/`.

## Publishing the docs (versioned, via `mike`)

Publishing is **opt-in and manual** — nothing is auto-published. The narrative
site is versioned with [`mike`](https://github.com/jimporter/mike). To publish
(order matters — the `gh-pages` branch must exist before Pages can point at it):

1. **Run the `deploy` job once** to create the `gh-pages` branch: GitHub
   **Actions → docs → Run workflow**, entering a version label (e.g. `0.1`). It
   runs `mike deploy --push --update-aliases <version> latest` and
   `mike set-default --push latest`, pushing the built site to `gh-pages`.
   (Locally instead: `cd docs && mike deploy --push <version> latest`.)
2. **Enable GitHub Pages** to serve that branch: repo **Settings → Pages →
   Build and deployment → Source: "Deploy from a branch" → `gh-pages` / `(root)`
   → Save**. The site then goes live at
   `https://<owner>.github.io/neural_data_decoding/`.

Note: **GitHub Pages on a *private* repo requires a paid plan** (Pro / Team /
Enterprise). On a free plan a private repo cannot host Pages without making the
site public — confirm the repo's visibility/plan before publishing proprietary
docs.

To publish locally instead: `cd docs && mike deploy --push <version> latest`.
The Sphinx API site (`docs/build/api`) is built and verified in CI but is not
yet wired into the `mike` publish flow (see ADR 015).
