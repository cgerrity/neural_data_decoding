# ADR 015 — Two doc toolchains (MkDocs + Sphinx)

**Status**: Accepted
**Date**: 2026-07-16

## Context

The MATLAB pipeline in `Processing_Functions_cgg/` carried no separate
documentation system. What documentation existed lived in inline comments and
in the original author's head — there was no browsable reference, no
prose "why" layer, and no CI gate against documentation drift. Porting to a
standalone Python library made that gap untenable: a newcomer must be able to
learn the pipeline, look up any public symbol, and understand the design
rationale without the author present.

Two distinct documentation needs fall out of that goal, and they pull toward
different tools:

- **Narrative / prose docs** — quickstart, user guides, cookbook, concepts,
  deployment, glossary, ADRs. These are hand-written Markdown, want a polished
  reading UX, and change deliberately at decision time. MkDocs Material is the
  strongest fit: Markdown-native, a curated `nav:` tree, and a good search /
  theme story.
- **API reference** — auto-generated from the NumPy-style docstrings that
  Critical Note #26 makes mandatory and CI-enforced. This wants mature
  `autodoc` + `napoleon` docstring parsing and `intersphinx` cross-linking out
  to `torch` / `numpy` / `scipy` / `pandas`. Sphinx is the standard tool for
  exactly this, and its warnings-as-errors build (`-W`) doubles as the
  missing-docstring gate.

No single tool does both jobs cleanly enough to justify collapsing them, so the
repo runs two builds and drives them from one script.

## Decision

Keep **two parallel doc toolchains**, each doing what it is best at, wired
together by `scripts/build_docs.sh`.

**Narrative → MkDocs Material** (`docs/mkdocs.yml`):

- `theme: material`, `docs_dir: narrative`, output `site_dir: build/narrative`,
  `strict: true`.
- A hand-curated `nav:` tree: Home, Quickstart, User guide, Concepts, Cookbook,
  Deployment, Glossary, Troubleshooting, Contributing, and Decision records
  (the ADR index + individual ADRs).
- The `mkdocstrings` Python handler *is* configured here (`paths: [../src]`,
  `docstring_style: numpy`), so a narrative page can pull a docstring inline
  when useful — but the full browsable API reference is **not** produced by
  MkDocs; that is Sphinx's job.

**API reference → Sphinx autodoc** (`docs/api/conf.py`):

- Extensions: `autodoc`, `autosummary`, `napoleon` (NumPy docstrings),
  `viewcode`, `intersphinx`, and `myst_parser` (so Markdown sources can sit
  alongside RST).
- `conf.py` inserts `<repo>/src` onto `sys.path`, so autodoc imports the
  package without requiring an editable `pip install -e .`.
- `autosummary_generate = False` — recursion is deliberately off; the structure
  stays flat, with one hand-written `.rst` per subpackage (`data`, `models`,
  `training`, `interop`, `sweeps`) driving `:automodule:`.
- `undoc-members = False` — `interrogate` is the single source of truth for
  coverage, not autodoc.
- `autodoc_typehints = "description"` and `napoleon_use_ivar = True` keep
  signatures and dataclass attributes from being documented twice.
- `sphinx_autodoc_typehints` is intentionally **not** enabled (incompatible
  with Sphinx 9+, and redundant with the settings above).
- Theme is `furo`; `intersphinx` maps out to python / numpy / torch / pandas /
  scipy.

**Unified driver** (`scripts/build_docs.sh`):

- `set -euo pipefail`, `cd` to the repo root, then dispatch on a single
  argument: `narrative`, `api`, or `both` (default `both`).
- `build_narrative` runs `mkdocs build --strict` → `docs/build/narrative/`.
- `build_api` runs `sphinx-build -W -b html docs/api docs/build/api` →
  `docs/build/api/`.

**Honest scope note.** The script builds the two sites into **sibling output
directories** under `docs/build/`; it does **not** currently merge them into one
site with a shared navigation bar, and it does **not** invoke `mike`. The
`mkdocs.yml` header comment ("assembles the unified site") and the `mike`
versioning noted in that file are aspirational — the true present behavior is
two independent HTML trees produced by one script.

## Consequences

**Positive**

- Each tool is used for what it is genuinely good at: Markdown-native prose in
  MkDocs Material, mature docstring autodoc + `intersphinx` cross-linking in
  Sphinx.
- The Sphinx `-W` build is a real CI gate — a missing or malformed docstring
  fails the build, satisfying Critical Note #26 with no extra machinery.
- Contributors author narrative pages in plain Markdown (low friction) while
  the API reference regenerates itself from docstrings the moment code lands.
- One entry point (`scripts/build_docs.sh both`) reproduces the full CI docs
  build locally.

**Negative**

- Two toolchains means two dependency sets and two config files to keep healthy
  (`mkdocs.yml` + `conf.py`), and two failure modes to understand.
- The outputs are **not** yet stitched into a single navigable site; a reader
  crossing from narrative docs to the API reference moves between two separate
  HTML trees. Unified nav is deferred work, and the header comment overstates
  the current state.
- `mike`-based versioning is documented but unwired, so there is a gap between
  the stated intent and the shipped script that a future editor must not mistake
  for working behavior.
- Docstring style must stay strictly NumPy across the codebase so both
  `napoleon` (Sphinx) and `mkdocstrings` (MkDocs) parse it identically.

## Alternatives considered

1. **Sphinx for everything** (narrative in RST/MyST + autodoc). Rejected:
   authoring the large prose surface (user guides, cookbook, concepts,
   deployment) is markedly more pleasant in Markdown with MkDocs Material's
   curated `nav:` and theme than in RST, even with `myst_parser` available.

2. **MkDocs + `mkdocstrings` for everything**, including the API reference.
   Rejected as the *primary* reference generator: at porting time Sphinx's
   `autodoc` + `napoleon` + `intersphinx` was the more battle-tested path for a
   fully auto-generated reference that cross-links into `torch` / `numpy` /
   `scipy`. The decision is only partially hedged — `mkdocstrings` *is* wired
   into `mkdocs.yml` for inline docstring use, but Sphinx owns the reference.

3. **No separate API site** — rely on in-editor docstrings and the Milestone E
   notebooks. Rejected: reference lookup needs a browsable, cross-linked site,
   and the Sphinx `-W` build is what enforces docstring coverage in CI
   (Critical Note #26).

4. **One merged output site** built by a stitching step in `build_docs.sh`.
   Considered and consistent with the long-term intent, but **not implemented**:
   the script currently emits two sibling trees. Unifying them (shared nav +
   `mike` versioning) is left as future work rather than claimed as done.

## References

- Narrative build config: `docs/mkdocs.yml`.
- API reference build config: `docs/api/conf.py` and the subpackage pages
  `docs/api/{data,models,training,interop,sweeps}.rst`.
- Unified build driver: `scripts/build_docs.sh`.
- Docstring-coverage / CI gate rationale: migration Critical Note #26 (Sphinx
  `-W` + `interrogate --fail-under=100`).
- Cross-reference scope of the two sites is narrowed further in
  [ADR 016](016_minimal_matlab_cross_referencing_in_api_docs.md).
- Contributor-facing extend/build workflow:
  `notebooks/09_production_deployment/09.6_extending_the_pipeline.ipynb`.
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) ·
  [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).
