# ADR 016 — Minimal MATLAB cross-referencing in API docs

**Status**: Accepted
**Date**: 2026-07-16

## Context

The whole pipeline is a Python port of the MATLAB sources in
`Processing_Functions_cgg/`, and the MATLAB reference was consulted line by line
during the port. The tempting default for the API reference is therefore to make
every public symbol point back at its origin: a boilerplate "MATLAB equivalent:
`cgg_foo.m`" section on each function, or hyperlinks from the rendered docs into
the `.m` files. That would turn the API reference into a bilingual concordance
whose entries are only fully legible with the MATLAB tree open alongside.

Two facts make that a bad fit here:

1. **The MATLAB sources are not part of this project.** They live outside the
   repo, resolved at runtime via `NDD_MATLAB_SOURCE_ROOT`, and are never
   published with the docs. There is nothing on the doc host to link *to* — any
   "link" to `cgg_lossELBO_v2.m` would be a dangling reference.
2. **The audience going forward reads Python, not MATLAB.** Once the port is
   trusted, a reader debugging the loss aggregator should not have to
   context-switch into a language they may not run. The Python pipeline is the
   artifact of record; the MATLAB lineage is provenance, not a prerequisite.

The user's explicit choice (recorded in the Milestone F documentation spec) was
"minimal" cross-referencing: the API docs should read like a standard PyTorch
library, with MATLAB origin concentrated in a few narrative places rather than
smeared across every docstring.

## Decision

The API reference reads standalone, and the code produces exactly that today.

**The toolchain carries no MATLAB linking machinery.** `docs/api/conf.py` is a
plain Sphinx `autodoc` + `napoleon` build over the NumPy-style docstrings. Its
`intersphinx_mapping` targets `python`, `numpy`, `torch`, `pandas`, and `scipy`
— there is no MATLAB inventory, and no `linkcode` or `extlinks` extension. The
per-subpackage `.rst` files (`interop.rst`, `models.rst`, …) are bare
`.. automodule::` stubs, so every rendered API page is 100% docstring-derived.
MATLAB names in the docs can therefore only ever appear as inline literal text,
never as a hyperlink and never as an auto-resolved cross-reference.

**Docstrings do cite MATLAB inline — as provenance, not as required reading.**
This is the honest nuance the title understates: the docstrings are *not*
MATLAB-free. Roughly 64 source files name a MATLAB symbol, and they do so
deliberately where parity is load-bearing — e.g. `losses/elbo.py` cites
`cgg_lossELBO_v2.m` and reproduces the `l2loss` normalization probe;
`interop/cm_table_format.py` cites `DATA_cggAllNetworkEncoderResults.m` and
reproduces the full `CM_Table` schema; `losses/classification.py` names
`crossentropy(Y, T, Weights)` and `cgg_softmaxLayer('SCT')`. The discipline that
keeps the docs standalone is that **each docstring reproduces the behavior in
prose** — the actual normalization result, the actual schema, the actual masking
rule — so the MATLAB citation is a breadcrumb for a parity auditor, never a
lookup the reader must perform to understand what the Python does.

**What "minimal" concretely forbids.** No function carries a formal, templated
"MATLAB equivalent" section; the concentrated MATLAB↔Python name mapping lives
in exactly the three narrative surfaces the Milestone F policy names — the
project `README.md`, the one-shot table in `docs/narrative/glossary.md`, and the
relevant ADR (where a MATLAB design actually drove a Python choice). The API
reference itself stays free of that mapping table.

So "reads standalone" means *comprehensible without opening a `.m` file*, not
*free of MATLAB mentions*. Both are true of the code today: the reference is
self-contained, and its inline MATLAB citations are provenance that a reader can
safely ignore. This is the API-side complement to the two-toolchain split in
[ADR 015 — Two doc toolchains (MkDocs + Sphinx)](015_two_doc_toolchains_mkdocs_plus_sphinx.md):
Sphinx carries only the standalone reference with inline provenance breadcrumbs,
while the MkDocs narrative owns the concentrated MATLAB mapping.

## Consequences

**Positive**

- The API reference is legible to a pure-Python reader with no MATLAB install
  and no access to `Processing_Functions_cgg/`. Nothing in the rendered docs
  depends on a source tree that isn't shipped.
- No dangling links: because MATLAB names are inline literals, not hyperlinks,
  the `-W` (warnings-as-errors) Sphinx build can never fail on an unresolvable
  MATLAB cross-reference.
- The MATLAB↔Python mapping has a single home per audience — glossary/README for
  lookup, ADRs for rationale — instead of drifting across hundreds of
  per-function stanzas that would each need maintenance.
- Provenance is preserved where it matters most: the load-bearing parity notes
  (e.g. the `l2loss` batch-size normalization) stay attached to the exact code
  they justify, so an auditor can trace a decision without a scavenger hunt.

**Negative**

- "Minimal cross-referencing" is a convention, not an enforced invariant. A
  contributor could add a templated "MATLAB equivalent" section, or over-cite
  MATLAB in a docstring, and nothing in the build would reject it. The policy
  relies on review and the examples set by existing docstrings.
- The line between "provenance breadcrumb" and "required reading" is a judgment
  call. A docstring that cites MATLAB but under-explains the behavior would
  quietly break the standalone property; only prose review catches that.
- A reader who *does* want to compare against MATLAB gets a name, not a
  location — they must resolve `NDD_MATLAB_SOURCE_ROOT` and grep the source
  tree themselves. That friction is intentional but real.

## Alternatives considered

1. **Per-function "MATLAB equivalent" sections.** Rejected: it bloats every
   docstring with a maintenance-heavy mapping, makes the reference read like a
   translation table rather than a library, and pushes the reader toward a
   source tree that isn't published.

2. **Hyperlink docstrings into the `.m` sources (via `linkcode`/`extlinks`).**
   Rejected as physically impossible here — the MATLAB files are external to the
   repo and absent from the doc host, so every link would dangle. It would also
   break the warnings-as-errors build.

3. **Strip all MATLAB mentions from docstrings for a "pure" library feel.**
   Rejected: the load-bearing parity notes (why reconstruction divides by batch
   size, why `TaskConfidence` is per-dimension) are provenance the next
   maintainer genuinely needs. Deleting them would trade a cosmetic win for lost
   institutional knowledge. The chosen middle path keeps the citations but
   demands standalone prose around them.

4. **Put the MATLAB mapping in the API reference instead of the narrative.**
   Rejected: it duplicates the glossary, couples the auto-generated reference to
   hand-maintained tables, and contradicts the ADR 015 division of labor between
   the Sphinx API build and the MkDocs narrative.

## References

- Toolchain with no MATLAB link machinery: `docs/api/conf.py` — `autodoc` +
  `napoleon` over NumPy docstrings; `intersphinx_mapping` covers
  python/numpy/torch/pandas/scipy only (no MATLAB inventory); no `linkcode` or
  `extlinks`. The API pages are bare `.. automodule::` stubs in
  `docs/api/interop.rst`, `docs/api/models.rst`, etc.
- Inline-provenance-but-standalone docstrings (representative):
  `src/neural_data_decoding/training/losses/elbo.py` (cites `cgg_lossELBO_v2.m`,
  reproduces the `l2loss` probe),
  `src/neural_data_decoding/interop/cm_table_format.py` (cites
  `DATA_cggAllNetworkEncoderResults.m`, reproduces the `CM_Table` schema),
  `src/neural_data_decoding/training/losses/classification.py` (cites
  `crossentropy` and `cgg_softmaxLayer('SCT')`).
- Concentrated MATLAB↔Python mapping (the only places it lives): `README.md` and
  the one-shot table in `docs/narrative/glossary.md`.
- Governing policy: `docs/PLAN.md`, Milestone F — "Cross-reference policy (per
  user choice — minimal)" and "Per-function docstrings do not include 'MATLAB
  equivalent' sections." The port carries no dedicated numbered note for the
  doc policy itself; the inline citations it permits are the load-bearing parity
  notes, e.g. Critical Note #16 (`CM_Table` as the primary interop surface) and
  Critical Note #38 (NaN-masked reconstruction normalization).
- Walkthrough: `notebooks/00_orientation/00.3_the_matlab_to_python_mental_model.ipynb`
  — where the reader is taught to treat the Python pipeline as the artifact of
  record and MATLAB as lineage, not a required lookup.
- Related decision: [ADR 015 — Two doc toolchains (MkDocs + Sphinx)](015_two_doc_toolchains_mkdocs_plus_sphinx.md).
