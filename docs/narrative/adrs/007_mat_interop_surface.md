# ADR 007 — MAT interop surface

**Status**: Accepted
**Date**: 2026-07-16

## Context

The Python port replaces the MATLAB *training* pipeline but deliberately keeps
the MATLAB *analysis* pipeline (`DATA_cggAllNetworkEncoderResults.m` →
`FIGURE_cggAllNetworkEncoderResults.m`, plus the sweep plotter
`cgg_plotParameterSweep.m`). Those scripts must load Python-trained output
*unmodified* — this is the T4 (output-format) tier of
[ADR 001](001_tiered_parity_not_bit_exact.md).

A MATLAB run emits many `.mat` artifacts into each fold directory: per-trial
confusion-matrix telemetry, per-epoch monitor/validation tables, serialized
network weights (`*-Current.mat`, `*-Optimal.mat`), the checkpoint iteration
counter, and an `EncodingParameters.yaml`. The tempting instinct is to mirror
*all* of it so the two pipelines are indistinguishable on disk.

But the migration plan pins down which artifacts the *analysis* actually reads.
Critical Note #16 states plainly: `CM_Table` is the **primary** interop output,
and "other `.mat` artifacts (network weights, etc.) are only needed for
parity-comparison fixtures, not for the analysis pipeline." Critical Note #25
adds that `cgg_plotParameterSweep.m` needs one more thing — a *field-symmetric*
`EncodingParameters.yaml` per run. Critical Note #15 supplies the third leg: the
analysis discovers runs by *walking a deterministically-named folder tree*, so
whatever we write is only found if it lands at a byte-exact path.

## Decision

Expose a **narrow, three-part interop surface** — nothing else the Python
pipeline writes is contractual to MATLAB:

1. **`CM_Table.mat` (+ `CM_Table_Validation.mat`)** — `write_cm_table_mat` in
   `interop/cm_table_format.py`. Six field *kinds*, one row per trial:
   `DataNumber` (`single`, `N×1` — a global, sparse trial id, *not* `1..N`),
   `TrueValue`, `Window_1 … Window_K`, `Aggregation_Prediction`, and
   `TaskConfidence` (all `double`, `N×D`), plus `TrialConfidence` (`double`,
   `N×1`). Notably it is serialized as a **SciPy struct of arrays** — a single
   `savemat` key `CM_Table` mapping to a dict of column arrays — **not** a
   MATLAB `table` (`scipy.io.savemat` cannot emit a `table`). MATLAB `load`s
   this as a struct; downstream field access (`CM_Table.Aggregation_Prediction`)
   works identically on structs and tables, with a `struct2table` shim living
   on the MATLAB side. The call fixes `do_compression=True` and
   `oned_as="column"` so 1-D columns keep MATLAB's column orientation. Two
   filenames matter: `CM_Table_Validation.mat` (rewritten per best-validation
   epoch, for model selection) and `CM_Table.mat` (written once from the
   restored Optimal weights on the held-out test set — the reported number).
   When the confidence heads are disabled, both confidence fields are filled
   with **ones**, so the struct is always complete.

2. **`EncodingParameters.yaml`** — `write_encoding_parameters_yaml` in
   `interop/parameter_yaml.py`. It takes a **schema template** listing *every*
   field that must appear plus the (possibly incomplete) run config, and emits
   every template field — filling the template default for anything the run
   didn't override. This makes YAMLs *field-symmetric across a sweep* even when
   values are defaults or zeros, which is exactly what `cgg_plotParameterSweep.m`
   needs to scan field-by-field. Python `snake_case` keys are translated to
   MATLAB's on-disk names at write time via the `PYTHON_TO_MATLAB_KEY` override
   table with a PascalCase fallback; `sort_keys=False` preserves template order.

3. **A byte-exact folder tree** — `build_matlab_run_dirs` in
   `interop/folder_hierarchy_matlab.py`. A *pure* function (no filesystem
   access) that reproduces MATLAB's deeply-nested, config-encoded path where
   each level is a formatted slice of the config (`Initial Learning Rate -
   1.00e-03 ~ …`), using helpers (`_fmt_exp`, `_fmt_hidden_size`, …) that each
   mirror a specific MATLAB `sprintf` pattern. It returns a `MatlabRunDirs`
   triple — `classifier_fold` (the leaf `Fold_{N}/` that receives the two
   CM_Tables, the YAML, and checkpoints), `autoencoder_fold`
   (`Information/Fold_{N}/` for Stage-1 weights), and `encoding_dir` (the
   discovery root the aggregator walks down from). The caller does the `mkdir`.

The surface's job is discovery: because the two contractual files land at
byte-exact paths, `DATA_cggAllNetworkEncoderResults.m` finds and groups
Python-trained runs with no modification.

Note that the surface named "MAT interop" is not purely `.mat`: it is two
`.mat` files **plus** one YAML **plus** a folder-naming contract. Network
weights, monitor tables, and the resume checkpoint remain Python-internal.

## Consequences

**Positive**

- Only three write paths must stay byte-compatible with MATLAB, so the port is
  free to store weights, optimizer state, and checkpoints in whatever
  Python-native format is convenient.
- The `savemat`-struct approach avoids reimplementing MATLAB's `table`
  container in Python while remaining loadable by the unchanged analysis
  scripts.
- The stable-schema YAML makes parameter sweeps plottable without the plotter
  ever hitting a missing field.
- The interop contract is small and testable: it is pinned by fixture-based
  parity tests (a real `106×4×59` `CM_Table.mat`, a reference YAML, and
  path-parity tests) rather than by a broad "match everything" goal.

**Negative**

- The contract is *implicit* on the MATLAB side — the analysis assumes exact
  field names, dtypes (`single` vs `double`), column orientation, and path
  strings. A mismatch fails *silently*: the aggregator simply skips a run whose
  path formatting differs (e.g. `0.001` where MATLAB writes `1.00e-03`), or the
  plots come out subtly wrong, with no error raised.
- Every new config field needs a `PYTHON_TO_MATLAB_KEY` entry (or must be
  correct under the PascalCase fallback) and must be added to the sweep schema
  template, or the YAML surface silently drifts from MATLAB's field set.
- Because a MATLAB `table` isn't actually written, the interop depends on the
  MATLAB-side `struct2table` shim staying in place for any table-typed caller.

## Alternatives considered

1. **Mirror all MATLAB `.mat` output** (weights, monitor tables, checkpoint
   counter, in MATLAB's exact formats). Rejected: Critical Note #16 confirms the
   analysis reads none of those; mirroring them would couple the port to MATLAB
   serialization details that no downstream consumer needs, purely for
   cosmetic sameness.

2. **Write a genuine MATLAB `table` for `CM_Table`.** Rejected: `scipy.io.savemat`
   cannot emit a `table`, and the struct-of-arrays form is field-access-identical
   downstream via the existing `struct2table` shim — a real `table` would require
   a MATLAB round-trip in the Python build with no analysis benefit.

3. **Emit only the fields a given run populates in `EncodingParameters.yaml`.**
   Rejected by Critical Note #25: `cgg_plotParameterSweep.m` scans field-by-field
   across a sweep and breaks on any missing field, so YAMLs must be
   field-symmetric — hence the schema-template writer.

4. **Use a flat run-id directory plus a metadata sidecar** instead of the
   config-encoded folder tree. Rejected by Critical Note #15: the MATLAB
   aggregator discovers and groups runs by *walking the named tree*, so a
   different layout is invisible to it.

## References

- CM_Table writer + schema: `src/neural_data_decoding/interop/cm_table_format.py`
  (`write_cm_table_mat`); migration Critical Note #16 (`docs/PLAN.md`);
  notebook `notebooks/08_output_and_analysis/08.2_writing_mat_files_for_matlab.ipynb`.
- Stable-schema YAML writer: `src/neural_data_decoding/interop/parameter_yaml.py`
  (`write_encoding_parameters_yaml`, `PYTHON_TO_MATLAB_KEY`); migration Critical
  Note #25 (`docs/PLAN.md`).
- Folder-hierarchy discovery paths: `src/neural_data_decoding/interop/folder_hierarchy_matlab.py`
  (`build_matlab_run_dirs`, `MatlabRunDirs`); migration Critical Note #15
  (`docs/PLAN.md`); notebook
  `notebooks/08_output_and_analysis/08.1_folder_hierarchy_generation.ipynb`.
- MATLAB consumers: `DATA_cggAllNetworkEncoderResults.m` (results aggregator),
  `cgg_plotParameterSweep.m` (sweep plotter).
- Parity tier: [ADR 001 — Tiered parity, not bit-exact](001_tiered_parity_not_bit_exact.md) (T4, output-format parity).
