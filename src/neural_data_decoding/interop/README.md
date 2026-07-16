# `neural_data_decoding.interop` — the MATLAB bridge

This subpackage is the **bounded** MATLAB-compatibility surface. Everything that
must match MATLAB conventions so the *unchanged* MATLAB analysis pipeline
(`DATA_cggAllNetworkEncoderResults.m` → `FIGURE_cggAllNetworkEncoderResults.m`,
plus `cgg_plotParameterSweep.m`) can consume Python-trained output lives here.
The rest of the codebase stays free to be idiomatic Python; only this subpackage
knows about MATLAB-side requirements. It is the T4 (output-format) parity tier.

## What it actually is (not just what the name implies)

"MAT interop" is **not purely `.mat`**. The contract MATLAB depends on is three
things: two `.mat` files, **one YAML**, and a **folder-naming convention**.
Network weights, monitor tables, and resume checkpoints are Python-internal and
are *not* contractual to MATLAB. Two more honesty notes worth reading before you
rely on this code:

- **`CM_Table` is a SciPy struct-of-arrays, not a genuine MATLAB `table`.**
  `scipy.io.savemat` cannot emit a `table`, so `write_cm_table_mat` writes a
  single key `CM_Table` mapping to a dict of column arrays. MATLAB `load`s it as
  a *struct*; field access is identical downstream. To materialize a real
  `table` on disk you must run `promote_struct_to_table`, which shells out to
  MATLAB and calls `struct2table` — a hand-off step, not on the training path.
- **`weight_converter` is MATLAB → PyTorch only, despite its module title
  saying "Bidirectional."** Only the `load_matlab_*` (weight import) and
  `matlab_*_to_pytorch_btc` (axis-order) directions exist today; there is **no**
  PyTorch → MATLAB export function. It covers GRU / LSTM encoders and the
  variational composite, used for T2 forward-parity fixtures.

## Key entry points

- **`build_matlab_run_dirs(...) -> MatlabRunDirs`** (`folder_hierarchy_matlab.py`)
  — a *pure* function (no filesystem access) that reproduces MATLAB's
  deeply-nested, config-encoded path; returns the `classifier_fold`,
  `autoencoder_fold`, and `encoding_dir` the aggregator walks. The caller does
  the `mkdir`.
- **`write_cm_table_mat(...)`** (`cm_table_format.py`) — the **primary** interop
  output. Writes `CM_Table_Validation.mat` (per best-validation epoch) and
  `CM_Table.mat` (once, from restored Optimal weights on the test set — the
  reported number). Confidence columns are filled with ones when the heads are
  disabled, so the struct is always complete.
- **`write_encoding_parameters_yaml(...)`** (`parameter_yaml.py`) — a
  stable-schema `EncodingParameters.yaml` writer. Emits *every* field in a schema
  template (filling defaults for anything the run didn't set) so YAMLs are
  field-symmetric across a sweep, which `cgg_plotParameterSweep.m` requires.
  Translates snake_case → MATLAB key names via `PYTHON_TO_MATLAB_KEY` (PascalCase
  fallback).

Supporting modules: `matlab_runner.py` (`run_matlab_batch`, `matlab_available`,
`find_matlab_executable` — drives `matlab -batch` as a subprocess, with an Apple
Silicon `arch -arm64` workaround for Rosetta Python) and `matlab_table_writer.py`
(`promote_struct_to_table` / `promote_structs_to_tables` — the struct → real
`table` shim).

## References

- ADR: [007 — MAT interop surface](../../../docs/narrative/adrs/007_mat_interop_surface.md)
  (the three-part contract and why nothing else is mirrored). Its parity tier is
  set by [001 — Tiered parity, not bit-exact](../../../docs/narrative/adrs/001_tiered_parity_not_bit_exact.md)
  (T4, output-format).
- Docs: API reference `docs/api/interop.rst`; sweep-comparison cookbook
  [`compare_two_sweep_configs.md`](../../../docs/narrative/cookbook/compare_two_sweep_configs.md).
- Notebooks — Module 08 (Output & Analysis): `08.1_folder_hierarchy_generation`,
  `08.2_writing_mat_files_for_matlab`, `08.4_the_mat_round_trip_test`,
  `08.6_running_matlab_analysis_on_python_output`.
