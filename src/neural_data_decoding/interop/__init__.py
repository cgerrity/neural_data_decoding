"""MATLAB ↔ Python interop layer.

This subpackage is the **bounded** MATLAB-compatibility surface. Everything
that has to match MATLAB conventions for the analysis pipeline to consume
Python output lives here:

* :mod:`folder_hierarchy_matlab` — MATLAB-parity 18-level long-folder
  layout (``Aggregate Data / Epoched Data / Epoch / Encoding / Target /
  ModelName / ... / Fold_N``). Output is discoverable by
  ``DATA_cggAllNetworkEncoderResults.m`` unchanged (Critical Note #15).
* :mod:`cm_table_format` — the ``CM_Table.mat`` schema (primary interop
  output; Critical Note #16).
* :mod:`parameter_yaml` — stable-schema ``EncodingParameters.yaml`` writer
  (Critical Note #25).

The rest of the codebase stays free to be idiomatic Python; only this
subpackage knows about MATLAB-side requirements.
"""

from neural_data_decoding.interop.cm_table_format import (
    TEST_CM_TABLE_FILENAME,
    VALIDATION_CM_TABLE_FILENAME,
    write_cm_table_mat,
)
from neural_data_decoding.interop.folder_hierarchy_matlab import (
    MatlabRunDirs,
    build_matlab_run_dirs,
)
from neural_data_decoding.interop.matlab_runner import (
    MatlabNotFoundError,
    find_matlab_executable,
    matlab_available,
    run_matlab_batch,
)
from neural_data_decoding.interop.matlab_table_writer import (
    describe_table_mat,
    promote_struct_to_table,
    promote_structs_to_tables,
)
from neural_data_decoding.interop.parameter_yaml import (
    ENCODING_PARAMETERS_FILENAME,
    read_encoding_parameters_yaml,
    write_encoding_parameters_yaml,
)
from neural_data_decoding.interop.weight_converter import (
    load_matlab_composite_weights,
    load_matlab_gru_encoder_weights,
    load_matlab_lstm_encoder_weights,
    matlab_cbt_to_pytorch_btc,
    matlab_ctb_to_pytorch_btc,
)

__all__ = [
    "ENCODING_PARAMETERS_FILENAME",
    "MatlabNotFoundError",
    "MatlabRunDirs",
    "TEST_CM_TABLE_FILENAME",
    "VALIDATION_CM_TABLE_FILENAME",
    "build_matlab_run_dirs",
    "describe_table_mat",
    "find_matlab_executable",
    "load_matlab_composite_weights",
    "load_matlab_gru_encoder_weights",
    "load_matlab_lstm_encoder_weights",
    "matlab_available",
    "matlab_cbt_to_pytorch_btc",
    "matlab_ctb_to_pytorch_btc",
    "promote_struct_to_table",
    "promote_structs_to_tables",
    "read_encoding_parameters_yaml",
    "run_matlab_batch",
    "write_cm_table_mat",
    "write_encoding_parameters_yaml",
]
