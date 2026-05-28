"""MATLAB ↔ Python interop layer.

This subpackage is the **bounded** MATLAB-compatibility surface. Everything
that has to match MATLAB conventions for the analysis pipeline to consume
Python output lives here:

* :mod:`folder_hierarchy` — deterministic result-directory naming.
* :mod:`cm_table_format` — the ``CM_Table.mat`` schema (primary interop
  output; Critical Note #16).
* :mod:`parameter_yaml` — stable-schema ``EncodingParameters.yaml`` writer
  (Critical Note #25).

The rest of the codebase stays free to be idiomatic Python; only this
subpackage knows about MATLAB-side requirements.
"""

from neural_data_decoding.interop.cm_table_format import (
    TRAINING_CM_TABLE_FILENAME,
    VALIDATION_CM_TABLE_FILENAME,
    write_cm_table_mat,
)
from neural_data_decoding.interop.folder_hierarchy import build_result_dir
from neural_data_decoding.interop.matlab_table_writer import (
    promote_struct_to_table,
)
from neural_data_decoding.interop.parameter_yaml import (
    ENCODING_PARAMETERS_FILENAME,
    read_encoding_parameters_yaml,
    write_encoding_parameters_yaml,
)
from neural_data_decoding.interop.weight_converter import (
    load_matlab_gru_encoder_weights,
    matlab_cbt_to_pytorch_btc,
    matlab_ctb_to_pytorch_btc,
)

__all__ = [
    "ENCODING_PARAMETERS_FILENAME",
    "TRAINING_CM_TABLE_FILENAME",
    "VALIDATION_CM_TABLE_FILENAME",
    "build_result_dir",
    "load_matlab_gru_encoder_weights",
    "matlab_cbt_to_pytorch_btc",
    "matlab_ctb_to_pytorch_btc",
    "promote_struct_to_table",
    "read_encoding_parameters_yaml",
    "write_cm_table_mat",
    "write_encoding_parameters_yaml",
]
