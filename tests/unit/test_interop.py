"""Tests for :mod:`neural_data_decoding.interop`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from neural_data_decoding.interop import (
    ENCODING_PARAMETERS_FILENAME,
    VALIDATION_CM_TABLE_FILENAME,
    MatlabRunDirs,
    build_matlab_run_dirs,
    read_encoding_parameters_yaml,
    write_cm_table_mat,
    write_encoding_parameters_yaml,
)
from neural_data_decoding.interop.parameter_yaml import translate_key


# ───────────────────────── folder_hierarchy_matlab ─────────────────────────


def _optimal_cfg(**overrides: object) -> dict[str, object]:
    """Return an Optimal-like cfg dict for the folder builder."""
    base: dict[str, object] = {
        "fold": 1,
        "epoch": "Decision",
        "target": "Dimension",
        "model_name": "GRU",
        "is_variational": True,
        "encoder_output_type": "Stochastic",
        "activation": "",
        "dropout": 0.5,
        "want_normalization": False,
        "bottle_neck_depth": 1,
        "data_width": 100,
        "window_stride": 50,
        "start_end_percent": [None, None],
        "normalization": "Channel - Z-Score",
        "hidden_sizes": [1000, 500, 250],
        "initial_learning_rate": 1e-3,
        "gradient_threshold": 100.0,
        "gradient_clip_type": "Global",
        "optimizer": "ADAM",
        "l2_factor": 1e-4,
        "mini_batch_size": 100,
        "max_worker_mini_batch_size": 100,
        "want_stratified_partition": True,
        "std_channel_offset": 0.03,
        "std_white_noise": 0.015,
        "std_random_walk": 7e-4,
        "std_time_shift": 100.0,
        "want_separate_time_shift": True,
        "subset": True,
        "loss_type_decoder": "MSE",
        "num_epochs_autoencoder": 0,
        "prior_proportion": 0.9,
        "rescale_loss_epoch": 0,
        "weight_reconstruction": 100.0,
        "weight_classification": 10.0,
        "weight_kl": 1.0,
        "weight_confidence": 1.0,
        "confidence_type": ["Trial", "Task"],
        "want_batch_correction": False,
        "dynamic_parameter_set": "Soft Three-Stage Curriculum - Shortened",
        "stitching_and_fusion_layer": "",
        "classifier_name": "Deep LSTM - Dropout 0.5",
        "classifier_hidden_size": [250, 100, 50],
        "weighted_loss": "Inverse",
        "multiple_instance_learning_type": "MIL",
    }
    base.update(overrides)
    return base


def test_matlab_run_dirs_classifier_leaf_is_fold_N(tmp_path: Path) -> None:
    """Classifier path ends in ``Classifier - ... / Fold_{N}``."""
    dirs = build_matlab_run_dirs(base_dir=tmp_path, cfg=_optimal_cfg(fold=3))
    assert dirs.classifier_fold.parts[-1] == "Fold_3"
    assert dirs.classifier_fold.parts[-2].startswith("Classifier - ")


def test_matlab_run_dirs_autoencoder_branch_uses_information_label(
    tmp_path: Path,
) -> None:
    """Autoencoder fold parallels classifier under ``Information / Fold_{N}``."""
    dirs = build_matlab_run_dirs(base_dir=tmp_path, cfg=_optimal_cfg(fold=7))
    assert dirs.autoencoder_fold.parts[-1] == "Fold_7"
    assert dirs.autoencoder_fold.parts[-2] == "Information"


def test_matlab_run_dirs_top_layout_matches_aggregate_data_chain(
    tmp_path: Path,
) -> None:
    """The shallow chain is ``Aggregate Data / Epoched Data / Epoch / Encoding / Target / ModelName``."""
    dirs = build_matlab_run_dirs(base_dir=tmp_path, cfg=_optimal_cfg())
    parts = dirs.classifier_fold.parts
    # Strip tmp_path prefix
    start = len(tmp_path.parts)
    assert parts[start:start + 6] == (
        "Aggregate Data",
        "Epoched Data",
        "Decision",
        "Encoding",
        "Dimension",
        "GRU",
    )


def test_model_parameters_optimal_render() -> None:
    """The Optimal config renders ``ModelParameters`` exactly as MATLAB does."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    name = dirs.classifier_fold.parts[7]   # ModelParameters position
    assert name == (
        "Variational - Stochastic Encoder ~ Dropout - 5.00e-01 "
        "~ Bottle Neck Depth - 1"
    )


def test_width_stride_renders_data_width_and_window_stride() -> None:
    """``Data Width - 100 ~ Window Stride - 50`` for the Optimal default."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[8] == "Data Width - 100 ~ Window Stride - 50"


def test_hidden_size_hyphen_joins_multi_layer_lists() -> None:
    """``[1000, 500, 250]`` renders as ``"Hidden Size - 1000-500-250"``."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[10] == "Hidden Size - 1000-500-250"


def test_learning_combines_lr_grad_optimizer_l2() -> None:
    """``Learning`` rolls up the LR, gradient threshold + clip type, optimizer, L2."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[11] == (
        "Initial Learning Rate - 1.00e-03 ~ Gradient Threshold - 1.00e+02 - Global "
        "~ Optimizer - ADAM ~ L2 Factor - 1.00e-04"
    )


def test_mini_batch_flags_stratification() -> None:
    """``Hierarchically Stratified`` shows up for the Optimal default."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[12] == (
        "Mini Batch Size - 100 ~ Max Accumulation - 100 ~ Hierarchically Stratified"
    )


def test_data_augmentation_emits_separate_time_shift_when_flag_set() -> None:
    """``Separate TimeShift`` label appears when ``want_separate_time_shift`` is True."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[13] == (
        "Channel Offset - 3.00e-02 ~ White Noise - 1.50e-02 ~ "
        "Random Walk - 7.00e-04 ~ Separate TimeShift - 1.00e+02"
    )


def test_is_subset_flag_uses_string_when_session_name_set() -> None:
    """``cfg.subset = '<session>'`` puts the literal session name as the folder."""
    cfg = _optimal_cfg(subset="Wo_Probe_01_23_02_13_003_01")
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=cfg)
    assert dirs.classifier_fold.parts[14] == "Wo_Probe_01_23_02_13_003_01"


def test_is_subset_label_default_for_bool_true() -> None:
    """``cfg.subset = True`` → ``"Subset"`` folder."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg(subset=True))
    assert dirs.classifier_fold.parts[14] == "Subset"


def test_is_subset_label_for_bool_false() -> None:
    """``cfg.subset = False`` → ``"All Sessions"`` folder."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg(subset=False))
    assert dirs.classifier_fold.parts[14] == "All Sessions"


def test_autoencoder_with_zero_epochs_still_renders_epochs_label() -> None:
    """``AutoEncoder - Epochs - 0`` for ``num_epochs_autoencoder = 0``."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[15] == (
        "AutoEncoder - Epochs - 0 ~ Loss Function - MSE "
        "~ Prior Proportion - 9.00e-01 ~ Rescale Epochs - 0"
    )


def test_loss_renders_confidence_with_sorted_and_joined_types() -> None:
    """``Weight Task and Trial Confidence`` — confidence types alphabetized."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[16] == (
        "Weight Reconstruction - 1.00e+02 ~ Weight Classification - 1.00e+01 "
        "~ Weight KL - 1.00e+00 ~ Weight Task and Trial Confidence - 1.00e+00"
    )


def test_loss_omits_confidence_when_weight_is_zero() -> None:
    """Confidence label disappears entirely for ``weight_confidence == 0``."""
    cfg = _optimal_cfg(weight_confidence=0, confidence_type=[])
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=cfg)
    assert dirs.classifier_fold.parts[16] == (
        "Weight Reconstruction - 1.00e+02 ~ Weight Classification - 1.00e+01 "
        "~ Weight KL - 1.00e+00"
    )


def test_dynamic_includes_stitching_and_fusion_when_set() -> None:
    """``S and F`` segment appended when ``stitching_and_fusion_layer != ''``."""
    cfg = _optimal_cfg(stitching_and_fusion_layer="Default")
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=cfg)
    assert dirs.classifier_fold.parts[17] == (
        "Dynamic Set - Soft Three-Stage Curriculum - Shortened ~ S and F - Default"
    )


def test_dynamic_omits_stitching_when_empty() -> None:
    """No S+F segment when ``stitching_and_fusion_layer`` is ``''``."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[17] == (
        "Dynamic Set - Soft Three-Stage Curriculum - Shortened"
    )


def test_classifier_renders_with_mil_marker() -> None:
    """``~ SCT`` suffix when ``multiple_instance_learning_type = 'MIL'``."""
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=_optimal_cfg())
    assert dirs.classifier_fold.parts[18] == (
        "Classifier - Deep LSTM - Dropout 0.5 ~ Hidden Size - 250-100-50 "
        "~ Weighted Loss - Inverse ~ SCT"
    )


def test_classifier_omits_sct_when_no_mil() -> None:
    """No ``SCT`` suffix without MIL."""
    cfg = _optimal_cfg(multiple_instance_learning_type="None")
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=cfg)
    assert dirs.classifier_fold.parts[18] == (
        "Classifier - Deep LSTM - Dropout 0.5 ~ Hidden Size - 250-100-50 "
        "~ Weighted Loss - Inverse"
    )


def test_matlab_run_dirs_returns_consistent_encoding_dir(tmp_path: Path) -> None:
    """All three returned paths share the same Encoding/{Target} ancestor."""
    dirs = build_matlab_run_dirs(base_dir=tmp_path, cfg=_optimal_cfg())
    assert isinstance(dirs, MatlabRunDirs)
    assert dirs.encoding_dir == dirs.classifier_fold.parents[len(dirs.classifier_fold.parts) - len(dirs.encoding_dir.parts) - 1]


def test_rejects_fold_zero(tmp_path: Path) -> None:
    """MATLAB is 1-indexed."""
    with pytest.raises(ValueError, match="fold must be >= 1"):
        build_matlab_run_dirs(base_dir=tmp_path, cfg=_optimal_cfg(fold=0))


def test_rejects_empty_required_fields(tmp_path: Path) -> None:
    """An empty Epoch / Target / ModelName fails fast."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        build_matlab_run_dirs(base_dir=tmp_path, cfg=_optimal_cfg(epoch=""))
    with pytest.raises(ValueError, match="must be a non-empty string"):
        build_matlab_run_dirs(base_dir=tmp_path, cfg=_optimal_cfg(target=""))
    with pytest.raises(ValueError, match="must be a non-empty string"):
        build_matlab_run_dirs(base_dir=tmp_path, cfg=_optimal_cfg(model_name=""))


def test_omegaconf_input_is_accepted(tmp_path: Path) -> None:
    """A DictConfig (not just dict) is a valid cfg input."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(_optimal_cfg())
    dirs = build_matlab_run_dirs(base_dir=tmp_path, cfg=cfg)
    assert "Fold_1" in dirs.classifier_fold.parts


def test_start_end_percent_appends_time_range() -> None:
    """A set ``start_end_percent`` adds the ``Time Percent`` segment."""
    cfg = _optimal_cfg(start_end_percent=[0.0, 0.5])
    dirs = build_matlab_run_dirs(base_dir=Path("/"), cfg=cfg)
    assert dirs.classifier_fold.parts[8] == (
        "Data Width - 100 ~ Window Stride - 50 ~ Time Percent - [0.0, 0.5]"
    )


# ───────────────────────── cm_table_format ─────────────────────────


def test_write_cm_table_round_trip_minimal(tmp_path: Path) -> None:
    """Milestone A shape: 1 window, no confidence — fields default appropriately."""
    out = tmp_path / VALIDATION_CM_TABLE_FILENAME
    n, d = 5, 2
    data_numbers = np.arange(1, n + 1, dtype=np.int32)
    true_values = np.zeros((n, d), dtype=np.float64)
    window = np.ones((n, d), dtype=np.float64)
    write_cm_table_mat(
        out,
        data_numbers=data_numbers,
        true_values=true_values,
        window_predictions=[window],
    )

    loaded = scipy.io.loadmat(str(out), squeeze_me=False, struct_as_record=False)
    table = loaded["CM_Table"][0, 0]
    assert hasattr(table, "DataNumber")
    assert hasattr(table, "TrueValue")
    assert hasattr(table, "Window_1")
    assert hasattr(table, "Aggregation_Prediction")
    assert hasattr(table, "TrialConfidence")
    assert hasattr(table, "TaskConfidence")
    np.testing.assert_array_equal(
        table.Aggregation_Prediction, window
    )  # Aggregate defaults to the single window
    assert table.TrialConfidence.shape == (n, 1)
    assert table.TaskConfidence.shape == (n, d)
    np.testing.assert_array_equal(table.TrialConfidence, np.ones((n, 1)))
    np.testing.assert_array_equal(table.TaskConfidence, np.ones((n, d)))


def test_write_cm_table_multiple_windows(tmp_path: Path) -> None:
    """Multiple windows each produce a ``Window_k`` column; aggregation is required."""
    out = tmp_path / "CM_Table.mat"
    n, d = 4, 1
    write_cm_table_mat(
        out,
        data_numbers=np.arange(1, n + 1, dtype=np.int32),
        true_values=np.zeros((n, d), dtype=np.float64),
        window_predictions=[
            np.zeros((n, d), dtype=np.float64),
            np.ones((n, d), dtype=np.float64),
        ],
        aggregation_prediction=np.full((n, d), 0.5, dtype=np.float64),
    )

    loaded = scipy.io.loadmat(str(out), squeeze_me=False, struct_as_record=False)
    table = loaded["CM_Table"][0, 0]
    assert hasattr(table, "Window_1")
    assert hasattr(table, "Window_2")
    np.testing.assert_array_equal(table.Window_2, np.ones((n, d)))


def test_write_cm_table_requires_aggregation_for_multi_window(tmp_path: Path) -> None:
    """With >1 window the aggregation must be supplied explicitly."""
    out = tmp_path / "CM_Table.mat"
    n, d = 3, 1
    with pytest.raises(ValueError, match="aggregation_prediction is required"):
        write_cm_table_mat(
            out,
            data_numbers=np.arange(1, n + 1, dtype=np.int32),
            true_values=np.zeros((n, d), dtype=np.float64),
            window_predictions=[
                np.zeros((n, d), dtype=np.float64),
                np.ones((n, d), dtype=np.float64),
            ],
        )


def test_write_cm_table_rejects_shape_mismatch(tmp_path: Path) -> None:
    """Window predictions must match TrueValue's shape."""
    out = tmp_path / "CM_Table.mat"
    n, d = 5, 2
    with pytest.raises(ValueError, match="window_predictions"):
        write_cm_table_mat(
            out,
            data_numbers=np.arange(1, n + 1, dtype=np.int32),
            true_values=np.zeros((n, d), dtype=np.float64),
            window_predictions=[np.zeros((n, d + 1), dtype=np.float64)],
        )


def test_write_cm_table_with_confidence(tmp_path: Path) -> None:
    """Confidence columns: TrialConfidence is (N, 1); TaskConfidence is (N, D)."""
    out = tmp_path / "CM_Table.mat"
    n, d = 4, 2
    trial_conf = np.array([0.1, 0.5, 0.9, 1.0])  # (N,) — per-trial scalar
    task_conf = np.array(  # (N, D) — per-dimension
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ]
    )
    write_cm_table_mat(
        out,
        data_numbers=np.arange(1, n + 1, dtype=np.int32),
        true_values=np.zeros((n, d), dtype=np.float64),
        window_predictions=[np.zeros((n, d), dtype=np.float64)],
        trial_confidence=trial_conf,
        task_confidence=task_conf,
    )

    loaded = scipy.io.loadmat(str(out), squeeze_me=False, struct_as_record=False)
    table = loaded["CM_Table"][0, 0]
    assert table.TrialConfidence.shape == (n, 1)
    assert table.TaskConfidence.shape == (n, d)
    np.testing.assert_allclose(table.TrialConfidence.ravel(), trial_conf)
    np.testing.assert_allclose(table.TaskConfidence, task_conf)


def test_task_confidence_default_matches_truevalue_dim_count(tmp_path: Path) -> None:
    """When task_confidence is omitted, the default is (N, num_dims), not (N, 1).

    Pinned by the real MATLAB fixture in
    tests/fixtures/reference_cm_tables/CM_Table.mat where TaskConfidence is
    (106, 4) — one confidence value per classification dimension.
    """
    out = tmp_path / "CM_Table.mat"
    n, d = 5, 4  # 4 dimensions like the real Quaddle/Dimension target
    write_cm_table_mat(
        out,
        data_numbers=np.arange(1, n + 1, dtype=np.int32),
        true_values=np.zeros((n, d), dtype=np.float64),
        window_predictions=[np.zeros((n, d), dtype=np.float64)],
    )
    loaded = scipy.io.loadmat(str(out), squeeze_me=False, struct_as_record=False)
    table = loaded["CM_Table"][0, 0]
    assert table.TaskConfidence.shape == (n, d)
    assert table.TrialConfidence.shape == (n, 1)


def test_task_confidence_wrong_shape_rejected(tmp_path: Path) -> None:
    """A 1-D task_confidence is rejected — it must be (N, num_dimensions)."""
    out = tmp_path / "CM_Table.mat"
    n, d = 5, 4
    with pytest.raises(ValueError, match="task_confidence must have shape"):
        write_cm_table_mat(
            out,
            data_numbers=np.arange(1, n + 1, dtype=np.int32),
            true_values=np.zeros((n, d), dtype=np.float64),
            window_predictions=[np.zeros((n, d), dtype=np.float64)],
            task_confidence=np.ones(n),  # (N,) — wrong, should be (N, D)
        )


# ───────────────────────── parameter_yaml ─────────────────────────


def test_write_yaml_emits_full_schema_even_for_partial_run_config(tmp_path: Path) -> None:
    """A run that only overrides one field still writes every schema field."""
    out = tmp_path / ENCODING_PARAMETERS_FILENAME
    schema = {
        "weight_kl": 1.0,
        "weight_reconstruction": 100.0,
        "weight_classification": 10.0,
        "epoch": "Decision",
        "target": "Dimension",
    }
    run = {"weight_kl": 5.0}
    merged = write_encoding_parameters_yaml(
        out, run_config=run, schema_template=schema
    )

    # Default is translate_keys=True — keys now in MATLAB form.
    assert merged["WeightKL"] == 5.0  # override applied + translated
    assert merged["WeightReconstruction"] == 100.0  # default kept
    assert set(merged.keys()) == {
        "WeightKL", "WeightReconstruction", "WeightClassification",
        "Epoch", "Target",
    }


def test_write_yaml_field_ordering_matches_schema(tmp_path: Path) -> None:
    """Schema insertion order is preserved on disk (helps with human diffs)."""
    out = tmp_path / ENCODING_PARAMETERS_FILENAME
    schema = {"b": 1, "a": 2, "c": 3}
    write_encoding_parameters_yaml(out, run_config={}, schema_template=schema)
    text = out.read_text()
    # Each field appears in the file in schema order (B/A/C → PascalCase).
    assert text.index("B:") < text.index("A:") < text.index("C:")


def test_read_yaml_round_trip(tmp_path: Path) -> None:
    """Writing then reading yields the same mapping (keys MATLAB-translated)."""
    out = tmp_path / ENCODING_PARAMETERS_FILENAME
    schema = {"weight_kl": 1.0, "epoch": "Decision"}
    run = {"weight_kl": 5.0, "epoch": "Synthetic_Easy"}
    written = write_encoding_parameters_yaml(out, run_config=run, schema_template=schema)
    loaded = read_encoding_parameters_yaml(out)
    assert loaded == written
    assert "WeightKL" in loaded


def test_write_yaml_creates_parent_directory(tmp_path: Path) -> None:
    """A missing parent directory is created — caller doesn't need to mkdir."""
    nested = tmp_path / "deeply" / "nested" / ENCODING_PARAMETERS_FILENAME
    write_encoding_parameters_yaml(
        nested, run_config={}, schema_template={"x": 1}
    )
    assert nested.exists()


def test_write_yaml_can_disable_translation(tmp_path: Path) -> None:
    """Setting translate_keys=False emits the Python snake_case names verbatim."""
    out = tmp_path / ENCODING_PARAMETERS_FILENAME
    schema = {"weight_kl": 1.0}
    written = write_encoding_parameters_yaml(
        out, run_config={}, schema_template=schema, translate_keys=False
    )
    assert "weight_kl" in written
    assert "WeightKL" not in written


# ───────────────────────── translate_key ─────────────────────────


@pytest.mark.parametrize(
    "python_key,expected",
    [
        # Common PascalCase via the fallback.
        ("weight_classification", "WeightClassification"),
        ("is_variational", "IsVariational"),
        ("epoch", "Epoch"),
        # Acronyms — must keep KL / STD / L2 / IDX uppercase.
        ("weight_kl", "WeightKL"),
        ("loss_factor_kl", "LossFactorKL"),
        ("std_channel_offset", "STDChannelOffset"),
        ("std_white_noise", "STDWhiteNoise"),
        ("l2_factor", "L2Factor"),
        ("starting_idx", "StartingIDX"),
        ("ending_idx", "EndingIDX"),
        # Embedded-underscore names MATLAB keeps as-is.
        ("loss_type_decoder", "LossType_Decoder"),
        ("loss_type_classifier", "LossType_Classifier"),
        ("freeze_cfg", "Freeze_cfg"),
        ("time_start", "Time_Start"),
        # camelCase exceptions from the reference YAML.
        ("max_worker_mini_batch_size", "maxworkerMiniBatchSize"),
        ("want_stratified_partition", "wantStratifiedPartition"),
        ("is_function", "isfunction"),
    ],
)
def test_translate_key_matches_reference_yaml(
    python_key: str, expected: str
) -> None:
    """Each translation mirrors the field names in the real EncodingParameters.yaml."""
    assert translate_key(python_key) == expected
