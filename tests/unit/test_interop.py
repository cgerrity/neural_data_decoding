"""Tests for :mod:`neural_data_decoding.interop`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from neural_data_decoding.interop import (
    ENCODING_PARAMETERS_FILENAME,
    VALIDATION_CM_TABLE_FILENAME,
    build_result_dir,
    read_encoding_parameters_yaml,
    write_cm_table_mat,
    write_encoding_parameters_yaml,
)
from neural_data_decoding.interop.parameter_yaml import translate_key


# ───────────────────────── folder_hierarchy ─────────────────────────


def test_build_result_dir_layout(tmp_path: Path) -> None:
    """Each hyperparameter becomes its own path component, in the documented order."""
    path = build_result_dir(
        base_dir=tmp_path,
        epoch="Synthetic_Easy",
        target="Dimension",
        model_name="Logistic Regression",
        fold=1,
        identifying_config={"lr": 0.01},
    )
    parts = path.parts
    # tmp_path / Synthetic_Easy / Dimension / Logistic Regression / cfg-<hash> / fold-1
    assert parts[-5] == "Synthetic_Easy"
    assert parts[-4] == "Dimension"
    assert parts[-3] == "Logistic Regression"
    assert parts[-2].startswith("cfg-")
    assert parts[-1] == "fold-1"


def test_build_result_dir_is_deterministic(tmp_path: Path) -> None:
    """Same config + same fold → byte-identical path."""

    def _call() -> Path:
        return build_result_dir(
            base_dir=tmp_path,
            epoch="Decision",
            target="Dimension",
            model_name="GRU",
            fold=2,
            identifying_config={"batch_size": 32, "lr": 0.001},
        )

    assert _call() == _call()


def test_build_result_dir_hash_changes_on_config_change(tmp_path: Path) -> None:
    """A different identifying_config produces a different hash bucket."""
    a = build_result_dir(
        base_dir=tmp_path,
        epoch="Decision",
        target="Dimension",
        model_name="GRU",
        fold=1,
        identifying_config={"lr": 0.001},
    )
    b = build_result_dir(
        base_dir=tmp_path,
        epoch="Decision",
        target="Dimension",
        model_name="GRU",
        fold=1,
        identifying_config={"lr": 0.01},
    )
    assert a != b


def test_build_result_dir_key_order_does_not_matter(tmp_path: Path) -> None:
    """Insertion order of the config dict must not affect the hash."""
    a = build_result_dir(
        base_dir=tmp_path,
        epoch="Decision",
        target="Dimension",
        model_name="GRU",
        fold=1,
        identifying_config={"a": 1, "b": 2},
    )
    b = build_result_dir(
        base_dir=tmp_path,
        epoch="Decision",
        target="Dimension",
        model_name="GRU",
        fold=1,
        identifying_config={"b": 2, "a": 1},
    )
    assert a == b


def test_build_result_dir_rejects_fold_zero(tmp_path: Path) -> None:
    """MATLAB is 1-indexed; 0 is a programming error."""
    with pytest.raises(ValueError, match="fold must be >= 1"):
        build_result_dir(
            base_dir=tmp_path,
            epoch="Decision",
            target="Dimension",
            model_name="GRU",
            fold=0,
        )


def test_build_result_dir_rejects_empty_components(tmp_path: Path) -> None:
    """An empty Epoch / Target / ModelName is silently catastrophic; reject."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        build_result_dir(
            base_dir=tmp_path, epoch="", target="Dimension", model_name="GRU", fold=1
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
