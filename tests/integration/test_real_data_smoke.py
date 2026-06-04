"""End-to-end smoke run on the real ``Decision_Data_0000011.mat`` fixture.

Pins Milestone D.7 + D.8: the CLI accepts ``--config-name real_data_base``
plus the ``data_dir`` / ``target_dir`` overrides, builds a
:class:`MatFileTrialDataset`, threads the per-window axes (T, A, C) into
the composite, and writes the standard ``CM_Table.mat`` outputs without
crashing.

Skipped automatically when the sample fixture is missing — the rest of
the unit suite still gates the components individually.

The output-root uses a short ``/tmp/ndd_smoke_<pid>_<id>`` prefix
rather than ``tmp_path`` because the MATLAB-parity long-folder
hierarchy (~1100 chars) can exceed macOS PATH_MAX 1024 when
combined with the deeper ``/private/var/folders/...`` pytest tmp
prefix. ACCRE Linux (PATH_MAX 4096) handles the full chain
natively, so this is dev-machine ergonomics, not a real-pipeline
constraint.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from neural_data_decoding.cli import main as cli_main


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "results" / "Decision"
DATA_FIXTURE = FIXTURE_DIR / "Decision_Data_0000011.mat"
TARGET_FIXTURE = FIXTURE_DIR / "Target_0000011.mat"


pytestmark = pytest.mark.skipif(
    not (DATA_FIXTURE.is_file() and TARGET_FIXTURE.is_file()),
    reason="Sample Decision_Data_0000011.mat fixture not present in results/Decision",
)


@pytest.fixture
def short_tmp_path() -> "Iterator[Path]":
    """``/tmp/ndd_smoke_<pid>_<uuid>`` — short enough for the MATLAB-folder chain.

    macOS pytest tmp paths (``/private/var/folders/...``) push the
    base prefix past 100 chars, which combined with the 1000-char
    MATLAB-parity output tree exceeds PATH_MAX. Using ``/tmp``
    directly trims the prefix to ~25 chars and keeps the full path
    under the limit on macOS dev machines.
    """
    base = Path("/tmp") / f"ndd_smoke_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_real_data_base_runs_end_to_end(short_tmp_path: Path) -> None:
    """``train --config-name real_data_base`` completes with the smoke fixture.

    Pinned outputs: a non-empty result directory containing
    ``CM_Table_Validation.mat``, ``CM_Table.mat``, ``EncodingParameters.yaml``,
    plus the ``optimal_state.pt`` / ``current_state.pt`` snapshots.
    """
    output_root = short_tmp_path / "results"
    argv = [
        "train",
        "--config-name", "real_data_base",
        "--output-root", str(output_root),
        "--override", f"data_dir='{FIXTURE_DIR}'",
        "--override", f"target_dir='{FIXTURE_DIR}'",
        # Shrink the run so the test stays in the default-suite budget.
        "--override", "num_epochs_full=1",
        "--override", "mini_batch_size=1",
        "--override", "hidden_sizes=[8, 4]",
        "--override", "classifier_hidden_size=[4]",
        # Disable confidence + MIL + curriculum to keep the smoke
        # minimal — D.7's purpose is just shape-correctness end-to-end.
        "--override", "weight_confidence=0",
        "--override", "multiple_instance_learning_type='None'",
        "--override", "dynamic_parameter_set='None'",
        # Pad classifier so trial-level confidence head is satisfied.
        "--force",
    ]
    rc = cli_main(argv)
    assert rc == 0

    # The CLI computes the result dir from the cfg's identifying fields;
    # we don't pin the exact path here (it changes when the cfg hash
    # changes), but the root must contain SOME result tree with the
    # standard files in a leaf.
    cm_test = list(output_root.rglob("CM_Table.mat"))
    cm_val = list(output_root.rglob("CM_Table_Validation.mat"))
    enc_params = list(output_root.rglob("EncodingParameters.yaml"))
    optimal_state = list(output_root.rglob("optimal_state.pt"))
    assert cm_test, "no CM_Table.mat written"
    assert cm_val, "no CM_Table_Validation.mat written"
    assert enc_params, "no EncodingParameters.yaml written"
    assert optimal_state, "no optimal_state.pt written"


def test_real_data_session_filter_keeps_pipeline_running(
    short_tmp_path: Path,
) -> None:
    """``--session <name>`` (matching the fixture's session) does not crash.

    The single fixture session is ``Wo_Probe_01_23_02_13_003_01``; filtering
    to it should keep the one trial and the run proceeds normally.
    """
    output_root = short_tmp_path / "results"
    argv = [
        "train",
        "--config-name", "real_data_base",
        "--output-root", str(output_root),
        "--override", f"data_dir='{FIXTURE_DIR}'",
        "--override", f"target_dir='{FIXTURE_DIR}'",
        "--override", "num_epochs_full=1",
        "--override", "mini_batch_size=1",
        "--override", "hidden_sizes=[8, 4]",
        "--override", "classifier_hidden_size=[4]",
        "--override", "weight_confidence=0",
        "--override", "multiple_instance_learning_type='None'",
        "--override", "dynamic_parameter_set='None'",
        "--session", "Wo_Probe_01_23_02_13_003_01",
        "--force",
    ]
    rc = cli_main(argv)
    assert rc == 0
