"""End-to-end convergence + reproducibility parity for Milestone A.

Two layers, matching the migration spec's convergence-parity goal
(``docs/PLAN.md``: "within 2σ across 5 seeds"):

1. **Runnable now (no MATLAB).** Bit-exact reproducibility isn't a cross-runtime
   parity goal (ADR 001), but *intra-Python* determinism is a prerequisite for
   any seed-ensemble study: the same ``(config, fold, seed)`` must reproduce the
   same run, and different seeds must produce a genuine distribution. These tests
   drive the real ``train`` CLI on synthetic data (reduced epochs for speed).

2. **Gated on MATLAB (skips otherwise).** The G6 distributional check compares a
   Python seed ensemble against a MATLAB reference ensemble using the pure
   comparators in
   :mod:`neural_data_decoding.training.convergence_metrics`. No reference-accuracy
   fixture has been recorded yet, so the scaffold skips with a clear reason
   rather than fabricating a target (never invent parity numbers — verify them).
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from neural_data_decoding.cli import main
from neural_data_decoding.training.convergence_metrics import (
    ks_2sample,
    means_within_n_sigma,
    summarize_runs,
    within_n_sigma,
)

_FINAL_ACC_RE = re.compile(r"Final validation accuracy:\s*([0-9.]+)")


def _run_final_val_accuracy(
    *, seed: int, epochs: int, output_root: Path
) -> float:
    """Run the ``train`` CLI once and return its final validation accuracy.

    Parameters
    ----------
    seed
        Global RNG seed passed through ``--seed``.
    epochs
        Number of full (supervised) epochs; kept small so the ensemble runs in
        the default suite's time budget.
    output_root
        Throwaway results directory for this run.

    Returns
    -------
    float
        The final-epoch validation accuracy parsed from the run banner.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = main(
            [
                "train",
                "--config-name", "A_logistic_synthetic",
                "--fold", "1",
                "--seed", str(seed),
                "--override", f"num_epochs_full={epochs}",
                "--output-root", str(output_root),
            ]
        )
    assert rc == 0, f"train exited {rc}"
    match = _FINAL_ACC_RE.search(buffer.getvalue())
    assert match is not None, "did not find a final validation accuracy in output"
    return float(match.group(1))


def test_same_seed_reproduces_exactly(tmp_path: Path) -> None:
    """Same ``(config, fold, seed)`` → identical final accuracy on repeat runs.

    This is the property the CLI's ``set_global_seed`` wiring exists to
    guarantee; without it, seed-ensemble parity would be measuring RNG noise.
    """
    a = _run_final_val_accuracy(seed=0, epochs=3, output_root=tmp_path / "a")
    b = _run_final_val_accuracy(seed=0, epochs=3, output_root=tmp_path / "b")
    assert a == b


def test_seed_ensemble_is_a_well_formed_distribution(tmp_path: Path) -> None:
    """Five seeds yield a valid accuracy distribution the comparators can consume.

    Uses the spec's five-seed convention. Asserts the ensemble is well-formed
    (right count, valid range, genuine spread across seeds) and that the pure
    comparators behave sensibly on it — the same call the MATLAB-gated check
    makes, but against the ensemble itself.
    """
    values = [
        _run_final_val_accuracy(seed=s, epochs=3, output_root=tmp_path / f"s{s}")
        for s in range(5)
    ]
    summary = summarize_runs(values)

    assert summary.n == 5
    assert all(0.0 <= v <= 1.0 for v in values)
    # Different seeds must actually move the result — otherwise the "ensemble"
    # is degenerate and no distributional check means anything.
    assert len(set(values)) >= 2
    assert summary.std > 0.0

    # The ensemble mean is (trivially but by construction) inside its own 2σ band,
    # and an ensemble is consistent with itself under both comparators.
    assert within_n_sigma(summary.mean, values, n_sigma=2.0)
    assert means_within_n_sigma(values, values, n_sigma=2.0)
    assert ks_2sample(values, values).pvalue == pytest.approx(1.0)


@pytest.mark.needs_matlab
def test_matlab_reference_convergence_scaffold(tmp_path: Path) -> None:
    """G6: Python seed ensemble is distributionally consistent with MATLAB.

    Skips until a MATLAB reference ensemble of final accuracies is recorded (the
    numbers must be produced empirically from the MATLAB pipeline, not invented).
    Once ``reference`` below is populated from a MATLAB run, this asserts the two
    ensembles agree in mean (within 2σ) and in shape (KS p-value not tiny).
    """
    # Populate from a MATLAB run of the equivalent A_logistic config, one final
    # validation accuracy per seed. Left empty deliberately — see docstring.
    reference: list[float] = []
    if not reference:
        pytest.skip("no MATLAB reference-accuracy ensemble recorded yet")

    python_values = [
        _run_final_val_accuracy(seed=s, epochs=20, output_root=tmp_path / f"s{s}")
        for s in range(5)
    ]
    assert means_within_n_sigma(python_values, reference, n_sigma=2.0)
    assert ks_2sample(python_values, reference).pvalue > 0.01
