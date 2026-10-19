"""Offline regression tests for result analysis."""

import pytest

from experiments.theory_driven.analysis.bootstrap import (
    compute_confidence_interval,
    paired_bootstrap,
)
from experiments.theory_driven.analysis.calibration import (
    compute_brier_score,
    compute_brier_skill_score,
    compute_ece,
)
from experiments.theory_driven.analysis.refutation_criteria import check_refutation_criteria


def test_bootstrap_preserves_list_input_and_constant_difference():
    scores_a = [0.25, 0.5, 0.75]
    scores_b = [0.0, 0.25, 0.5]

    result = paired_bootstrap(scores_a, scores_b, n_samples=20)

    assert result["mean_difference"] == pytest.approx(0.25)
    assert result["ci_lower"] == pytest.approx(0.25)
    assert result["ci_upper"] == pytest.approx(0.25)
    assert scores_a == [0.25, 0.5, 0.75]
    assert scores_b == [0.0, 0.25, 0.5]
    assert compute_confidence_interval([0.5, 0.5], n_samples=20) == (0.5, 0.5)


def test_calibration_preserves_metrics_with_list_inputs():
    predictions = [0.25, 0.75]
    outcomes = [0, 1]

    assert compute_brier_score(predictions, outcomes) == pytest.approx(0.0625)
    assert compute_brier_skill_score(predictions, outcomes) == pytest.approx(0.75)
    assert compute_brier_skill_score(predictions, outcomes, 0.5) == pytest.approx(0.75)
    assert compute_ece(predictions, outcomes) == pytest.approx(0.25)
    assert predictions == [0.25, 0.75]
    assert outcomes == [0, 1]


def test_refutation_analysis_skips_none_scores():
    valid = {
        arm: [{"benchmark": "hover", "budget": 50, "test_mean": score}]
        for arm, score in [("A", 0.0), ("B", 0.5), ("K-Jev", 0.75)]
    }
    with_failed_runs = {
        arm: runs + [{"benchmark": "hover", "budget": 50, "test_mean": None}]
        for arm, runs in valid.items()
    }

    expected = check_refutation_criteria(valid, n_bootstrap=20)
    actual = check_refutation_criteria(with_failed_runs, n_bootstrap=20)

    assert actual == expected
    assert actual["R1_B"]["delta"] == pytest.approx(0.5)


def test_refutation_analysis_skips_all_failed_runs():
    results = {
        arm: [{"benchmark": "hover", "budget": 50, "test_mean": None}]
        for arm in ["A", "B", "K-Jev"]
    }

    assert check_refutation_criteria(results, n_bootstrap=20) == {}
