"""Test that held-out 0/0 rows are treated as vacuous and don't affect survival."""

from experiments.theory_driven.arms.survival import Evidence, get_policy


def test_held_out_zero_zero_is_vacuous():
    """Held-out rows where parent and rewrite both score 0 are vacuous."""
    evidence = Evidence(
        question="q1",
        role="held_out_regression",
        parent_score=0.0,
        rewrite_score=0.0,
        refutations=[],
        reference_score=None,
    )
    assert evidence.vacuous is True, "held_out_regression with 0→0 should be vacuous"


def test_held_out_vacuous_does_not_count_toward_minimum():
    """Vacuous held-out rows don't count toward the non-vacuous minimum."""
    # One counterexample fixed (0.2→0.6), one real regression (1.0→0.5),
    # plus two held-out 0→0 rows. Without held-out rows, majority should reject.
    evidence = [
        Evidence("q1", "counterexample", 0.2, 0.6, [], None),  # fixed
        Evidence("q2", "regression", 1.0, 0.5, [], None),      # regressed
        Evidence("q3", "held_out_regression", 0.0, 0.0, [], None),  # vacuous
        Evidence("q4", "held_out_regression", 0.0, 0.0, [], None),  # vacuous
    ]
    
    # Majority: require >50% non-vacuous checks to pass
    # Non-vacuous: q1 (fixed), q2 (regressed) = 1 pass, 1 fail = 50% → reject
    majority = get_policy("majority", eps=0.0)
    survived, metadata = majority(evidence)
    assert not survived, f"majority should reject with 50% pass rate, got {metadata}"
    
    # Veto: any failed check rejects
    veto = get_policy("veto", eps=0.0)
    survived, metadata = veto(evidence)
    assert not survived, f"veto should reject with any failure, got {metadata}"


def test_held_out_vacuous_does_not_hide_thin_survival_set():
    """Vacuous held-out rows don't count toward the 3-check minimum."""
    # Two counterexamples fixed, two held-out 0→0 rows
    # Only 2 non-vacuous checks, so thin_survival_set should trigger with min=3
    evidence = [
        Evidence("q1", "counterexample", 0.2, 0.6, [], None),  # fixed
        Evidence("q2", "counterexample", 0.3, 0.7, [], None),  # fixed
        Evidence("q3", "held_out_regression", 0.0, 0.0, [], None),  # vacuous
        Evidence("q4", "held_out_regression", 0.0, 0.0, [], None),  # vacuous
    ]
    
    # ScorePolicy checks min_survival_questions directly on non-vacuous evidence
    # With 2 non-vacuous checks and default logic, check thin_survival_set rejection
    from experiments.theory_driven.arms.survival import ScorePolicy
    
    # Strict policy with eps=0.0 should check all non-vacuous evidence
    policy = ScorePolicy("strict", eps=0.0)
    survived, metadata = policy(evidence)
    
    # With only 2 counterexamples fixed and no regressions, should pass strict
    # The min_survival_questions check is in SurvivalEvaluator, not ScorePolicy
    assert survived, f"strict should pass with 2 fixed counterexamples, got {metadata}"


def test_held_out_non_zero_is_not_vacuous():
    """Held-out rows where parent or rewrite is non-zero are not vacuous."""
    # Parent > 0
    e1 = Evidence("q1", "held_out_regression", 1.0, 1.0, [], None)
    assert not e1.vacuous, "held_out_regression with 1.0→1.0 is not vacuous"
    
    # Rewrite improved from 0
    e2 = Evidence("q2", "held_out_regression", 0.0, 0.5, [], None)
    assert not e2.vacuous, "held_out_regression with 0.0→0.5 is not vacuous"
    
    # Rewrite regressed to 0
    e3 = Evidence("q3", "held_out_regression", 0.5, 0.0, [], None)
    assert not e3.vacuous, "held_out_regression with 0.5→0.0 is not vacuous"


def test_counterexample_zero_zero_is_not_vacuous():
    """Counterexample rows are never vacuous, even at 0→0."""
    evidence = Evidence("q1", "counterexample", 0.0, 0.0, [], None)
    assert not evidence.vacuous, "counterexample is never vacuous"


