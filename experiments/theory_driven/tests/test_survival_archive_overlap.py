"""Archive references survive overlap during real offline survival rollouts."""

import pytest

from experiments.theory_driven.arms.survival import get_policy
from experiments.theory_driven.tests.test_survival import make_evaluator


@pytest.mark.parametrize("overlap_role", ["counterexample", "defense", "held_out_regression"])
def test_prepare_preserves_archive_veto_for_overlapping_question(overlap_role):
    import dspy

    evaluator, lm = make_evaluator(
        [0.3, 0.3, 0.6, 0.6], min_survival_questions=2,
        held_out_regression_samples=1 if overlap_role == "held_out_regression" else 0,
        trainset=[dspy.Example(question="r").with_inputs("question")],
    )
    archived_question = "c" if overlap_role == "counterexample" else "r"
    evaluator.archive[archived_question] = 0.9
    parent = {"answer": "parent"}
    evidence = evaluator.prepare(
        parent, [{"counterexample_ids": [0], "severity": 0}],
        [{"Inputs": {"question": q}} for q in ("c", "r")],
        protected_ids=[1] if overlap_role == "defense" else None,
    )
    evaluated = evaluator.evaluate(parent, "answer", "rewrite", evidence)

    admitted, rationale = get_policy("veto")(evaluated)

    assert admitted is False
    assert rationale["reason"] == "archive_veto"
    assert rationale["vetoed_questions"] == [archived_question]
    assert rationale["counterexamples_fixed"] == 1
    # The historical constraint survives without double-counting regression votes.
    assert rationale["regressions_total"] == (2 if overlap_role == "counterexample" else 1)
    assert rationale["regressions_kept"] == (1 if overlap_role == "counterexample" else 0)
    overlap = [row for row in evaluated if row.question == archived_question]
    assert [row.role for row in overlap] == [overlap_role, "archive"]
    assert overlap[-1].reference_score == 0.9
    assert overlap[-1].parent_score == 0.3
    assert evaluator.last_metadata["non_vacuous_questions"] == 2
    assert evaluator.archive == {archived_question: 0.9}
    assert len(lm.history) == 4  # One observation per unique question per candidate.
