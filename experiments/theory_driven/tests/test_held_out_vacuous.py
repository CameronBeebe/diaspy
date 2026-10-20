"""Held-out rows follow the existing vacuous rule: a no-regression check is vacuous iff reference == 0."""
import dspy
import pytest
from dspy.utils import DummyLM

from experiments.theory_driven.arms.survival import Evidence, SurvivalEvaluator, get_policy

E = Evidence


@pytest.mark.parametrize("role", ["regression", "archive", "defense", "held_out_regression"])
@pytest.mark.parametrize("rewrite", [0.0, 0.5, None])
def test_reference_zero_is_vacuous_whatever_the_rewrite_score(role, rewrite):
    assert E("q", role, 0.0, rewrite, []).vacuous is True


@pytest.mark.parametrize("role", ["regression", "defense", "held_out_regression"])
@pytest.mark.parametrize("rewrite", [0.0, 1.0, None])
def test_positive_reference_is_never_vacuous(role, rewrite):
    assert E("q", role, 0.5, rewrite, []).vacuous is False


def test_archive_reference_score_override_decides_vacuity():
    # Archive rows compare against their historical reference, not the current parent.
    assert E("a", "archive", 0.0, 0.0, [], 0.9).vacuous is False
    assert E("a", "archive", 0.5, 0.5, [], 0.0).vacuous is True


def test_counterexample_is_never_vacuous():
    assert E("q", "counterexample", 0.0, 0.0, []).vacuous is False


BASE = [E("c", "counterexample", 0.2, 0.6, [{}]), E("k", "regression", 1.0, 0.5, [])]


@pytest.mark.parametrize("policy", ["strict", "majority", "veto", "half_no_regress"])
@pytest.mark.parametrize("role", ["regression", "held_out_regression"])
@pytest.mark.parametrize("rewrite", [0.0, 0.3])
def test_reference_zero_rows_cannot_flip_a_reject_to_admit(policy, role, rewrite):
    rule = get_policy(policy, 0.0)
    assert rule(BASE)[0] is False  # a real 1.0 -> 0.5 regression is rejected
    padded = [*BASE, E("z1", role, 0.0, rewrite, []), E("z2", role, 0.0, rewrite, [])]
    assert rule(padded)[0] is False


def test_vacuous_held_out_rows_do_not_hide_thin_survival_set():
    """Two corroborated counterexamples plus two zero-parent held-out rows: still thin at min=3."""
    gold = {q: dspy.Example(question=q, answer="right").with_inputs("question")
            for q in ("c1", "c2", "h1", "h2")}

    def metric(example, prediction, *args):
        return dspy.Prediction(score=float(example.answer == prediction.answer))

    task = DummyLM([{"answer": "wrong"}] * 20)  # every parent score is 0
    evaluator = SurvivalEvaluator(dspy.Predict("question -> answer"), metric, gold, task,
                                  min_survival_questions=3, held_out_regression_samples=2,
                                  trainset=[gold["h1"], gold["h2"]], seed=0)
    refs = [{"claim": "fails", "counterexample_ids": [0, 1], "severity": 0, "proposed_fix": "fix"}]
    batch = [{"Inputs": {"question": q}} for q in ("c1", "c2")]
    evidence = evaluator.prepare({"self": "parent"}, refs, batch)
    roles = {e.question: (e.role, e.vacuous) for e in evidence}
    assert roles == {"c1": ("counterexample", False), "c2": ("counterexample", False),
                     "h1": ("held_out_regression", True), "h2": ("held_out_regression", True)}
    assert evaluator.last_metadata["non_vacuous_questions"] == 2
    assert evaluator.last_metadata["thin_survival_set"] is True
    assert evaluator.last_metadata["held_out_regression_count"] == 2
