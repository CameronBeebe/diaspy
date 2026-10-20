"""Offline score-based admission and real DSPy survival rollouts."""

import importlib
import importlib.util

import pytest


def survival_module():
    name = "experiments.theory_driven.arms.survival"
    assert importlib.util.find_spec(name) is not None, "typed survival engine is missing"
    return importlib.import_module(name)


@pytest.mark.parametrize("parent,rewrite,expected", [(0.3, 0.6, True), (0.3, 0.3, False), (0.3, 0.2, False)])
def test_strict_single_counterexample_uses_relative_improvement(parent, rewrite, expected):
    s = survival_module()
    evidence = [s.Evidence("q", "counterexample", parent, rewrite, [])]
    admitted, rationale = s.get_policy("strict")(evidence)
    assert admitted is expected
    assert rationale["counterexamples_fixed"] == int(expected)


@pytest.mark.parametrize("policy,fixed,kept,archive,expected", [
    ("strict", [True, False], [True], [], False),
    ("strict", [True], [False], [], False),
    ("strict", [True], [True], [], True),
    ("majority", [True], [], [], True),
    ("majority", [True, True, False, False], [True], [], False),
    ("majority", [True, True, False], [True, True, False], [], True),
    ("majority", [True], [True, False], [], False),
    ("majority", [False], [True, True], [], False),
    ("majority", [True], [True, True], [False], True),
    ("veto", [True], [True, True], [False], False),
    ("veto", [True], [], [True], True),
    ("half_no_regress", [True, False], [True], [], True),
    ("half_no_regress", [True, False], [False], [], False),
    ("half_no_regress", [False], [True], [], False),
    ("half_no_regress", [True], [], [False], False),
    ("half_no_regress", [], [True], [], False),
    ("strict", [], [True], [], False),
    ("majority", [], [True], [], False),
    ("veto", [], [True], [], False),
])
def test_policy_truth_table(policy, fixed, kept, archive, expected):
    s = survival_module()
    rows = [s.Evidence(f"c{i}", "counterexample", 0.3, 0.6 if passed else 0.3, [])
            for i, passed in enumerate(fixed)]
    rows += [s.Evidence(f"r{i}", "regression", 0.8, 0.8 if passed else 0.7, [])
             for i, passed in enumerate(kept)]
    rows += [s.Evidence(f"a{i}", "archive", 0.2, 0.9 if passed else 0.8, [], 0.9)
             for i, passed in enumerate(archive)]
    admitted, rationale = s.get_policy(policy)(rows)
    assert admitted is expected
    if not fixed:
        assert rationale["reason"] == "no_corroborated_counterexample"
    if policy == "veto" and False in archive:
        assert rationale["vetoed_questions"] == ["a0"]


@pytest.mark.parametrize("policy", ["strict", "majority", "veto"])
@pytest.mark.parametrize("reverse_overlap", [False, True])
@pytest.mark.parametrize("parent,reference,rewrite,other_rewrite,archive_veto", [
    (1.0, 1.0, 1.0, 0.0, False),  # One kept question, not two votes.
    (1.0, 0.6, 0.8, 1.0, False),  # Defense is stricter than the archive.
    (0.6, 1.0, 0.8, 1.0, True),   # Archive is stricter than the defense.
])
def test_overlapping_regressions_get_one_vote_requiring_all_constraints(
        policy, reverse_overlap, parent, reference, rewrite, other_rewrite, archive_veto):
    s = survival_module()
    overlap = [s.Evidence("r", "defense", parent, rewrite, []),
               s.Evidence("r", "archive", parent, rewrite, [], reference)]
    if reverse_overlap:
        overlap.reverse()
    rows = [s.Evidence("c", "counterexample", 0.3, 0.6, []), *overlap,
            s.Evidence("z", "regression", 1.0, other_rewrite, [])]

    admitted, rationale = s.get_policy(policy)(rows)

    assert rationale["regressions_total"] == 2
    assert rationale["regressions_kept"] == 1
    assert admitted is False  # A distinct-question tie is never a majority.
    expected_veto = policy == "veto" and archive_veto
    assert rationale["vetoed_questions"] == (["r"] if expected_veto else [])
    assert rationale["reason"] == (
        "archive_veto" if expected_veto else "insufficient_fixes_or_regressions")


def test_epsilon_reference_dropped_and_custom_policy():
    s = survival_module()
    rows = [s.Evidence("c", "counterexample", 0.3, 0.4, []),
            s.Evidence("r", "defense", 0.9, 0.8, []),
            s.Evidence("d", "uncorroborated-dropped", 1, None, [])]
    assert s.get_policy("strict", eps=0.1)(rows)[0] is False
    rows[0].rewrite_score = 0.5
    assert s.get_policy("strict", eps=0.1)(rows)[0] is True
    policy = s.get_policy("strict")
    assert s.get_policy(policy) is policy
    with pytest.raises(ValueError, match="Unknown survival policy"):
        s.get_policy("typo")


def make_evaluator(scores, **kwargs):
    import dspy
    from dspy.utils import DummyLM
    from experiments.theory_driven.tests.test_popperian_regressions import AnswerProgram
    s = survival_module()
    assert hasattr(s, "SurvivalEvaluator"), "shared evaluator is missing"
    lm = DummyLM([{"answer": str(score)} for score in scores])
    gold = {q: dspy.Example(question=q).with_inputs("question") for q in ("c", "r", "z", "a")}
    evaluator = s.SurvivalEvaluator(
        AnswerProgram(), lambda gold, pred, *args: dspy.Prediction(score=float(pred.answer)),
        gold, lm, **kwargs)
    return evaluator, lm


def test_prepare_corroborates_pads_observes_full_parent_and_evaluates_every_row():
    evaluator, lm = make_evaluator([0.3, 1, 0.8, 0.3, 0.9, 0.8])
    from experiments.theory_driven.arms.popperian import Refutation
    refs = [Refutation(claim="fails", counterexample_ids=[0, 1, -1, 30], severity=0, proposed_fix="fix")]
    batch = [{"Inputs": {"question": q}, "Score": 0} for q in ("c", "r", "z")]
    parent = {"answer": "observed parent"}
    evidence = evaluator.prepare(parent, refs, batch)
    assert [(e.question, e.role, e.parent_score) for e in evidence] == [
        ("c", "counterexample", 0.3), ("r", "uncorroborated-dropped", 1),
        ("r", "regression", 1), ("z", "regression", 0.8)]
    assert len(lm.history) == 3
    assert all("observed parent" in h["messages"][0]["content"] for h in lm.history)
    evaluated = evaluator.evaluate(parent, "answer", "rewrite", evidence)
    assert [e.rewrite_score for e in evaluated] == [0.3, None, 0.9, 0.8]
    assert all(e.rewrite_score is None for e in evidence)
    assert len(lm.history) == 6  # first failed counterexample does not short-circuit
    assert evaluated[0].refutations[0]["severity"] == 0


def test_defense_uses_observed_median_and_archive_keeps_original_reference():
    evaluator, lm = make_evaluator([0.3, 1, 0.8, 0.4, 0.6, 1, 0.8, 0.7])
    evaluator.archive["a"] = 0.9
    refs = [{"counterexample_ids": [0], "severity": 0}]
    parent = {"answer": "parent"}
    batch = [{"Inputs": {"question": q}} for q in ("c", "r", "z")]
    evidence = evaluator.prepare(parent, refs, batch, protected_ids=[0, 1, 2, -1, 30])
    # With deduplication: "c" appears only once as counterexample (higher precedence)
    assert [(e.question, e.role) for e in evidence] == [
        ("c", "counterexample"), ("r", "defense"), ("z", "defense"), ("a", "archive")]
    archived = evidence[-1]
    assert (archived.parent_score, archived.reference_score) == (0.4, 0.9)
    evaluated = evaluator.evaluate(parent, "answer", "rewrite", evidence)
    assert len(lm.history) == 8  # 4 prepare + 4 evaluate
    assert survival_module().get_policy("veto")(evaluated)[0] is False
    assert evaluator.archive == {"a": 0.9}


@pytest.mark.parametrize("policy", ["strict", "majority", "veto"])
def test_severity_independent_admission_and_fixed_only_archive(policy):
    s = survival_module()
    outcomes = []
    for severity in (0.0, 1.0):
        evaluator, _ = make_evaluator([0.3, 0.3, 0.3, 0.6, 0.6, 0.3], archive_cap=1)
        refs = [{"counterexample_ids": [0, 1, 2], "severity": severity}]
        batch = [{"Inputs": {"question": q}} for q in ("c", "r", "z")]
        rows = evaluator.prepare({"answer": "parent"}, refs, batch)
        rows = evaluator.evaluate({"answer": "parent"}, "answer", "rewrite", rows)
        admitted, _ = s.get_policy(policy)(rows)
        assert hasattr(evaluator, "admit_archive"), "admitted fixes must be archived"
        if admitted:
            evaluator.admit_archive(rows)
        outcomes.append((admitted, evaluator.archive))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0] == ((False, {}) if policy == "strict" else (True, {"r": 0.6}))


@pytest.mark.parametrize("policy", ["strict", "majority", "veto", "half_no_regress"])
def test_zero_reference_checks_are_flagged_and_excluded(policy):
    from dataclasses import asdict
    s = survival_module()
    rows = [s.Evidence("c", "counterexample", 0, 1, []),
            s.Evidence("r", "regression", 1, 0, []),
            s.Evidence("z", "regression", 0, 0, []),
            s.Evidence("a", "archive", 1, 0, [], 0)]
    admitted, rationale = s.get_policy(policy)(rows)
    assert not admitted
    assert rationale["regressions_kept"] == 0
    assert rationale["regressions_total"] == 1
    assert [asdict(e)["vacuous"] for e in rows] == [False, False, True, True]


def test_zero_defense_dropped_padding_prefers_positive_and_thin_is_logged(caplog):
    evaluator, _ = make_evaluator([0, 0, 0.6, 0], min_survival_questions=2)
    rows = evaluator.prepare({"answer": "parent"}, [{"counterexample_ids": [0], "severity": 0}],
                             [{"Inputs": {"question": q}} for q in ("c", "r", "z")],
                             protected_ids=[1])
    assert [(e.question, e.role) for e in rows] == [
        ("c", "counterexample"), ("r", "uncorroborated_defense"), ("z", "regression")]
    evaluator.min_survival_questions = 4
    evaluator.prepare({"answer": "parent"}, [], [{"Inputs": {"question": "c"}}])
    assert "thin_survival_set" in caplog.text


def test_explicit_legacy_severity_filter_warns_and_zero_archive_cap(caplog):
    evaluator, _ = make_evaluator([0.3], severity_tau=0.5, min_survival_questions=0, archive_cap=0)
    assert "severity_tau" in caplog.text
    rows = evaluator.prepare({"answer": "parent"}, [{"counterexample_ids": [0], "severity": 0.1}],
                             [{"Inputs": {"question": "c"}}])
    assert not rows
    evaluator.admit_archive([survival_module().Evidence("c", "counterexample", 0.3, 0.6, [])])
    assert evaluator.archive == {}
