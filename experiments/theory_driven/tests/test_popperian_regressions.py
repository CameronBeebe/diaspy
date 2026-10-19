"""Exercise real DSPy typed critique/rewrite and survival with offline LMs."""

import logging
from typing import cast

import dspy
from dspy.utils import DummyLM

from experiments.theory_driven.arms.gepa_default import GEPADefaultArm
from experiments.theory_driven.arms.popperian import PopperianArm, PopperianProposer
from experiments.theory_driven.arms.prose_critic import ProseCriticArm
import pytest
from experiments.theory_driven.config.budget_tracker import UnifiedBudget


class AnswerProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.answer(question=question)


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(score=float(gold.answer == pred.answer), feedback="Must answer yes")


def make_proposer(task_answer="no", max_attempts=1, program=None, refutations=None, **kwargs):
    task = DummyLM([{"answer": "no"}, {"answer": task_answer}])
    if refutations is None:
        refutations = [{"claim": "wrong answer", "counterexample_ids": [0],
                        "severity": 1.0, "proposed_fix": "answer yes"}]
    critic = DummyLM([
        {"refutations": refutations},
        {"new_instruction": "Accepted rewrite"},
    ])
    gold = dspy.Example(question="q", answer="yes").with_inputs("question")
    budget = UnifiedBudget(100, 1.0, task, critic, roles={"task": task, "critic": critic})
    proposer = PopperianProposer(
        program or AnswerProgram(), metric, {"q": gold}, critic, task, budget,
        max_attempts=max_attempts, **kwargs,
    )
    return proposer, task, critic, budget


def test_rejected_rewrite_keeps_parent_not_base_instruction():
    proposer, task, critic, budget = make_proposer()
    result = proposer({"answer": "Parent not base"}, {"answer": [{"Inputs": {"question": "q"}}]}, ["answer"])
    assert result == {"answer": "Parent not base"}
    assert len(critic.history) == 2
    assert len(task.history) == 2
    assert budget.used_calls() == 4
    assert proposer.log[-1]["survived"] is False


@pytest.mark.parametrize("refutations", [
    [],
    [{"claim": "minor", "counterexample_ids": [0], "severity": 0.49, "proposed_fix": "fix"}],
    [{"claim": "invalid", "counterexample_ids": [1, 100], "severity": 1.0, "proposed_fix": "fix"}],
], ids=["empty", "all-low-severity", "out-of-range"])
def test_no_valid_survival_questions_keeps_parent(refutations):
    proposer, task, critic, budget = make_proposer(
        task_answer="yes", refutations=refutations, severity_tau=0.5, min_survival_questions=0)
    candidate = {"answer": "Parent not base"}
    result = proposer(candidate, {"answer": [{"Inputs": {"question": "q"}}]}, ["answer"])
    assert result == candidate
    assert proposer.log[-1]["survived"] is False
    assert proposer.log[-1]["checked_ids"] == []
    assert proposer.archive == {}
    assert len(task.history) == 1  # parent is observed even with no surviving citation
    assert len(critic.history) == 2
    assert budget.used_calls() == 3


@pytest.mark.parametrize("counterexample_id", [-1, -2])
@pytest.mark.parametrize("archive_answer", [None, "yes", "no"], ids=["no-archive", "archive-passes", "archive-fails"])
def test_negative_survival_ids_are_ignored(counterexample_id, archive_answer):
    refutations = [{"claim": "invalid", "counterexample_ids": [counterexample_id],
                    "severity": 1.0, "proposed_fix": "fix"}]
    proposer, task, _, _ = make_proposer(
        task_answer=archive_answer or "yes", refutations=refutations, min_survival_questions=0)
    if archive_answer is not None:
        proposer.gold["archived"] = dspy.Example(question="archived", answer="yes").with_inputs("question")
        proposer.archive["archived"] = 1.0
    candidate = {"answer": "Parent not base"}
    result = proposer(candidate, {"answer": [{"Inputs": {"question": "q"}}]}, ["answer"])
    assert result == candidate  # no corroborated counterexample, even if archive kept
    assert proposer.log[-1]["survived"] is False
    assert proposer.log[-1]["rationale"]["reason"] == "no_corroborated_counterexample"
    assert proposer.log[-1]["checked_ids"] == ([] if archive_answer is None else ["archived"])
    assert proposer.archive == ({} if archive_answer is None else {"archived": 1.0})
    assert len(task.history) == (1 if archive_answer is None else 3)


def test_no_rewrite_attempts_keeps_parent():
    proposer, task, critic, _ = make_proposer(max_attempts=0)
    result = proposer({"answer": "Parent"}, {"answer": [{"Inputs": {"question": "q"}}]}, ["answer"])
    assert result == {"answer": "Parent"}
    assert not task.history
    assert len(critic.history) <= 1
    assert not proposer.log


def test_accepted_rewrite_changes_instruction():
    proposer, task, critic, budget = make_proposer(task_answer="yes")
    result = proposer({"answer": "Parent"}, {"answer": [{"Inputs": {"question": "q"}}]}, ["answer"])
    assert result == {"answer": "Accepted rewrite"}
    assert proposer.log[-1]["survived"] is True
    assert proposer.archive == {"q": 1.0}
    assert budget.used_calls() == len(task.history) + len(critic.history) == 4


@pytest.mark.parametrize("policy", ["strict", "majority", "veto"])
def test_policy_reaches_proposer_and_trace_preserves_descriptive_evidence(policy):
    proposer, _, _, _ = make_proposer(task_answer="yes", survival_policy=policy)
    proposer({"answer": "Parent"}, {"answer": [{"Inputs": {"question": "q"}}]}, ["answer"])
    record = proposer.log[-1]
    assert record["policy"] == policy
    assert record["severity_usage"] == "descriptive_only"
    assert record["rationale"]["counterexamples_fixed"] == 1
    assert record["evidence"] == [{"question": "q", "role": "counterexample", "parent_score": 0.0,
                                   "rewrite_score": 1.0, "reference_score": None,
                                   "refutations": record["refutations"]}]


def test_unknown_policy_raises_at_proposer_and_arm_construction():
    with pytest.raises(ValueError, match="Unknown survival policy"):
        make_proposer(survival_policy="typo")
    with pytest.raises(ValueError, match="Unknown survival policy"):
        PopperianArm("B", "offline", UnifiedBudget(100, 1.0), survival_policy="typo")


class TwoPredictorProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Predict("question, context -> evidence")
        self.answer = dspy.Predict("question, evidence -> answer")

    def forward(self, question, context):
        evidence = self.retrieve(question=question, context=context).evidence
        return self.answer(question=question, evidence=evidence)


def test_survival_runs_full_parent_candidate_and_all_example_inputs():
    program = TwoPredictorProgram()
    task = DummyLM([{"evidence": "support"}, {"answer": "no"},
                    {"evidence": "support"}, {"answer": "yes"}])
    critic = DummyLM([
        {"refutations": [{"claim": "wrong answer", "counterexample_ids": [0],
                          "severity": 1.0, "proposed_fix": "answer yes"}]},
        {"new_instruction": "Proposed answer instruction"},
    ])
    gold = dspy.Example(question="q", context="required paragraph", answer="yes").with_inputs("question", "context")
    proposer = PopperianProposer(program, metric, {"q": gold}, critic, task, UnifiedBudget(100, 1, task, critic))
    candidate = {"retrieve": "Parent retrieval instruction", "answer": "Parent answer instruction"}
    result = proposer(candidate, {"answer": [{"Inputs": {"question": "q", "evidence": "old"}}]}, ["answer"])
    assert result == {"answer": "Proposed answer instruction"}
    assert len(task.history) == 4
    assert "Parent answer instruction" in task.history[1]["messages"][0]["content"]
    assert "Parent retrieval instruction" in task.history[2]["messages"][0]["content"]
    assert "Parent retrieval instruction" in task.history[0]["messages"][0]["content"]
    assert "required paragraph" in task.history[0]["messages"][-1]["content"]
    assert "Proposed answer instruction" in task.history[3]["messages"][0]["content"]
    assert "Parent retrieval instruction" not in cast(type[dspy.Signature], program.retrieve.signature).instructions
    assert candidate["answer"] == "Parent answer instruction"


@pytest.mark.parametrize("arm_type", [GEPADefaultArm, PopperianArm, ProseCriticArm])
def test_real_gepa_uses_history_stopper_and_common_reflection_keyword(arm_type, caplog):
    caplog.set_level(logging.INFO)
    task = DummyLM([{"answer": "no"}] * 50)
    if arm_type is GEPADefaultArm:
        reflection = DummyLM([{"new_instruction": "Better instruction"}] * 20, adapter=dspy.JSONAdapter())
    elif arm_type is PopperianArm:
        reflection = DummyLM([
            {"refutations": [{"claim": "wrong", "counterexample_ids": [0], "severity": 1.0, "proposed_fix": "fix"}]},
            {"new_instruction": "Better instruction"},
            {"new_instruction": "Another instruction"},
        ])
    else:
        reflection = DummyLM([{"critique": "wrong"}, {"new_instruction": "Better instruction"}])
    budget = UnifiedBudget(3, 1.0, task, reflection, roles={"task": task, "reflection": reflection})
    arm = arm_type("test", "offline", budget, max_metric_calls=50, reflection_minibatch_size=1, seed=17)
    gold = dspy.Example(question="q", answer="yes").with_inputs("question")
    with dspy.context(lm=task):
        optimized = arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task, reflection_lm=reflection)
    assert isinstance(optimized, dspy.Module)
    assert budget(None)
    assert 3 <= budget.used_calls() < 15  # stopped mid-run, not at 50 metric calls
    assert len(reflection.history) > 0
    assert arm.seed == 17
    assert arm.max_metric_calls == 50
    expected_reflection_calls = {GEPADefaultArm: 1, PopperianArm: 3, ProseCriticArm: 2}
    assert len(reflection.history) == expected_reflection_calls[arm_type]
    assert "reflection failed" not in caplog.text
    assert "Traceback" not in caplog.text
