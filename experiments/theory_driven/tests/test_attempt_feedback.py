"""Offline real DSPy prompts, observed feedback, and avoided task calls."""
import json

import dspy
import pytest
from dspy.utils import DummyLM

from experiments.theory_driven.arms.popperian import PopperianProposer
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.tests.test_millian import AnswerProgram, metric


def make_b(rewrites, task_scores=(0.2, 0.2, 0.8), batch=None, refs=None):
    if refs is None:
        refs = [{"claim": "fails", "counterexample_ids": [0], "severity": 0, "proposed_fix": "fix"}]
    reflection = DummyLM([{"refutations": refs}, *[{"new_instruction": text} for text in rewrites]])
    task = DummyLM([{"answer": str(score)} for score in task_scores])
    gold = dspy.Example(question="q", answer="1").with_inputs("question")
    budget = UnifiedBudget(100, 1, task, reflection)
    proposer = PopperianProposer(AnswerProgram(), metric, {"q": gold}, reflection, task, budget,
                                 max_attempts=len(rewrites), min_survival_questions=1)
    return proposer, task, reflection, budget, {"answer": batch or [{"Inputs": {"question": "q"}}]}


def failed_input(history):
    content = history["messages"][-1]["content"]
    return json.JSONDecoder().raw_decode(content.split("[[ ## failed_attempts ## ]]\n", 1)[1])[0]


def test_b_second_attempt_receives_typed_observed_scores_in_dspy_prompt():
    proposer, task, reflection, _, batch = make_b(["first", "second"])
    assert proposer({"answer": "parent"}, batch, ["answer"]) == {"answer": "second"}
    assert failed_input(reflection.history[1]) == []
    assert failed_input(reflection.history[2]) == [{
        "instruction": "first", "per_question": [{"question_id": "q", "role": "counterexample",
        "parent_score": 0.2, "rewrite_score": 0.2, "fixed_or_kept": False}],
        "rejection_reason": "insufficient_fixes_or_regressions"}]
    assert len(task.history) == 3


@pytest.mark.parametrize("parent_score,refs", [(1, None), (0, []), (0, [
    {"claim": "invalid", "counterexample_ids": [-1, 10], "severity": 1, "proposed_fix": "fix"}])])
def test_b_no_corroborated_counterexample_uses_real_reflective_outputs(parent_score, refs):
    batch = [{"Inputs": {"question": "q"}, "Generated Outputs": {"answer": str(parent_score)},
              "Feedback": "Improve the score"}]
    proposer, task, reflection, budget, dataset = make_b(["unused"], batch=batch, refs=refs)
    assert proposer({"answer": "parent"}, dataset, ["answer"]) == {"answer": "parent"}
    assert not task.history
    assert len(reflection.history) == budget.used_calls() == 1
    assert proposer.log[-1]["rationale"]["reason"] == "no_corroborated_counterexample"
    assert proposer.log[-1]["calls_consumed"] == 0
    assert proposer.log[-1]["critique_calls_consumed"] == 1


@pytest.mark.parametrize("rewrites,expected_task_calls", [(["  parent\n"], 0), (["first", " first\n"], 1)])
def test_b_normalized_duplicate_skips_all_task_calls(rewrites, expected_task_calls):
    proposer, task, _, _, batch = make_b(rewrites, task_scores=[0.2], batch=[{
        "Inputs": {"question": "q"}, "Generated Outputs": {"answer": "0.2"}, "Feedback": "Improve"}])
    proposer.archive["q"] = 0.5
    assert proposer({"answer": "parent"}, batch, ["answer"]) == {"answer": "parent"}
    assert len(task.history) == expected_task_calls
    assert proposer.log[-1]["rationale"]["reason"] == "duplicate_rewrite"
    assert proposer.log[-1]["tested_questions"] == []


def test_m_both_retry_prompts_receive_all_failed_rivals():
    from experiments.theory_driven.tests.test_millian import make_proposer
    proposer, task, reflection, _, _, events, batch = make_proposer(
        prosecution=(0, 0), millian=(0, 1), max_attempts=2)
    proposer({"answer": "parent"}, batch, ["answer"])
    failures = failed_input(reflection.history[3])
    assert [item["instruction"] for item in failures] == ["Prosecution", "Millian"]
    assert failed_input(reflection.history[4]) == failures
    assert failures[0]["per_question"][0] == {
        "question_id": "q0", "role": "counterexample", "parent_score": 0,
        "rewrite_score": 0, "fixed_or_kept": False}
    assert failures[0]["per_question"][-1]["fixed_or_kept"] is False
    assert failures[1]["per_question"][-1]["fixed_or_kept"] is True
    assert events[0]["attempts"][1]["prosecution"]["rationale"]["reason"] == "duplicate_rewrite"
    assert len(task.history) == 8


@pytest.mark.parametrize("text", ["Rival", " Rival\n"])
def test_m_identical_rivals_are_evaluated_once_and_reused(text):
    from experiments.theory_driven.tests.test_millian import make_proposer
    proposer, task, _, _, budget, events, batch = make_proposer(
        prosecution=(1, 1), prosecution_text="Rival", millian_text=text)
    proposer({"answer": "parent"}, batch, ["answer"])
    assert len(task.history) == 6
    assert budget.used_calls() == 10
    assert events[0]["prosecution"]["evidence"] == events[0]["millian"]["evidence"]
    assert events[0]["millian"]["rationale"]["evaluation_reused"] is True
    assert events[0]["role_calls"]["millian_survival"] == 0


def test_m_no_counterexample_only_calls_prosecution_and_defense():
    from experiments.theory_driven.tests.test_millian import make_proposer
    proposer, task, reflection, defender, budget, events, batch = make_proposer()
    for row in batch["answer"]:
        row["Generated Outputs"] = {"answer": "1"}
        row["Feedback"] = "Correct; no score in this feedback"
    assert proposer({"answer": "parent"}, batch, ["answer"]) == {"answer": "parent"}
    assert not task.history
    assert len(reflection.history) == len(defender.history) == 1
    assert budget.used_calls() == 2
    assert events[0]["calls_consumed"] == 0
    assert events[0]["rationale"]["reason"] == "no_corroborated_counterexample"
    assert events[0]["role_calls"] == {"prosecutor": 1, "defender": 1, "parent": 0}


def test_m_parent_duplicates_do_not_even_observe_archive():
    from experiments.theory_driven.tests.test_millian import make_proposer
    proposer, task, _, _, _, events, batch = make_proposer(
        prosecution_text="parent", millian_text=" parent\n")
    for row in batch["answer"]:
        row["Generated Outputs"] = {"answer": "0.2"}
    proposer.gold["archived"] = dspy.Example(question="archived", answer="1").with_inputs("question")
    proposer.archive["archived"] = 1
    proposer({"answer": "parent"}, batch, ["answer"])
    assert not task.history
    assert events[0]["prosecution"]["rationale"]["reason"] == "duplicate_rewrite"
    assert events[0]["tested_questions"] == []


@pytest.mark.parametrize("arm", ["B", "M"])
def test_real_hotpot_reflective_answer_short_circuits_without_numeric_feedback(arm):
    from experiments.theory_driven.arms.millian import MillianProposer
    from experiments.theory_driven.benchmarks.hotpotqa import HotpotQAProgram, hotpotqa_metric
    refs = [{"claim": "incorrect", "counterexample_ids": [0], "severity": 1, "proposed_fix": "fix"}]
    reflection = DummyLM([{"refutations": refs}, {"rebuttals": []}])
    task = DummyLM([])
    gold = dspy.Example(question="Which city?", paragraphs=[], answer="Paris").with_inputs("question", "paragraphs")
    budget = UnifiedBudget(100, 1, reflection, task)
    cls = PopperianProposer if arm == "B" else MillianProposer
    proposer = cls(HotpotQAProgram(), hotpotqa_metric, {gold.question: gold}, reflection, task, budget)
    batch = {"answer": [{"Inputs": {"question": gold.question, "context": ""},
                         "Generated Outputs": {"answer": "Paris"}, "Feedback": "Correct"}]}
    assert proposer({"answer": "parent"}, batch, ["answer"]) == {"answer": "parent"}
    assert not task.history
    assert budget.used_calls() == (1 if arm == "B" else 2)
    assert proposer.log[-1]["calls_consumed"] == 0
