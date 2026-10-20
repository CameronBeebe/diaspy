"""Offline prompt/feedback contracts; DummyLM does not establish critic quality."""

from typing import Any

import dspy
from dspy.utils import DummyLM
import pytest

from experiments.theory_driven.arms.millian import Defense, DialecticalRewrite, MillianProposer
from experiments.theory_driven.arms.popperian import PopperianProposer, Refutation, Rewrite
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.benchmarks.hotpotqa import (
    HotpotQABenchmark, hotpotqa_metric,
)


def test_common_hotpotqa_predictor_requests_only_minimal_answer_without_gold():
    program = HotpotQABenchmark().get_program()
    gold = dspy.Example(
        question="When was the division established?",
        paragraphs=[{"title": "Division", "text": "Established in 2002."}],
        answer="GOLD_ONLY_SENTINEL",
        supporting_titles=["SUPPORT_ONLY_SENTINEL"],
    ).with_inputs("question", "paragraphs")
    lm = DummyLM([{"answer": "2002"}])
    with dspy.context(lm=lm):
        result = program(**gold.inputs())
    prompt = "\n".join(message["content"] for message in lm.history[0]["messages"])
    assert "minimal short answer" in prompt.lower()
    assert "no explanations" in prompt.lower()
    assert "GOLD_ONLY_SENTINEL" not in prompt
    assert "SUPPORT_ONLY_SENTINEL" not in prompt
    predictor: Any = dict(program.named_predictors())["answer"]
    assert set(predictor.signature.input_fields) == {"context", "question"}
    assert set(predictor.signature.output_fields) == {"answer"}
    assert result.answer == "2002"


@pytest.mark.parametrize("answer,expected", [
    ("2002", 1.0),
    ("The division was established in 2002.", 1 / 3),
    ("It was not 2002.", 0.4),
    ("1999", 0.0),
])
def test_feedback_explains_token_f1_without_certifying_semantic_correctness(answer, expected):
    result = hotpotqa_metric(
        dspy.Example(answer="2002", supporting_titles=["Division"]),
        dspy.Prediction(answer=answer, retrieved_titles=["Division"]),
    )
    assert result.score == pytest.approx(expected)
    feedback = result.feedback.lower()
    assert "token-f1" in feedback
    assert "precision" in feedback and "recall" in feedback
    assert "extra explanation tokens" in feedback
    assert "not a semantic correctness judgment" in feedback
    assert "minimal short answer" in feedback
    assert "no explanations" in feedback
    assert "gold answer: 2002" in feedback


@pytest.mark.parametrize("titles,status", [
    (["Division"], "no annotated support titles missing"),
    ([], "potential retrieval limitation, not proof of an instruction defect"),
    (None, "support coverage unknown"),
])
def test_feedback_separates_observed_retrieval_coverage_from_instruction_defects(titles, status):
    pred = dspy.Prediction(answer="1999")
    if titles is not None:
        pred.retrieved_titles = titles
    result = hotpotqa_metric(
        dspy.Example(answer="2002", supporting_titles=["Division"]), pred,
    )
    assert result.score == 0.0
    assert status in result.feedback.lower()
    assert "title coverage does not establish evidence sufficiency" in result.feedback.lower()
    assert "ambiguity" in result.feedback.lower()
    if titles == []:
        assert "Missed supporting-fact titles: Division" in result.feedback


@pytest.mark.parametrize("proposer_type", [PopperianProposer, MillianProposer])
def test_critic_receives_metric_guidance_and_can_decline_to_refute(proposer_type):
    program = HotpotQABenchmark().get_program()
    gold = dspy.Example(
        question="When?", paragraphs=[], answer="2002", supporting_titles=["Division"],
    ).with_inputs("question", "paragraphs")
    prediction = dspy.Prediction(answer="The division began in 2002.", retrieved_titles=[])
    feedback = hotpotqa_metric(gold, prediction).feedback
    task = DummyLM([])
    reflection = DummyLM([{"refutations": []}, {"rebuttals": []}], adapter=dspy.JSONAdapter())
    proposer = proposer_type(
        program, hotpotqa_metric, {gold.question: gold}, reflection, task,
        UnifiedBudget(100, 1, task, reflection),
    )
    predictor: Any = dict(program.named_predictors())["answer"]
    instruction = predictor.signature.instructions
    batch = {"answer": [{
        "Inputs": {"question": gold.question, "context": "No relevant support."},
        "Generated Outputs": {"answer": prediction.answer}, "Feedback": feedback,
    }]}
    assert proposer({"answer": instruction}, batch, ["answer"]) == {}
    assert not task.history
    assert proposer.log[-1]["refutations"] == []
    assert proposer.log[-1]["returned_unchanged_parent"] is True
    messages = reflection.history[0]["messages"]
    guidance = messages[0]["content"].lower()
    assert "low score is an observation, not proof" in guidance
    assert "return an empty list" in guidance
    assert "do not invent requirements" in guidance
    assert "actual instruction, output, and available evidence" in guidance
    assert "task and metric guidance in the feedback" in guidance
    assert "token-f1" not in guidance  # task specifics belong to feedback, not reusable arms
    assert feedback in messages[-1]["content"]


@pytest.mark.parametrize("signature,response,extra", [
    (Rewrite, {"new_instruction": "Parent"}, {"failed_attempts": []}),
    (Defense, {"rebuttals": []}, {"examples_with_feedback": "Observed score only."}),
    (DialecticalRewrite, {"new_instruction": "Parent", "rejected_refutations": "Unsupported"},
     {"rebuttals": [], "failed_attempts": []}),
])
def test_downstream_roles_do_not_convert_critic_claims_into_facts(signature, response, extra):
    refutation = Refutation(
        claim="Low score proves reasoning failed", counterexample_ids=[0], severity=1,
        proposed_fix="Add an explanation",
    )
    lm = DummyLM([response])
    with dspy.context(lm=lm):
        dspy.Predict(signature)(current_instruction="Parent", refutations=[refutation], **extra)
    guidance = " ".join(lm.history[0]["messages"][0]["content"].lower().split())
    assert "refutations are hypotheses, not established facts" in guidance
    assert "preserve the task's output contract" in guidance
    assert "do not invent requirements" in guidance
    assert "token-f1" not in guidance
