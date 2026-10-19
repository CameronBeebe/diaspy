"""Stock GEPA tracing must observe decisions without changing any LM calls."""

import json
from typing import Any, cast

import dspy
import pytest
from dspy.utils import DummyLM

from experiments.theory_driven.arms.gepa_default import GEPADefaultArm
from experiments.theory_driven.config.budget_tracker import UnifiedBudget


class AnswerProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.answer(question=question)


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return float(pred.answer)


class StockObserver:
    def __init__(self):
        self.proposals = []
        self.accepted = []
        self.rejected = []

    def on_proposal_end(self, event):
        self.proposals.append(dict(event["new_instructions"]))

    def on_candidate_accepted(self, event):
        self.accepted.append(dict(event))

    def on_candidate_rejected(self, event):
        self.rejected.append(dict(event))


def run_stock(wrapped, scores, instructions, observer=None, **kwargs):
    task = DummyLM([{"answer": str(score)} for score in scores])
    reflection = DummyLM(
        [{"new_instruction": text} for text in instructions], adapter=dspy.JSONAdapter(),
    )
    budget = UnifiedBudget(100, 1, task, reflection)
    train = [dspy.Example(question="training question", answer="1").with_inputs("question")]
    val = [dspy.Example(question="validation question", answer="1").with_inputs("question")]
    callbacks = [observer] if observer is not None else []
    events = []
    if wrapped:
        arm = GEPADefaultArm("A", "offline", budget, max_metric_calls=len(scores),
                             reflection_minibatch_size=1, callbacks=callbacks, **kwargs)
        optimized = arm.compile(AnswerProgram(), metric, train, val, task_lm=task,
                                reflection_lm=reflection, event_sink=events.append)
    else:
        optimizer = dspy.GEPA(
            metric=metric, reflection_lm=cast(Any, reflection), max_metric_calls=len(scores),
            reflection_minibatch_size=1, use_merge=False, seed=0, track_stats=True,
            num_threads=1, gepa_kwargs={"stop_callbacks": [budget], "callbacks": callbacks, **kwargs},
        )
        with dspy.context(lm=task):
            optimized = optimizer.compile(AnswerProgram(), trainset=train, valset=val)
    return optimized, task, reflection, events


def test_stock_proposals_decisions_and_lm_histories_are_unchanged(caplog):
    instructions = ["Accepted first line\nFull second line with reasoning.",
                    "Rejected first line\nDo not truncate this either."]
    # The first rewrite is accepted into GEPA's pool but is NOT the final best.
    scores = [0.5, 0.2, 0.8, 0.4, 0.2, 0.2]
    stock, instrumented = StockObserver(), StockObserver()
    plain, ptask, preflection, _ = run_stock(False, scores, instructions, stock)
    traced, task, reflection, events = run_stock(True, scores, instructions, instrumented)

    assert len(events) == 2
    assert stock.proposals == instrumented.proposals == [{"answer": text} for text in instructions]
    assert stock.accepted == instrumented.accepted
    assert stock.rejected == instrumented.rejected
    assert len(task.history) == len(ptask.history) == 6
    assert len(reflection.history) == len(preflection.history) == 2  # no critic calls
    assert [row["messages"] for row in task.history] == [row["messages"] for row in ptask.history]
    assert [row["messages"] for row in reflection.history] == [row["messages"] for row in preflection.history]
    # DSPy dynamically constructs signatures; its annotations omit instructions.
    seed = cast(Any, dict(AnswerProgram().named_predictors())["answer"].signature).instructions
    assert cast(Any, dict(traced.named_predictors())["answer"].signature).instructions == seed
    assert cast(Any, dict(plain.named_predictors())["answer"].signature).instructions == seed
    assert [event["accepted"] for event in events] == [True, False]
    assert [event["proposed_instructions"] for event in events] == stock.proposals
    assert all(event["kind"] == "gepa_proposal" for event in events)
    assert [event["parent_instructions"] for event in events] == [{"answer": seed}] * 2
    assert all(event["minibatch_ids"] == [0] for event in events)
    assert all(event["minibatch_questions"] == ["training question"] for event in events)
    assert [event["parent_scores"] for event in events] == [[0.2], [0.2]]
    assert [event["proposed_scores"] for event in events] == [[0.8], [0.2]]
    assert events[0]["candidate_id"] == 1
    assert events[0]["validation_score"] == 0.4
    assert events[0]["validation_scores"] == {0: 0.4}
    assert events[1]["candidate_id"] is None
    assert events[1]["reason"] == stock.rejected[0]["reason"]
    assert json.loads(json.dumps(events))[0]["proposed_instructions"]["answer"] == instructions[0]
    assert "reflection failed" not in caplog.text
    assert "Callback " not in caplog.text
    assert "Traceback" not in caplog.text


def test_failed_reflection_is_not_claimed_as_rejection():
    # GEPA retries a failed batched reflection once; the wrapper must not retry.
    _, ptask, preflection, _ = run_stock(False, [0, 0], [" ", " "])
    _, task, reflection, events = run_stock(True, [0, 0], [" ", " "])
    assert len(task.history) == len(ptask.history) == 2
    assert len(reflection.history) == len(preflection.history) == 2
    assert len(events) == 2
    assert all(event["accepted"] is None for event in events)
    assert all(event["status"] == "proposal_failed" for event in events)
    assert all("empty instruction" in event["error"] for event in events)
    assert all(event["proposed_instructions"] is None for event in events)


def test_fatal_acceptance_error_preserves_pending_proposal():
    class BrokenAcceptance:
        def should_accept(self, proposal, state):
            raise RuntimeError("scripted acceptance failure")

    events = []
    task = DummyLM([{"answer": "0"}] * 3)
    reflection = DummyLM([{"new_instruction": "First line\nPending full text"}],
                         adapter=dspy.JSONAdapter())
    arm = GEPADefaultArm("A", "offline", UnifiedBudget(100, 1, task, reflection),
                         max_metric_calls=3, reflection_minibatch_size=1,
                         acceptance_criterion=BrokenAcceptance())
    gold = dspy.Example(question="q", answer="1").with_inputs("question")
    with pytest.raises(RuntimeError, match="scripted acceptance failure"):
        arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task,
                    reflection_lm=reflection, event_sink=events.append)
    assert len(events) == 1
    assert events[0]["accepted"] is None
    assert events[0]["status"] == "interrupted"
    assert "scripted acceptance failure" in events[0]["error"]
    assert events[0]["proposed_instructions"] == {"answer": "First line\nPending full text"}
    assert events[0]["parent_scores"] == events[0]["proposed_scores"] == [0.0]


def test_nonseed_parent_and_final_program_are_preserved():
    instructions = ["Intermediate\nWhole instruction", "Winner\nWhole final instruction"]
    scores = [0, 0.1, 0.8, 0.8, 0.2, 0.9, 0.9]
    plain, ptask, preflection, _ = run_stock(False, scores, instructions)
    traced, task, reflection, events = run_stock(True, scores, instructions)
    assert cast(Any, dict(plain.named_predictors())["answer"].signature).instructions == instructions[1]
    assert cast(Any, dict(traced.named_predictors())["answer"].signature).instructions == instructions[1]
    assert events[1]["parent_instructions"] == {"answer": instructions[0]}
    assert events[1]["parent_candidate_id"] == 1
    assert [event["candidate_id"] for event in events] == [1, 2]
    assert len(task.history) == len(ptask.history) == 7
    assert len(reflection.history) == len(preflection.history) == 2


def test_delegation_preserves_input_output_identity_and_full_parent():
    from experiments.theory_driven.arms.gepa_trace import GEPAProposalTrace

    parent = {"answer": "Parent\nAnswer", "other": "Preserved\nOther component"}
    batch = {"answer": [{"Inputs": {"question": "q"}, "Feedback": "bad"}]}
    components = ["answer"]
    updates = {"answer": "Rewritten\nFull instruction"}
    calls = []

    def delegate(candidate, reflective_dataset, components_to_update):
        assert candidate is parent and reflective_dataset is batch and components_to_update is components
        calls.append(1)
        return updates

    trace = GEPAProposalTrace()
    cast(Any, trace).delegate = delegate
    assert trace(parent, batch, components) is updates
    trace.flush()
    assert len(calls) == 1
    assert trace.log[0]["parent_instructions"] == parent
    assert trace.log[0]["proposed_candidate"] == {**parent, **updates}
    assert trace.log[0]["accepted"] is None  # proposal existence is not admission


def test_no_instruction_updates_are_not_an_actual_rejection():
    from experiments.theory_driven.arms.gepa_trace import GEPAProposalTrace

    trace = GEPAProposalTrace()
    # The real stock delegate omits a component absent from the reflective data.
    assert trace({"answer": "Parent"}, {}, ["answer"]) == {}
    trace.flush()
    assert trace.log[0]["status"] == "no_updates"
    assert trace.log[0]["accepted"] is None


def test_acceptance_is_observed_not_recomputed_from_minibatch_scores():
    # GEPA explicitly allows equality with this nondefault acceptance policy.
    _, _, _, events = run_stock(True, [0, 0, 0, 0], ["Equal scores\nStill admitted"],
                                acceptance_criterion="improvement_or_equal")
    assert events[0]["parent_scores"] == events[0]["proposed_scores"] == [0.0]
    assert events[0]["accepted"] is True
    assert events[0]["candidate_id"] == 1


def test_failed_reflection_retry_does_not_steal_successful_proposal_decision():
    _, task, reflection, events = run_stock(True, [0, 0, 0], [" ", "Valid\nRetry"])
    assert len(task.history) == 3 and len(reflection.history) == 2
    assert len(events) == 2
    assert events[0]["status"] == "proposal_failed" and events[0]["accepted"] is None
    assert events[1]["status"] == "rejected" and events[1]["accepted"] is False
    assert events[1]["proposed_instructions"] == {"answer": "Valid\nRetry"}
