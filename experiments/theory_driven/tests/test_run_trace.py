"""Offline persistence of sparse events and Popperian admission decisions."""

import json
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

from experiments.theory_driven.arms.popperian import PopperianProposer, PopperianArm
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.tests.test_popperian_regressions import AnswerProgram, metric

from experiments.theory_driven.events.run_trace import RunTrace
from experiments.theory_driven.events.storage import EventStorage


def read_events(root, trace):
    return [json.loads(line) for line in (root / trace.event_path).read_text().splitlines()]


def test_sparse_trace_is_persisted_immediately_and_finalized(tmp_path):
    trace = RunTrace(tmp_path, "experiment", "benchmark", "B", {"seed": 7})
    other = RunTrace(tmp_path, "experiment", "benchmark", "B", {"seed": 7})
    assert trace.run_id != other.run_id
    assert not Path(trace.event_path).is_absolute()
    initial = read_events(tmp_path, trace)
    assert [event["event_type"] for event in initial] == ["initialization"]
    assert initial[0]["metadata"] == {"seed": 7, "arm_name": "B"}
    paths = trace.finalize({"status": "error", "error": "provider failure"})
    assert paths == {"epistemic_events": trace.event_path}
    events = EventStorage(tmp_path).load_events(tmp_path / trace.event_path)
    assert [event.event_type for event in events] == ["initialization", "finalization"]
    assert events[-1].metadata["error"] == "provider failure"
    assert all(event.run_id == trace.run_id for event in events)


def proposal_record(survived):
    return {
        "attempt": 1, "predictor": "answer", "old_instruction": "old secret",
        "new_instruction": "new secret", "survived": survived,
        "refutations": [{"claim": "secret claim", "counterexample_ids": [0],
                         "severity": 1.0, "proposed_fix": "secret fix"}],
        "tested_questions": ["secret question"], "calls_consumed": 3,
        "cost_consumed": 0.25,
    }


def test_proposal_outcomes_are_saved_before_finalize(tmp_path):
    trace = RunTrace(tmp_path, "experiment", "benchmark", "B", {})
    for survived in (False, True):
        trace.proposal(proposal_record(survived))
        event = read_events(tmp_path, trace)[-1]
        step = event["step"]
        assert step["decision"]["accepted"] is survived
        assert step["decision"]["survival_check_passed"] is survived
        assert "GEPA" in step["decision"]["decision_reason"]
        assert event["metadata"]["decision_scope"] == "proposal_survival_admission"
        assert event["metadata"]["tested_questions"] == ["secret question"]
        assert step["conjecture"]["old_instruction"] == "old secret"
        assert step["conjecture"]["new_instruction"] == "new secret"
        assert step["conjecture"]["module_name"] == "answer"
        assert step["refutations"][0]["claim"] == "secret claim"
        assert step["test_selection"]["test_type"] == "survival_check"
        assert step["test_selection"]["example_count"] == 1
        assert step["calls_consumed"] == 3
        assert step["cost_consumed"] == 0.25
        assert step["lakatos_label"] == "unknown"
    assert len(read_events(tmp_path, trace)) == 3


def test_sanitizer_covers_all_persisted_event_text(tmp_path):
    trace = RunTrace(
        tmp_path, "secret experiment", "secret benchmark", "B", {"nested": {"value": "secret"}},
        sanitizer=lambda text: text.replace("secret", "[REDACTED]"),
    )
    trace.proposal(proposal_record(True))
    trace.finalize({"error": "secret"})
    text = (tmp_path / trace.event_path).read_text()
    assert "secret" not in text
    assert "[REDACTED]" in text


@pytest.mark.parametrize("later_error", [False, True])
def test_proposer_persists_each_attempt_before_next_attempt(tmp_path, later_error, monkeypatch):
    trace = RunTrace(tmp_path, "experiment", "benchmark", "B", {})
    task = DummyLM([{"answer": "no"}, {"answer": "no"}, {"answer": "yes"}])
    critic = DummyLM([
        {"refutations": proposal_record(True)["refutations"]},
        {"new_instruction": "first"}, {"new_instruction": "second"},
    ])
    gold = dspy.Example(question="q", answer="yes").with_inputs("question")
    budget = UnifiedBudget(100, 1.0, task, critic)
    proposer = PopperianProposer(
        AnswerProgram(), metric, {"q": gold}, critic, task, budget,
        event_sink=trace.proposal,
    )
    rewrite = proposer.rewrite
    def observed_rewrite(**kwargs):
        if proposer.log:
            events = read_events(tmp_path, trace)
            assert events[-1]["step"]["decision"]["accepted"] is False
            if later_error:
                raise RuntimeError("later rewrite failed")
        return rewrite(**kwargs)
    monkeypatch.setattr(proposer, "rewrite", observed_rewrite)
    candidate = {"answer": "parent"}
    dataset = {"answer": [{"Inputs": {"question": "q"}}]}
    if later_error:
        with pytest.raises(RuntimeError, match="later rewrite failed"):
            proposer(candidate, dataset, ["answer"])
    else:
        assert proposer(candidate, dataset, ["answer"]) == {"answer": "second"}
    steps = [event["step"] for event in read_events(tmp_path, trace)[1:]]
    assert [step["calls_consumed"] for step in steps] == ([4] if later_error else [4, 2])
    assert [step["decision"]["accepted"] for step in steps] == ([False] if later_error else [False, True])
    assert all(step["conjecture"]["old_instruction"] == "parent" for step in steps)
    assert all(step["conjecture"]["module_name"] == "answer" for step in steps)


def test_arm_compile_forwards_proposal_sink(tmp_path):
    trace = RunTrace(tmp_path, "experiment", "benchmark", "B", {})
    task = DummyLM([{"answer": "no"}] * 50)
    critic = DummyLM([
        {"refutations": proposal_record(True)["refutations"]},
        {"new_instruction": "first"}, {"new_instruction": "second"},
    ])
    budget = UnifiedBudget(3, 1.0, task, critic)
    arm = PopperianArm("B", "offline", budget, max_metric_calls=50, reflection_minibatch_size=1)
    gold = dspy.Example(question="q", answer="yes").with_inputs("question")
    arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task,
                reflection_lm=critic, event_sink=trace.proposal)
    assert len(read_events(tmp_path, trace)) == 3


def test_survival_event_reports_only_questions_actually_tested(tmp_path):
    class CostedDummyLM(DummyLM):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            self.history[-1]["cost"] = 0.125
            return result

    trace = RunTrace(tmp_path, "experiment", "benchmark", "B", {})
    task = CostedDummyLM([{"answer": "no"}] * 4)
    refutations = [{"claim": "wrong", "counterexample_ids": [0, 1],
                    "severity": 1.0, "proposed_fix": "fix"}]
    critic = CostedDummyLM([{"refutations": refutations}, {"new_instruction": "new"}])
    golds = {q: dspy.Example(question=q, answer="yes").with_inputs("question") for q in ("q1", "q2")}
    proposer = PopperianProposer(
        AnswerProgram(), metric, golds, critic, task,
        UnifiedBudget(100, 1.0, task, critic), max_attempts=1, event_sink=trace.proposal,
    )
    proposer({"answer": "old"}, {"answer": [{"Inputs": {"question": q}} for q in golds]}, ["answer"])
    event = read_events(tmp_path, trace)[-1]
    assert event["metadata"]["tested_questions"] == ["q1", "q2"]
    assert event["step"]["test_selection"]["example_count"] == 2
    assert event["step"]["calls_consumed"] == 6
    assert event["step"]["cost_consumed"] == 0.75
