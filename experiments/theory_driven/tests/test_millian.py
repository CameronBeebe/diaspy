"""Real offline typed debate calls and score-based Millian selection."""

import json
from typing import Any

import dspy
import pytest
from dspy.utils import DummyLM

from experiments.theory_driven.config.budget_tracker import UnifiedBudget


class AnswerProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.answer(question=question)


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(score=float(pred.answer), feedback="Improve the score")


def make_proposer(parent=(0, 0, 1, 1), prosecution=(1, 0), millian=(0.6, 1),
                  stance="contest", severity=0, shared=False, max_attempts=1,
                  prosecution_text="Prosecution", millian_text="Millian", **kwargs):
    from experiments.theory_driven.arms.millian import MillianProposer

    refs = [{"claim": "fails q0", "counterexample_ids": [0], "severity": severity,
             "proposed_fix": "improve q0 without harming q2"}]
    rebuttals = [{"refutation_index": 0, "stance": stance, "argument": "preserve q2",
                  "protected_ids": [1, 2]}]
    critique = {"refutations": refs}
    defense = {"rebuttals": rebuttals}
    rewrites = [{"new_instruction": prosecution_text},
                {"new_instruction": millian_text, "rejected_refutations": "None; preserve q2"}]
    reflection = DummyLM(([critique, defense] if shared else [critique]) + rewrites * max_attempts)
    defender = reflection if shared else DummyLM([defense])
    task = DummyLM([{"answer": str(score)} for score in (*parent, *((*prosecution, *millian) * max_attempts))])
    gold = {f"q{i}": dspy.Example(question=f"q{i}", answer="1").with_inputs("question")
            for i in range(len(parent))}
    budget = UnifiedBudget(1000, 1, task, reflection, defender)
    events = []
    proposer = MillianProposer(
        AnswerProgram(), metric, gold, reflection, task, budget,
        defender_lm=defender, min_survival_questions=2, max_attempts=max_attempts,
        event_sink=events.append, **kwargs,
    )
    batch: dict[str, list[dict[str, Any]]] = {"answer": [{"Inputs": {"question": question}} for question in gold]}
    return proposer, task, reflection, defender, budget, events, batch


def test_defense_is_corroborated_and_both_real_rewrites_are_scored():
    from experiments.theory_driven.arms.millian import DebateRecord

    proposer, task, reflection, defender, budget, events, batch = make_proposer()
    result = proposer({"answer": "Parent"}, batch, ["answer"])
    assert result == {"answer": "Millian"}
    assert len(events) == len(proposer.log) == 1
    record = DebateRecord.model_validate(events[0])
    payload = json.loads(record.model_dump_json())
    assert payload["kind"] == "millian_debate"
    assert payload["policy"] == "strict"
    assert payload["winner"] == "millian"
    assert payload["received_view_survived"] is False
    assert payload["prosecution"]["admitted"] is False
    assert payload["millian"]["admitted"] is True
    left, right = payload["prosecution"]["evidence"], payload["millian"]["evidence"]
    assert [(e["question"], e["role"], e["parent_score"]) for e in left] == [
        (e["question"], e["role"], e["parent_score"]) for e in right]
    assert any(e["question"] == "q2" and e["role"] == "defense" for e in left)
    assert any(e["question"] == "q1" and e["role"] == "uncorroborated_defense" for e in left)
    assert proposer.archive == {"q0": 0.6}
    assert len(task.history) == 8
    assert len(reflection.history) == 3
    assert len(defender.history) == 1
    assert budget.used_calls() == payload["calls_consumed"] == 12
    assert sum(payload["role_calls"].values()) == 12
    assert payload["role_calls"] == {"prosecutor": 1, "defender": 1, "prosecution_rewrite": 1,
                                     "millian_rewrite": 1, "parent": 4, "prosecution_survival": 2,
                                     "millian_survival": 2}
    assert "preserve q2" in reflection.history[2]["messages"][-1]["content"]


def test_real_gepa_compile_and_event_sink(caplog):
    from experiments.theory_driven.arms.millian import MillianArm, MillianProposer

    task = DummyLM([{"answer": "0"}] * 50)
    reflection = DummyLM([
        {"refutations": [{"claim": "wrong", "counterexample_ids": [0], "severity": 0, "proposed_fix": "fix"}]},
        {"rebuttals": [{"refutation_index": 0, "stance": "concede", "argument": "wrong", "protected_ids": []}]},
        {"new_instruction": "Prosecution"},
        {"new_instruction": "Millian", "rejected_refutations": "none"},
    ])
    budget = UnifiedBudget(3, 1, task, reflection)
    events = []
    arm = MillianArm("M", "offline", budget, max_metric_calls=50, max_attempts=1,
                     reflection_minibatch_size=1, survival_policy="majority")
    gold = dspy.Example(question="q", answer="1").with_inputs("question")
    optimized = arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task,
                            reflection_lm=reflection, event_sink=events.append)
    assert isinstance(optimized, dspy.Module)
    assert isinstance(arm.proposer, MillianProposer)
    assert arm.proposer.defender_lm is reflection
    assert len(reflection.history) == 4
    assert budget.stop_triggered
    assert len(events) == 1
    assert events[0]["policy"] == "majority"
    assert arm.get_stats()["survival_policy"] == "majority"
    assert "reflection failed" not in caplog.text
    assert "Traceback" not in caplog.text


def test_arm_rejects_unknown_policy_at_construction():
    from experiments.theory_driven.arms.millian import MillianArm

    with pytest.raises(ValueError):
        MillianArm("M", "offline", UnifiedBudget(10, 1), survival_policy="unknown")


@pytest.mark.parametrize("policy", ["strict", "majority", "veto"])
def test_stance_and_severity_never_change_admission_or_archive(policy):
    outcomes = []
    for stance, severity in [("contest", 0), ("concede", 1)]:
        proposer, _, _, _, _, events, batch = make_proposer(
            stance=stance, severity=severity, survival_policy=policy,
        )
        result = proposer({"answer": "Parent"}, batch, ["answer"])
        outcomes.append((result, proposer.archive.copy(), events[0]["winner"],
                         events[0]["prosecution"]["rationale"], events[0]["millian"]["rationale"]))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("prosecution,millian,ptext,mtext,winner", [
    ((1, 1), (0.6, 1), "long prosecution", "M", "prosecution"),
    ((0.6, 1), (1, 1), "P", "long millian", "millian"),
    ((1, 1), (1, 1), "long prosecution", "M", "millian"),
    ((1, 1), (1, 1), "P", "long millian", "prosecution"),
    ((1, 1), (1, 1), "PP", "MM", "prosecution"),
])
def test_selection_orders_gain_then_length_then_prosecution(prosecution, millian, ptext, mtext, winner):
    proposer, _, _, _, _, events, batch = make_proposer(
        prosecution=prosecution, millian=millian, prosecution_text=ptext, millian_text=mtext,
    )
    result = proposer({"answer": "Parent"}, batch, ["answer"])
    assert events[0]["winner"] == winner
    assert result["answer"] == (ptext if winner == "prosecution" else mtext)
    assert events[0]["prosecution"]["admitted"] and events[0]["millian"]["admitted"]


def test_failed_rivals_keep_parent_retry_cap_and_emit_one_round_record():
    proposer, task, reflection, defender, budget, events, batch = make_proposer(
        prosecution=(0, 0), millian=(0, 1), max_attempts=2,
    )
    assert proposer({"answer": "Not base"}, batch, ["answer"]) == {}  # Returns {} when all rivals fail
    assert len(events) == 1
    assert len(events[0]["attempts"]) == 2
    assert events[0]["winner"] is None
    assert events[0]["received_view_survived"] is True
    assert events[0]["returned_unchanged_parent"] is True
    assert not proposer.archive
    assert len(task.history) == 8  # repeated rivals on attempt 2 are not evaluated again
    assert len(reflection.history) == 5
    assert len(defender.history) == 1
    assert budget.used_calls() == events[0]["calls_consumed"] == 14


def test_aliased_reflection_and_defender_have_exact_role_counts():
    proposer, task, reflection, defender, budget, events, batch = make_proposer(shared=True)
    assert reflection is defender
    proposer({"answer": "Parent"}, batch, ["answer"])
    record = events[0]
    assert len(reflection.history) == 4
    assert sum(record["role_calls"].values()) == budget.used_calls() == 12
    assert record["role_calls"]["defender"] == 1
    assert record["role_calls"]["prosecutor"] == 1
    assert record["cost_consumed"] == budget.used_usd() == 0
    assert len(task.history) == 8


def test_protected_partial_score_at_median_is_kept_not_dropped():
    proposer, _, _, _, _, events, batch = make_proposer(
        parent=(0, 0.6, 0.6, 1), prosecution=(1, 0.6, 0.6), millian=(1, 0.6, 0.6),
    )
    proposer({"answer": "Parent"}, batch, ["answer"])
    rows = events[0]["joint_set"]
    assert {e["question"] for e in rows if e["role"] == "defense"} == {"q1", "q2"}
    assert not any(e["role"] == "uncorroborated_defense" for e in rows)


def test_zero_attempts_make_no_calls():
    proposer, _, _, _, budget, events, batch = make_proposer(max_attempts=0)
    assert proposer({"answer": "Parent"}, batch, ["answer"]) == {"answer": "Parent"}
    assert budget.used_calls() == 0
    assert not events


class ContextProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Predict("question, context -> evidence")
        self.answer = dspy.Predict("question, evidence -> answer")

    def forward(self, question, context):
        evidence = self.retrieve(question=question, context=context).evidence
        return self.answer(question=question, evidence=evidence)


def test_full_parent_and_all_inputs_are_used_in_both_rival_rollouts():
    from experiments.theory_driven.arms.millian import MillianProposer

    task = DummyLM([
        {"evidence": "support"}, {"answer": "0"},
        {"evidence": "support"}, {"answer": "0.5"},
        {"evidence": "support"}, {"answer": "1"},
    ])
    reflection = DummyLM([
        {"refutations": [{"claim": "wrong", "counterexample_ids": [0], "severity": 0, "proposed_fix": "fix"}]},
        {"rebuttals": []},
        {"new_instruction": "P"},
        {"new_instruction": "M", "rejected_refutations": "none"},
    ])
    budget = UnifiedBudget(100, 1, task, reflection)
    example = dspy.Example(question="q", context="required context", answer="1").with_inputs("question", "context")
    program = ContextProgram()
    proposer = MillianProposer(program, metric, {"q": example}, reflection, task, budget, max_attempts=1)
    parent = {"retrieve": "Parent retrieval", "answer": "Parent answer"}
    result = proposer(parent, {"answer": [{"Inputs": {"question": "q"}}]}, ["answer"])
    assert result == {"answer": "M"}
    for index in (0, 2, 4):
        assert "Parent retrieval" in task.history[index]["messages"][0]["content"]
        assert "required context" in task.history[index]["messages"][-1]["content"]
    assert parent["answer"] == "Parent answer"
    assert "Parent retrieval" not in str(program.retrieve.signature)
    assert budget.used_calls() == 10
    assert proposer.log[0]["role_calls"]["parent"] == 2
    assert proposer.log[0]["role_calls"]["prosecution_survival"] == 2
    assert proposer.log[0]["role_calls"]["millian_survival"] == 2


def test_no_corroborated_counterexample_keeps_parent():
    proposer, _, _, _, _, events, batch = make_proposer(
        parent=(1, 0, 1, 1), prosecution=(1, 1), millian=(1, 1),
    )
    assert proposer({"answer": "Parent"}, batch, ["answer"]) == {}  # Returns {} when no corroboration
    assert events[0]["received_view_survived"]
    assert events[0]["prosecution"]["rationale"]["reason"] == "no_corroborated_counterexample"
    assert not proposer.archive


def test_gain_counts_each_joint_question_once_even_when_it_has_two_roles():
    proposer, _, _, _, _, events, batch = make_proposer(
        parent=(0, 0.6, 0.6, 1), prosecution=(0.8, 0.6, 0.6), millian=(0.7, 0.7, 0.65),
    )
    proposer.archive["q0"] = 0.5
    assert proposer({"answer": "Parent"}, batch, ["answer"]) == {"answer": "Millian"}
    assert events[0]["prosecution"]["total_gain"] == pytest.approx(0.8)
    assert events[0]["millian"]["total_gain"] == pytest.approx(0.85)


class CostedDummyLM(DummyLM):
    """Offline provider-cost fixture, retaining real DSPy calls and histories."""

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        self.history[-1]["cost"] = 0.01
        return result


def test_role_costs_include_distinct_defender_without_counting_existing_calls():
    proposer, _, reflection, _, budget, events, batch = make_proposer()
    defender = CostedDummyLM([
        {"rebuttals": []},
        {"rebuttals": [{"refutation_index": 0, "stance": "contest", "argument": "protect",
                        "protected_ids": [1, 2]}]},
    ])
    with dspy.context(lm=defender):
        proposer.defense(current_instruction="Before round", refutations=[], examples_with_feedback="")
    proposer.defender_lm = defender
    budget.lms.append(defender)
    proposer({"answer": "Parent"}, batch, ["answer"])
    record = events[0]
    assert record["role_costs"]["defender"] == pytest.approx(0.01)
    assert record["cost_consumed"] == pytest.approx(0.01)
    assert budget.used_usd() == pytest.approx(0.02)
    assert record["calls_consumed"] == budget.used_calls() - 1
    assert len(reflection.history) == 3
