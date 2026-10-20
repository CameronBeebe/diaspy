"""Raw wire-response regressions: no repair, truncation, or implicit regeneration."""

import json
from pathlib import Path
from typing import Any, cast

import dspy
import pytest
from dspy.utils import DummyLM
from dspy.utils.exceptions import AdapterParseError

from experiments.theory_driven.arms.millian import MillianArm, MillianProposer
from experiments.theory_driven.arms.popperian import PopperianArm, PopperianProposer
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.tests.test_millian import AnswerProgram, metric


REFUTATIONS = [
    {"claim": "first objection", "counterexample_ids": [0], "severity": 1, "proposed_fix": "first fix"},
    {"claim": "second objection", "counterexample_ids": [1], "severity": 1, "proposed_fix": "second fix"},
]

RECORDED = json.loads((Path(__file__).parent / "fixtures" / "structured_responses.json").read_text())


class RawReplayLM(DummyLM):
    """Keep malformed bytes intact instead of DummyLM's dict formatting."""

    def _format_answer_fields(self, field_names_and_values):
        return field_names_and_values["raw"]


def raw_lm(*responses):
    return RawReplayLM([{"raw": text} for text in responses])


def make_proposer(critique, defense='{"rebuttals": []}', proposer_type=MillianProposer):
    reflection = raw_lm(critique, '{"new_instruction": "P"}',
                        '{"new_instruction": "M", "rejected_refutations": "none"}')
    defender = raw_lm(defense)
    task = DummyLM([{"answer": str(score)} for score in (0, 0, 1, 1, 1, 1)])
    gold = {f"q{i}": dspy.Example(question=f"q{i}", answer="GOLD_ONLY").with_inputs("question")
            for i in range(2)}
    budget = UnifiedBudget(100, 1, task, reflection, defender)
    events = []
    kwargs: dict[str, Any] = {"defender_lm": defender} if proposer_type is MillianProposer else {}
    proposer = proposer_type(AnswerProgram(), metric, gold, reflection, task, budget,
                             max_attempts=1, min_survival_questions=2,
                             event_sink=events.append, **kwargs)
    batch = {"answer": [{"Inputs": {"question": question}} for question in gold]}
    return proposer, reflection, defender, task, budget, events, batch


@pytest.mark.parametrize("arm_type", [PopperianArm, MillianArm])
@pytest.mark.parametrize("call_cap", [3, 100])
def test_real_compile_latches_recorded_parse_failure_before_gepa_retry(arm_type, call_cap):
    # GEPA retries failed batched reflection per task, then suppresses failures.
    # A valid retry must never turn the recorded malformed response into success.
    reflection = raw_lm(RECORDED["B14"], '{"refutations": []}')
    defender = raw_lm('{"rebuttals": []}')
    task = DummyLM([{"answer": "0"}] * 20)
    budget = UnifiedBudget(call_cap, 1, task, reflection, defender)
    events = []
    kwargs: dict[str, Any] = {"defender_lm": defender} if arm_type is MillianArm else {}
    arm = arm_type("strict", "offline", budget, max_metric_calls=10,
                   reflection_minibatch_size=1, max_attempts=1, **kwargs)
    gold = dspy.Example(question="q", answer="1").with_inputs("question")
    with pytest.raises(AdapterParseError) as caught:
        arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task,
                    reflection_lm=reflection, event_sink=events.append)
    assert caught.value.lm_response == RECORDED["B14"]
    assert len(reflection.history) == 1
    assert not defender.history
    assert arm.proposer is not None
    assert not events and not arm.proposer.log
    assert not arm.proposer.archive
    assert len(task.history) == 2  # seed + parent only, no later iteration
    assert arm.proposer.structured_adapter.failure is caught.value


@pytest.mark.parametrize("arm_type", [PopperianArm, MillianArm])
@pytest.mark.parametrize("nonempty", [False, True])
def test_real_compile_preserves_valid_empty_and_nonempty_paths(arm_type, nonempty, caplog):
    refs = [REFUTATIONS[0]] if nonempty else []
    reflection = raw_lm(json.dumps({"refutations": refs}), '{"new_instruction": "P"}',
                        '{"new_instruction": "M", "rejected_refutations": "none"}')
    defender = raw_lm('{"rebuttals": []}')
    task = DummyLM([{"answer": "0"}] * 2 + [{"answer": "1"}] * 20)
    budget = UnifiedBudget(3, 1, task, reflection, defender)
    events = []
    kwargs: dict[str, Any] = {"defender_lm": defender} if arm_type is MillianArm else {}
    arm = arm_type("strict", "offline", budget, max_metric_calls=10,
                   reflection_minibatch_size=1, max_attempts=1, min_survival_questions=1, **kwargs)
    gold = dspy.Example(question="q", answer="1").with_inputs("question")
    program = AnswerProgram()
    parent = cast(Any, program.answer.signature).instructions
    optimized = arm.compile(program, metric, [gold], [gold], task_lm=task,
                            reflection_lm=reflection, event_sink=events.append)
    assert isinstance(optimized, dspy.Module)
    assert arm.proposer is not None
    assert arm.proposer.structured_adapter.failure is None
    assert len(events) == 1
    assert events == arm.proposer.log
    assert events[0]["refutations"] == refs
    assert events[0]["survived"] is nonempty
    if nonempty:
        assert cast(Any, dict(optimized.named_predictors())["answer"].signature).instructions == "P"
        assert arm.proposer.archive == {"q": 1.0}
    else:
        assert cast(Any, dict(optimized.named_predictors())["answer"].signature).instructions == parent
        assert events[0]["rationale"]["reason"] == "no_corroborated_counterexample"
    expected_calls = (3 if arm_type is MillianArm else 2) if nonempty else 1
    assert len(reflection.history) == expected_calls
    assert len(defender.history) == (1 if arm_type is MillianArm else 0)
    assert "reflection failed" not in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.parametrize("arm_type,role,reflection_calls,defender_calls", [
    (PopperianArm, "rewrite", 2, 0),
    (MillianArm, "defense", 1, 1),
    (MillianArm, "rewrite", 2, 1),
    (MillianArm, "dialectical_rewrite", 3, 1),
])
def test_real_compile_shares_fatal_latch_across_all_structured_roles(
    arm_type, role, reflection_calls, defender_calls,
):
    malformed = '{"new_instruction": "lost"}\n{"new_instruction": "retained"}'
    reflection = raw_lm(
        json.dumps({"refutations": [REFUTATIONS[0]]}),
        malformed if role == "rewrite" else '{"new_instruction": "P"}',
        malformed if role == "dialectical_rewrite" else '{"refutations": []}',
        '{"refutations": []}',
    )
    defender = raw_lm(malformed if role == "defense" else '{"rebuttals": []}', '{"rebuttals": []}')
    task = DummyLM([{"answer": "0"}] * 20)
    budget = UnifiedBudget(100, 1, task, reflection, defender)
    events = []
    kwargs: dict[str, Any] = {"defender_lm": defender} if arm_type is MillianArm else {}
    arm = arm_type("strict", "offline", budget, max_metric_calls=10,
                   reflection_minibatch_size=1, max_attempts=1, **kwargs)
    gold = dspy.Example(question="q", answer="1").with_inputs("question")
    with pytest.raises(AdapterParseError) as caught:
        arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task,
                    reflection_lm=reflection, event_sink=events.append)
    assert caught.value.lm_response == malformed
    assert len(reflection.history) == reflection_calls
    assert len(defender.history) == defender_calls
    assert len(task.history) == 2
    assert arm.proposer is not None
    assert not events and not arm.proposer.log and not arm.proposer.archive


@pytest.mark.parametrize("arm_type", [PopperianArm, MillianArm])
def test_fatal_latch_is_fresh_for_each_compile_on_same_arm(arm_type):
    reflection = raw_lm(RECORDED["B14"], '{"refutations": []}')
    defender = raw_lm('{"rebuttals": []}')
    task = DummyLM([{"answer": "0"}] * 20)
    budget = UnifiedBudget(100, 1, task, reflection, defender)
    kwargs: dict[str, Any] = {"defender_lm": defender} if arm_type is MillianArm else {}
    arm = arm_type("strict", "offline", budget, max_metric_calls=2,
                   reflection_minibatch_size=1, max_attempts=1, **kwargs)
    gold = dspy.Example(question="q", answer="1").with_inputs("question")
    with pytest.raises(AdapterParseError):
        arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task, reflection_lm=reflection)
    failed_proposer = arm.proposer
    events = []
    optimized = arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task,
                            reflection_lm=reflection, event_sink=events.append)
    assert isinstance(optimized, dspy.Module)
    assert arm.proposer is not None and arm.proposer is not failed_proposer
    assert len(reflection.history) == 2
    assert len(events) == 1
    assert events[0]["rationale"]["reason"] == "no_corroborated_counterexample"


def test_fragmented_prosecution_fails_before_defense_or_persistence():
    malformed = "\n".join(json.dumps({"refutations": [ref]}) for ref in REFUTATIONS)
    proposer, reflection, defender, task, budget, events, batch = make_proposer(malformed)
    # Even a permissive ambient adapter must not weaken structured role boundaries.
    with dspy.context(adapter=dspy.JSONAdapter()), pytest.raises(AdapterParseError):
        proposer({"answer": "Parent"}, batch, ["answer"])
    assert len(reflection.history) == budget.used_calls() == 1
    assert not defender.history and not task.history
    assert not proposer.log and not events
    assert not proposer.archive


@pytest.mark.parametrize("malformed", [
    '{"refutations": [], "refutations": []}',
    json.dumps({"refutations": [REFUTATIONS[0]]}).replace(
        '"claim": "first objection"', '"claim": "lost objection", "claim": "first objection"'),
    '{"refutations": [], "unexpected": "must not be dropped"}',
    '[{"refutations": []}]',
    '{"refutations": [], "unexpected": NaN}',
])
def test_ambiguous_json_is_rejected_before_typed_parsing(malformed):
    from experiments.theory_driven.arms.popperian import Critique
    from experiments.theory_driven.arms.strict_json import StrictJSONAdapter

    lm = raw_lm(malformed)
    with dspy.context(lm=lm, adapter=StrictJSONAdapter()), pytest.raises(AdapterParseError):
        dspy.Predict(Critique)(current_instruction="Parent", examples_with_feedback="")
    assert len(lm.history) == 1


@pytest.mark.parametrize("proposer_type", [PopperianProposer, MillianProposer])
@pytest.mark.parametrize("response", [RECORDED["B14"], RECORDED["M6"]])
def test_recorded_fragmented_critique_fails_closed_without_regeneration(proposer_type, response):
    proposer, reflection, defender, task, budget, events, batch = make_proposer(response, proposer_type=proposer_type)
    with pytest.raises(AdapterParseError):
        proposer({"answer": "Parent"}, batch, ["answer"])
    assert len(reflection.history) == budget.used_calls() == 1
    assert not defender.history and not task.history
    assert not events and not proposer.log


def test_defense_cannot_silently_drop_a_rebuttal():
    rebuttals = [{"refutation_index": i, "stance": "contest", "argument": f"defense {i}",
                 "protected_ids": [i]} for i in range(2)]
    malformed = "\n".join(json.dumps({"rebuttals": [item]}) for item in rebuttals)
    proposer, reflection, defender, task, budget, events, batch = make_proposer(
        json.dumps({"refutations": REFUTATIONS}), malformed)
    with dspy.context(adapter=dspy.JSONAdapter()), pytest.raises(AdapterParseError):
        proposer({"answer": "Parent"}, batch, ["answer"])
    assert len(reflection.history) == len(defender.history) == 1
    assert budget.used_calls() == 2
    assert not task.history and not events and not proposer.log


@pytest.mark.parametrize("proposer_type", [PopperianProposer, MillianProposer])
def test_rewrite_rejects_multiple_output_objects(proposer_type):
    proposer, _, defender, task, _, events, batch = make_proposer(
        json.dumps({"refutations": REFUTATIONS}), proposer_type=proposer_type)
    reflection = raw_lm(json.dumps({"refutations": REFUTATIONS}),
                        '{"new_instruction": "lost"}\n{"new_instruction": "retained"}',
                        '{"new_instruction": "M", "rejected_refutations": "none"}')
    proposer.critic_lm = reflection
    task.adapter = dspy.JSONAdapter()
    with dspy.context(adapter=dspy.JSONAdapter()), pytest.raises(AdapterParseError):
        proposer({"answer": "Parent"}, batch, ["answer"])
    assert len(reflection.history) == 2
    assert len(task.history) == 2  # parent evidence only, no partial rewrite rollout
    assert len(defender.history) == (1 if proposer_type is MillianProposer else 0)
    assert not events and not proposer.log


def test_dialectical_rewrite_rejects_multiple_output_objects():
    proposer, _, _, task, _, events, batch = make_proposer(json.dumps({"refutations": REFUTATIONS}))
    reflection = raw_lm(json.dumps({"refutations": REFUTATIONS}), '{"new_instruction": "P"}',
                        '{"new_instruction": "lost", "rejected_refutations": "one"}\n'
                        '{"new_instruction": "retained", "rejected_refutations": "two"}')
    proposer.critic_lm = reflection
    task.adapter = dspy.JSONAdapter()
    with dspy.context(adapter=dspy.JSONAdapter()), pytest.raises(AdapterParseError):
        proposer({"answer": "Parent"}, batch, ["answer"])
    assert len(reflection.history) == 3
    assert len(task.history) == 2
    assert not events and not proposer.log


@pytest.mark.parametrize("proposer_type", [PopperianProposer, MillianProposer])
@pytest.mark.parametrize("response", [json.dumps({"refutations": REFUTATIONS}), RECORDED["B15"]],
                         ids=["two-native-refutations", "recorded-B15"])
def test_valid_refutations_survive_all_consumers_and_task_boundary(proposer_type, response):
    refs = json.loads(response)["refutations"]
    rebuttals = [{"refutation_index": i, "stance": "contest", "argument": f"defense {i}",
                 "protected_ids": []} for i in range(len(refs))]
    proposer, reflection, defender, task, budget, events, batch = make_proposer(
        response, json.dumps({"rebuttals": rebuttals}), proposer_type)
    ambient = dspy.ChatAdapter()
    with dspy.context(adapter=ambient):
        assert proposer({"answer": "Parent"}, batch, ["answer"])
        assert dspy.settings.adapter is ambient
    assert events[0]["refutations"] == refs
    assert proposer.log == events
    for entry in reflection.history[1:] + defender.history:
        user = entry["messages"][-1]["content"]
        field = user.split("[[ ## refutations ## ]]\n", 1)[1].split("\n\n", 1)[0]
        assert json.loads(field) == refs
    if proposer_type is MillianProposer:
        assert events[0]["rebuttals"] == rebuttals
        user = reflection.history[2]["messages"][-1]["content"]
        field = user.split("[[ ## rebuttals ## ]]\n", 1)[1].split("\n\n", 1)[0]
        assert json.loads(field) == rebuttals
        assert events[0]["role_calls"]["prosecutor"] == events[0]["role_calls"]["defender"] == 1
    assert budget.used_calls() == len(task.history) + len(reflection.history) + len(defender.history)
    assert task.history
    for entry in task.history:
        prompt = str(entry["messages"])
        assert "GOLD_ONLY" not in prompt
        assert "Respond with a JSON object" not in prompt
        assert "[[ ## answer ## ]]" in entry["outputs"][0]
    assert all("Respond with a JSON object" in entry["messages"][-1]["content"]
               for entry in reflection.history + defender.history)


@pytest.mark.parametrize("proposer_type", [PopperianProposer, MillianProposer])
def test_valid_empty_refutations_are_not_parse_failures(proposer_type):
    proposer, reflection, defender, task, _, events, batch = make_proposer(
        '{"refutations": []}', proposer_type=proposer_type)
    assert proposer({"answer": "Parent"}, batch, ["answer"]) == {}
    assert events[0]["refutations"] == []
    assert len(reflection.history) == 1
    assert len(defender.history) == (1 if proposer_type is MillianProposer else 0)
    assert not task.history


@pytest.mark.parametrize("value", [
    json.dumps(REFUTATIONS),
    "\n".join(json.dumps([ref]) for ref in REFUTATIONS),
    [{**REFUTATIONS[0], "counterexample_ids": "[0] [1]"}],
    [{**REFUTATIONS[0], "counterexample_ids": ["0"]}],
    [{**REFUTATIONS[0], "severity": "1"}],
    [{**REFUTATIONS[0], "unrecognized_objection": "must not disappear"}],
])
def test_structured_fields_require_native_typed_json_without_dropped_members(value):
    from experiments.theory_driven.arms.popperian import Critique
    from experiments.theory_driven.arms.strict_json import StrictJSONAdapter

    lm = raw_lm(json.dumps({"refutations": value}))
    with dspy.context(lm=lm, adapter=StrictJSONAdapter()), pytest.raises(AdapterParseError):
        dspy.Predict(Critique)(current_instruction="Parent", examples_with_feedback="")
    assert len(lm.history) == 1
