"""The unified stopper, not GEPA's metric cap, terminates optimization."""
from typing import Any
import logging

import dspy
from dspy.utils import DummyLM

from experiments.theory_driven.arms import GEPADefaultArm, PopperianArm
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
import pytest


class ObservedBudget(UnifiedBudget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = []

    def __call__(self, state):
        stop = super().__call__(state)
        self.observations.append((self.used_calls(), stop))
        return stop


@pytest.mark.parametrize("arm_class", [GEPADefaultArm, PopperianArm])
def test_history_stopper_stops_before_large_metric_cap(arm_class, caplog):
    caplog.set_level(logging.INFO)
    task = DummyLM([{"answer": "wrong"}] * 100)
    if arm_class is GEPADefaultArm:
        reflection = DummyLM([{"new_instruction": "Read the question carefully."}] * 50,
                             adapter=dspy.JSONAdapter())
        extra: dict[str, Any] = {}
        iteration_bound = 3  # old minibatch + reflection + new minibatch
    else:
        reflection = DummyLM([item for _ in range(50) for item in (
            {"refutations": [{"claim": "wrong", "counterexample_ids": [0], "severity": 1.0, "proposed_fix": "read"}]},
            {"new_instruction": "Read the question carefully."},
        )], adapter=dspy.JSONAdapter())
        extra = {"max_attempts": 1}
        iteration_bound = 5  # old + critique + rewrite + survival + new
    budget = ObservedBudget(4, 1.0, task, reflection)
    arm = arm_class("test", "offline", budget, max_metric_calls=100,
                    reflection_minibatch_size=1, **extra)
    examples = [dspy.Example(question="q", answer="gold").with_inputs("question")]

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return dspy.Prediction(score=0.0, feedback="wrong; gold answer is gold")

    arm.compile(dspy.Predict("question -> answer"), metric, examples, examples,
                task_lm=task, reflection_lm=reflection)
    assert budget.observations[-1][1] is True
    assert budget.used_calls() == budget.observations[-1][0]
    assert 4 <= budget.used_calls() < 4 + iteration_bound
    assert budget.used_calls() < 100
    assert "reflection failed" not in caplog.text
    assert "Traceback" not in caplog.text
