"""History accounting uses actual offline DSPy calls, never manual counters."""

import dspy
from dspy.utils import DummyLM
import pytest

from experiments.theory_driven.config import budget_tracker


def test_counts_unique_lm_histories_by_role_and_includes_later_evaluation():
    assert hasattr(budget_tracker, "UnifiedBudget")
    task = DummyLM([{"answer": "task"}, {"answer": "test"}])
    reflection = DummyLM([{"answer": "reflection"}])
    critic = DummyLM([{"answer": "critic"}])
    budget = budget_tracker.UnifiedBudget(
        4, 1.0, task, reflection, critic, task,
        roles={"task": task, "reflection": reflection, "critic": critic},
    )
    predict = dspy.Predict("question -> answer")
    for lm in (task, reflection, critic):
        with dspy.context(lm=lm):
            predict(question="count me")
    task.history[0]["cost"] = 0.1
    reflection.history[0]["cost"] = None
    critic.history[0]["cost"] = 0.2
    assert budget.used_calls() == 3
    assert budget.used_usd() == pytest.approx(0.3)
    assert not budget(None)
    assert budget.get_stats()["stop_triggered"] is False
    with dspy.context(lm=task):
        predict(question="held out evaluation")
    stats = budget.get_stats()
    assert budget.used_calls() == sum(len(lm.history) for lm in (task, reflection, critic)) == 4
    assert stats["role_calls"] == {"task": 2, "reflection": 1, "critic": 1}
    assert stats["total_units"] == 4
    assert budget(None)
    assert budget.get_stats()["stop_triggered"] is True


def test_cost_cap_and_overshoot_are_reported_from_provider_history():
    lm = DummyLM([{"answer": "one"}, {"answer": "two"}])
    budget = budget_tracker.UnifiedBudget(1, 0.25, lm, roles={"task": lm, "critic": lm})
    predict = dspy.Predict("question -> answer")
    for _ in range(2):
        with dspy.context(lm=lm):
            predict(question="one in-flight iteration")
    lm.history[0]["cost"] = 0.2
    lm.history[1]["cost"] = 0.3
    stats = budget.get_stats()
    assert stats["overshoot_units"] == 1
    assert stats["overshoot_usd"] == pytest.approx(0.25)
    assert sum(stats["role_calls"].values()) == 2
    assert sum(stats["role_costs"].values()) == pytest.approx(0.5)
    cost_only = budget_tracker.UnifiedBudget(100, 0.5, lm)
    assert cost_only(None)
