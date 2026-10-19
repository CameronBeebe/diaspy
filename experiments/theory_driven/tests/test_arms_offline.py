"""Harness configuration must reach the real GEPA arm constructors."""
import pytest
from dspy.utils import DummyLM

from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.harness import ExperimentHarness


@pytest.mark.parametrize("name", ["A", "B", "C", "M"])
def test_seed_and_budget_reach_every_gepa_arm(name, tmp_path):
    lm = DummyLM([{"answer": "unused"}])
    budget = UnifiedBudget(17, 1.0, lm)
    config = ExperimentConfig(results_dir=str(tmp_path))
    harness = ExperimentHarness(config)
    arm = harness.create_arm(name, budget, seed=29, budget=17)
    assert getattr(arm, "seed") == 29
    assert getattr(arm, "max_metric_calls") == 17
    assert arm.budget_tracker is budget
