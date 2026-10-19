"""Real proposer-pattern comparison is exercised by test_smoke.py."""
from experiments.theory_driven.config import ExperimentConfig


def test_stage0_selects_only_the_two_compared_arms():
    stage = ExperimentConfig().get_stage_config(0)
    assert stage.arms == ["A", "B"]
