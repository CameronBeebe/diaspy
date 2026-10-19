"""Real proposer-pattern comparison is exercised by test_smoke.py."""
from experiments.theory_driven.config import ExperimentConfig


def test_stage0_selects_compared_arm_variants():
    stage = ExperimentConfig().get_stage_config(0)
    assert stage.arms == ["A", "B@half_no_regress", "M@half_no_regress", "B@majority"]
