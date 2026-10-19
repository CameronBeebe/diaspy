"""Stage-zero configuration and no-call planning regressions."""
from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.preflight import PreFlightGate
from pathlib import Path


def test_stage0_uses_hotpot_and_fifty_held_out_examples():
    config = ExperimentConfig()
    stage = config.get_stage_config(0)
    assert stage.benchmarks == ["hotpotqa"]
    assert stage.arms == ["A", "B"]
    assert config.test_size == stage.test_size == 50
    assert (config.train_size, config.val_size) == (20, 20)
    assert stage.models == ["openrouter/google/gemma-4-26b-a4b-it:free"]


def test_free_model_plan_has_zero_cost_and_separate_roles():
    config = ExperimentConfig()
    plan = PreFlightGate(Path(__file__).parents[1])._compute_planned_calls(
        config.get_stage_config(0), config
    )
    assert plan["max_cost"] == 0.0
    assert plan["total_runs"] == 2
    assert {"task", "reflection", "critic"} <= plan["roles"].keys()
    assert plan["test_calls"] == 100
    assert plan["canary_calls"] == 8
    assert plan["roles"]["canary"]["calls"] == 8
