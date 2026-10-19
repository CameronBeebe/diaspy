"""Stage-zero configuration and no-call planning regressions."""
from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.preflight import PreFlightGate
from pathlib import Path


def test_stage0_uses_hotpot_and_fifty_held_out_examples():
    config = ExperimentConfig()
    stage = config.get_stage_config(0)
    assert stage.benchmarks == ["hotpotqa"]
    assert stage.arms == ["A", "B", "M"]
    assert stage.survival_policy == "majority"
    assert stage.min_survival_questions == 3
    assert config.test_size == stage.test_size == 50
    assert (config.train_size, config.val_size) == (20, 20)
    assert stage.models == ["openrouter/qwen/qwen3.5-9b"]
    assert stage.canary_model == stage.models[0]


def test_paid_smoke_plan_has_unknown_cost_and_separate_roles():
    config = ExperimentConfig()
    plan = PreFlightGate(Path(__file__).parents[1])._compute_planned_calls(
        config.get_stage_config(0), config
    )
    assert plan["max_cost"] is None
    assert plan["cost_cap"] == 1.0
    assert plan["total_runs"] == 3
    assert {"task", "reflection", "critic"} <= plan["roles"].keys()
    assert plan["test_calls"] == 150
    assert plan["proposer_calls_per_attempt"]["M"] == 4
    assert plan["roles"]["critic"]["optimization_range"][1] == 100
    assert plan["canary_calls"] == 8
    assert plan["roles"]["canary"]["calls"] == 8
