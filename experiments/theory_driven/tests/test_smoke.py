"""Stage-zero runs through the real runner, harness, GEPA and evaluation."""
from pathlib import Path
import logging

import dspy

from dspy.utils import DummyLM
import pytest

from experiments.theory_driven.config import ExperimentConfig


def test_harness_imports():
    from experiments.theory_driven.harness import ExperimentHarness
    assert callable(ExperimentHarness)


def test_runner_has_offline_dependency_injection():
    from experiments.theory_driven import run_staged
    assert callable(getattr(run_staged, "run_stage0", None))


def scripted_lms():
    task = DummyLM([{"answer": "deliberately incorrect"}] * 300)
    # Stock DSPy GEPA uses JSONAdapter for its new_instruction field.
    reflection = DummyLM([{"new_instruction": "Answer concisely using the retrieved evidence."}] * 100,
                         adapter=dspy.JSONAdapter())
    refutations = [{"claim": "Answer unsupported by evidence", "counterexample_ids": [0],
                    "severity": 1.0, "proposed_fix": "Read the supporting paragraphs"}]
    critic = DummyLM([item for _ in range(100) for item in (
        {"refutations": refutations},
        {"new_instruction": "Use supporting paragraphs; output only the answer."},
    )])
    return {"task": task, "reflection": reflection, "critic": critic}


@pytest.fixture
def tiny_run(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    from experiments.theory_driven.benchmarks import HotpotQABenchmark
    from experiments.theory_driven.harness import ExperimentHarness
    from experiments.theory_driven.run_staged import run_stage0

    config = ExperimentConfig(train_size=1, val_size=1, test_size=1,
                              results_dir=str(tmp_path))
    for arm in config.arms:
        arm.kwargs.update(reflection_minibatch_size=1)
        if arm.name == "B":
            arm.kwargs["max_attempts"] = 1
    fixture = Path(__file__).parent / "fixtures" / "hotpotqa.json"
    runs = []

    def factory(model_config, seed):
        lms = scripted_lms()
        runs.append(lms)
        return lms

    harness = ExperimentHarness(config, lm_factory=factory,
                                benchmark_factory=lambda name: HotpotQABenchmark(name, fixture_path=fixture))
    stage = config.get_stage_config(0)
    stage.budgets = [8]
    results = run_stage0(config, harness=harness, stage=stage)
    assert "reflection failed" not in caplog.text
    assert "Traceback" not in caplog.text
    return results, runs


def test_both_arms_complete_and_account_for_test_evaluation(tiny_run):
    results, runs = tiny_run
    assert [r["arm"] for r in results] == ["A", "B"]
    for result, lms in zip(results, runs, strict=True):
        assert result["status"] == "success", result
        assert result["test_size"] == len(result["test_scores"]) == 1
        assert result["test_mean"] == 0.0
        stats = result["budget_stats"]
        assert stats["total_units"] == sum(len(lm.history) for lm in lms.values())
        assert stats["test_calls"] == 1
        assert stats["optimization_calls"] >= result["budget"]
        assert stats["optimization_overshoot"] == stats["optimization_calls"] - result["budget"]
        assert stats["total_units"] == stats["optimization_calls"] + stats["test_calls"]
        assert stats["stop_triggered"]
        assert stats["total_cost"] == 0.0


def test_arm_patterns_differ_on_same_seed_and_data(tiny_run):
    results, runs = tiny_run
    a, b = runs
    assert len(a["reflection"].history) > 0
    assert len(a["critic"].history) == 0
    assert len(b["reflection"].history) == 0
    assert len(b["critic"].history) > 0
    assert any("refutations" in str(entry["messages"]) for entry in b["critic"].history)
    assert results[1]["arm_stats"]["proposer_log"]
    assert all(not item["survived"] for item in results[1]["arm_stats"]["proposer_log"])
