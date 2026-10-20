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


def scripted_lms(millian=False):
    task = DummyLM([{"answer": "deliberately incorrect"}] * 300)
    # Stock DSPy GEPA uses JSONAdapter for its new_instruction field.
    reflection = DummyLM([{"new_instruction": "Answer concisely using the retrieved evidence."}] * 100,
                         adapter=dspy.JSONAdapter())
    refutations = [{"claim": "Answer unsupported by evidence", "counterexample_ids": [0],
                    "severity": 1.0, "proposed_fix": "Read the supporting paragraphs"}]
    round_answers = [
        {"refutations": refutations},
        {"new_instruction": "Use supporting paragraphs; output only the answer."},
    ]
    if millian:
        round_answers = [
            {"refutations": refutations},
            {"rebuttals": [{"refutation_index": 0, "stance": "concede",
                            "argument": "Address the observed error", "protected_ids": []}]},
            {"new_instruction": "Use supporting paragraphs; output only the answer."},
            {"new_instruction": "Use evidence while preserving correct answers.",
             "rejected_refutations": "None; objection conceded."},
        ]
    critic = DummyLM(round_answers * 100, adapter=dspy.JSONAdapter())
    return {"task": task, "reflection": reflection, "critic": critic}


@pytest.fixture
def tiny_run(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    from experiments.theory_driven import harness as harness_module
    monkeypatch.setattr(harness_module, "run_stage_canary", lambda *args: {
        "passed": True, "message": "mocked offline canary", "artifacts": {}, "total_cost": 0.0,
    }, raising=False)
    from experiments.theory_driven.benchmarks import HotpotQABenchmark
    from experiments.theory_driven.harness import ExperimentHarness
    from experiments.theory_driven.run_staged import run_stage0

    config = ExperimentConfig(train_size=1, val_size=1, test_size=1,
                              results_dir=str(tmp_path))
    for arm in config.arms:
        arm.kwargs.update(reflection_minibatch_size=1)
        if arm.name in {"B", "M"}:
            arm.kwargs["max_attempts"] = 1
    fixture = Path(__file__).parent / "fixtures" / "hotpotqa.json"
    runs = []

    def factory(model_config, seed):
        lms = scripted_lms(millian=len(runs) == 2)
        runs.append(lms)
        return lms

    harness = ExperimentHarness(config, lm_factory=factory,
                                benchmark_factory=lambda name: HotpotQABenchmark(name, fixture_path=fixture))
    stage = config.get_stage_config(0)
    stage.arms = ["A", "B@majority", "M@half_no_regress"]
    stage.budgets = [8]
    results = run_stage0(config, harness=harness, stage=stage)
    assert "reflection failed" not in caplog.text
    assert "Traceback" not in caplog.text
    for result in results:
        result["_test_results_dir"] = str(tmp_path)
    return results, runs


def test_all_three_arms_complete_and_account_for_test_evaluation(tiny_run):
    results, runs = tiny_run
    assert [r["arm"] for r in results] == ["A", "B@majority", "M@half_no_regress"]
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
        import json
        root = Path(result["_test_results_dir"])
        call_rows = [json.loads(line) for line in (root / result["artifacts"]["lm_calls"]).read_text().splitlines()]
        assert len(call_rows) == stats["total_units"]
        assert all(row["error"] is None for row in call_rows)
        events = [json.loads(line) for line in (root / result["artifacts"]["events"]).read_text().splitlines()]
        assert events[0]["event_type"] == "initialization"
        assert events[-1]["event_type"] == "finalization"
        proposals = [event for event in events if event["event_type"] == "optimization_step"]
        if result["arm_base"] == "A":
            assert proposals
            for event in proposals:
                record = event["metadata"]["proposal"]
                assert record["proposed_instructions"]
                assert record["accepted"] is False
                assert record["parent_scores"] is not None
            assert stats["critic_lm_calls"] == 0
        if result["arm_base"] == "M":
            assert len(proposals) == len(result["arm_stats"]["proposer_log"])
            assert proposals
            for event in proposals:
                debate = event["metadata"]["debate"]
                assert debate["policy"] == "half_no_regress"
                assert debate["received_view_survived"]
                assert debate["role_calls"]["defender"] == 1
                assert debate["calls_consumed"] == sum(debate["role_calls"].values())
                prosecution = debate["prosecution"]["evidence"]
                millian = debate["millian"]["evidence"]
                assert [(row["question"], row["role"], row["parent_score"]) for row in prosecution] == [
                    (row["question"], row["role"], row["parent_score"]) for row in millian]
        if result["arm_base"] == "B":
            assert result["survival_policy"] == "majority"
            assert len(proposals) == len(result["arm_stats"]["proposer_log"])
            assert all(not event["step"]["decision"]["accepted"] for event in proposals)


def test_arm_patterns_differ_on_same_seed_and_data(tiny_run):
    results, runs = tiny_run
    a, b, m = runs
    assert len(a["reflection"].history) > 0
    assert len(a["critic"].history) == 0
    assert len(b["reflection"].history) == 0
    assert len(b["critic"].history) > 0
    assert any("refutations" in str(entry["messages"]) for entry in b["critic"].history)
    assert results[1]["survival_policy"] == "majority"
    assert results[2]["survival_policy"] == "half_no_regress"
    assert any("rebuttals" in str(entry["messages"]) for entry in m["critic"].history)
    assert not any("rebuttals" in str(entry["messages"]) for entry in b["critic"].history)
    assert results[1]["arm_stats"]["proposer_log"]
    assert all(not item["survived"] for item in results[1]["arm_stats"]["proposer_log"])
