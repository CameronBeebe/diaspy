"""Variant identity and resumability depend on effective scientific config."""
import json
import pytest
from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.harness import ExperimentHarness


def test_variant_parsing_and_validation():
    from experiments.theory_driven.config.variants import parse_arm_variant
    assert parse_arm_variant("A") == ("A", None)
    assert parse_arm_variant("B@majority") == ("B", "majority")
    assert parse_arm_variant("M@half_no_regress") == ("M", "half_no_regress")
    for value in ("A@majority", "B@bad", "B@", "B@strict@veto", "../B"):
        with pytest.raises(ValueError):
            parse_arm_variant(value)


def test_resume_configuration_hash_covers_provider_policy_and_budget(tmp_path):
    from experiments.theory_driven.config.resume import fingerprint, completed_result
    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    original = fingerprint(config, stage, "hotpotqa", "B@majority", 100, 0)
    config.model_config.provider_allow_fallbacks = True
    assert fingerprint(config, stage, "hotpotqa", "B@majority", 100, 0) != original
    config.model_config.provider_allow_fallbacks = False
    assert fingerprint(config, stage, "hotpotqa", "B@half_no_regress", 100, 0) != original
    assert fingerprint(config, stage, "hotpotqa", "B@majority", 50, 0) != original
    next(a for a in config.arms if a.name == "B").kwargs["max_tokens"] = 123
    assert fingerprint(config, stage, "hotpotqa", "B@majority", 100, 0) != original
    row = {"benchmark": "hotpotqa", "arm": "B@majority", "budget": 100, "seed": 0,
           "config_hash": original, "status": "error", "test_mean": None, "test_size": 50,
           "test_scores": []}
    (tmp_path / "incomplete.json").write_text(json.dumps(row))
    assert completed_result(tmp_path, "hotpotqa", "B@majority", 100, 0, original, 50) is None


def test_variants_write_distinct_result_names(tmp_path):
    harness = ExperimentHarness(ExperimentConfig(results_dir=str(tmp_path)))
    for variant in ("B@majority", "B@half_no_regress"):
        harness._save_results({"benchmark": "hotpotqa", "arm": variant, "budget": 100,
                               "seed": 0, "timestamp": "same-time"})
    assert len(list(tmp_path.glob("hotpotqa_B@*.json"))) == 2


def test_resume_skips_only_completed_matching_configuration(tmp_path, monkeypatch):
    from experiments.theory_driven import harness as module
    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    stage.arms = ["A", "B@majority"]
    canaries = []
    monkeypatch.setattr(module, "run_stage_canary", lambda *args: canaries.append(1) or {
        "passed": True, "total_cost": 0.0, "artifacts": {}})
    harness = ExperimentHarness(config)
    calls = []
    def run(benchmark, arm, budget, seed, **kwargs):
        calls.append(arm)
        return {"benchmark": benchmark, "arm": arm, "budget": budget, "seed": seed,
                "status": "success", "test_mean": 0.5, "test_scores": [0.5]*stage.test_size,
                "test_size": stage.test_size, "budget_stats": {"total_cost": 0.01},
                "timestamp": str(len(calls)), "config_hash": kwargs["config_hash"]}
    monkeypatch.setattr(harness, "run_single", run)
    first = harness.run_stage(stage)
    # Simulate normal run_single persistence at the seam.
    for index, result in enumerate(first):
        (tmp_path / f"run{index}.json").write_text(json.dumps(result))
    again = harness.run_stage(stage, resume_dir=tmp_path)
    assert len(again) == 2 and calls == ["A", "B@majority"]
    assert len(canaries) == 2
    config.train_size += 1
    harness.run_stage(stage, resume_dir=tmp_path)
    assert calls == ["A", "B@majority", "A", "B@majority"]


def _completed_row(stage, *, arm="A", cost=0.8, config_hash="matching"):
    return {"benchmark": stage.benchmarks[0], "arm": arm, "budget": stage.budgets[0],
            "seed": 0, "status": "success", "test_mean": 0.5,
            "test_scores": [0.5] * stage.test_size, "test_size": stage.test_size,
            "budget_stats": {"total_cost": cost}, "timestamp": "saved",
            "config_hash": config_hash}


@pytest.mark.parametrize("dimension", ["arm", "model", "benchmark", "budget", "seed"])
def test_resume_reserves_later_completed_cost_before_any_missing_run(
        tmp_path, monkeypatch, capsys, dimension):
    from experiments.theory_driven import harness as module
    from experiments.theory_driven.config.resume import fingerprint
    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    stage.arms = ["A"]
    if dimension == "arm":
        stage.arms.append("B@majority")
    elif dimension == "model":
        stage.models.append("offline/second-model")
    elif dimension == "benchmark":
        stage.benchmarks.append("hover")
    elif dimension == "budget":
        stage.budgets.append(200)
    else:
        stage.n_seeds = 2
    # Save the last combination, leaving the first combination missing.
    config.model_config.task_model = stage.models[-1]
    config.model_config.reflection_model = stage.models[-1]
    row = _completed_row(stage, arm=stage.arms[-1])
    row.update(benchmark=stage.benchmarks[-1], budget=stage.budgets[-1], seed=stage.n_seeds - 1)
    row["config_hash"] = fingerprint(config, stage, row["benchmark"], row["arm"],
                                     row["budget"], row["seed"])
    (tmp_path / "saved.json").write_text(json.dumps(row))
    canaries = []
    monkeypatch.setattr(module, "run_stage_canary", lambda *args: canaries.append(1) or {
        "passed": True, "total_cost": 0.01, "artifacts": {}})
    harness = ExperimentHarness(config)
    calls = []

    def run(benchmark, arm, budget, seed, **kwargs):
        calls.append((benchmark, arm, budget, seed, config.model_config.task_model))
        assert kwargs["max_cost"] == pytest.approx(0.19)
        result = _completed_row(stage, arm=arm, cost=0.1, config_hash=kwargs["config_hash"])
        result.update(benchmark=benchmark, budget=budget, seed=seed)
        return result

    monkeypatch.setattr(harness, "run_single", run)
    results = harness.run_stage(stage, resume_dir=tmp_path)
    assert calls == [(stage.benchmarks[0], "A", stage.budgets[0], 0, stage.models[0])]
    assert len(canaries) == 1
    assert len(results) == 2
    assert results[1]["resumed_from"] == str((tmp_path / "saved.json").resolve())
    assert sum(r["budget_stats"]["total_cost"] for r in results) + 0.01 == pytest.approx(0.91)
    assert "Resume: skipping" in capsys.readouterr().out


@pytest.mark.parametrize("saved_cost,new_cost,arms", [
    (0.8, 0.5, ["A", "B@majority"]),  # Review repro: total would be $1.31.
    (0.8, 0.5, ["A", "M@majority", "B@majority"]),
    (1.0, None, ["B@majority"]),  # All resumed, but canary exceeds the cap.
    (None, 1.0, ["A"]),  # Last fresh run must also enforce the aggregate cap.
])
def test_stage_rejects_aggregate_cost_over_cap(tmp_path, monkeypatch, saved_cost, new_cost, arms):
    from experiments.theory_driven import harness as module
    from experiments.theory_driven.config.resume import fingerprint
    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    stage.arms = arms
    config.model_config.task_model = config.model_config.reflection_model = stage.models[0]
    if saved_cost is not None:
        row = _completed_row(stage, arm="B@majority", cost=saved_cost,
                             config_hash=fingerprint(config, stage, "hotpotqa", "B@majority", 100, 0))
        (tmp_path / "saved.json").write_text(json.dumps(row))
    monkeypatch.setattr(module, "run_stage_canary", lambda *args: {
        "passed": True, "total_cost": 0.01, "artifacts": {}})
    harness = ExperimentHarness(config)
    calls = []

    def run(benchmark, arm, budget, seed, **kwargs):
        calls.append(arm)
        return _completed_row(stage, arm=arm, cost=new_cost, config_hash=kwargs["config_hash"])

    monkeypatch.setattr(harness, "run_single", run)
    with pytest.raises(RuntimeError, match="Stage dollar cap"):
        harness.run_stage(stage, resume_dir=tmp_path)
    assert calls == ([] if new_cost is None else ["A"])


@pytest.mark.parametrize("invalid", [
    {"test_scores": None}, {"test_scores": 1}, {"test_scores": "xx"},
    {"test_scores": {"a": 0.5, "b": 0.5}}, {"test_scores": [0.5]},
    *({"test_scores": [0.5, value]} for value in
      (None, "0.5", True, {}, float("nan"), float("inf"), -float("inf"))),
    *({"test_mean": value} for value in
      (None, "0.5", True, float("nan"), float("inf"), -float("inf"))),
    *({"budget_stats": value} for value in (None, [], {}, "invalid")),
    *({"budget_stats": {"total_cost": value}} for value in
      (None, "0.8", True, -0.1, float("nan"), float("inf"), -float("inf"))),
])
def test_completed_result_skips_malformed_records(tmp_path, invalid):
    from experiments.theory_driven.config.resume import completed_result
    stage = ExperimentConfig().get_stage_config(0)
    stage.test_size = 2
    row = _completed_row(stage)
    (tmp_path / "z-malformed.json").write_text(json.dumps({**row, **invalid}))
    args = (tmp_path, "hotpotqa", "A", 100, 0, "matching", 2)
    assert completed_result(*args) is None
    # A malformed newer record must not hide an older completed match.
    (tmp_path / "a-valid.json").write_text(json.dumps(row))
    resumed = completed_result(*args)
    assert resumed is not None
    assert resumed["resumed_from"] == str((tmp_path / "a-valid.json").resolve())


def test_completed_result_requires_recorded_cost(tmp_path):
    from experiments.theory_driven.config.resume import completed_result
    stage = ExperimentConfig().get_stage_config(0)
    row = _completed_row(stage)
    del row["budget_stats"]
    (tmp_path / "no-cost.json").write_text(json.dumps(row))
    assert completed_result(tmp_path, "hotpotqa", "A", 100, 0, "matching", stage.test_size) is None


@pytest.mark.parametrize("package", ["dspy", "gepa"])
def test_resume_fingerprint_tracks_installed_dependency_versions(monkeypatch, package):
    from importlib import metadata
    from experiments.theory_driven.config.resume import fingerprint
    config = ExperimentConfig()
    stage = config.get_stage_config(0)
    original = fingerprint(config, stage, "hotpotqa", "A", 100, 0)
    installed_version = metadata.version
    monkeypatch.setattr(metadata, "version", lambda name:
                        installed_version(name) + ".changed" if name == package else installed_version(name))
    assert fingerprint(config, stage, "hotpotqa", "A", 100, 0) != original
