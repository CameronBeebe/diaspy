"""Stage-level failure propagation and artifact wiring; no provider requests."""
import json
from pathlib import Path

import pytest
from dspy.utils import DummyLM

from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.harness import ExperimentHarness
from experiments.theory_driven.tests.test_harness import FixtureBenchmark
from experiments.theory_driven.run_staged import run_stage0


def test_stage_stops_after_first_error_and_keeps_result(monkeypatch, tmp_path):
    config = ExperimentConfig(train_size=1, val_size=1, test_size=1, results_dir=str(tmp_path))
    created = []

    def factory(config, seed):
        created.append(seed)
        return {role: DummyLM([]) for role in ("task", "reflection", "critic")}

    harness = ExperimentHarness(config, lm_factory=factory, benchmark_factory=lambda name: FixtureBenchmark())

    class FailingArm:
        def compile(self, **kwargs):
            raise RuntimeError("HTTP 400 reasoning: expected object")
        def get_stats(self):
            return {}

    monkeypatch.setattr(harness, "create_arm", lambda *args, **kwargs: FailingArm())
    stage = config.get_stage_config(0)
    stage.run_canary = False
    with pytest.raises(RuntimeError, match="expected object"):
        run_stage0(config, harness=harness, stage=stage)
    assert len(created) == 1
    files = list(tmp_path.glob("hotpotqa_*.json"))
    assert len(files) == 1
    result = json.loads(files[0].read_text())
    assert result["status"] == "error"
    artifacts = result["artifacts"]
    for relative in artifacts.values():
        assert not Path(relative).is_absolute()
        assert (tmp_path / relative).is_file()
    events = [json.loads(line) for line in (tmp_path / artifacts["events"]).read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["initialization", "finalization"]
    assert "expected object" in (tmp_path / artifacts["lm_calls"]).read_text()


def test_stage_canary_failure_prevents_arm_creation(monkeypatch, tmp_path):
    from experiments.theory_driven import harness as harness_module
    calls = []
    def canary(config, stage, lm_factory):
        calls.append("canary")
        return {"passed": False, "message": "canary HTTP 400", "artifacts": {}, "total_cost": 0.0}
    monkeypatch.setattr(harness_module, "run_stage_canary", canary, raising=False)
    harness = ExperimentHarness(ExperimentConfig(results_dir=str(tmp_path)))
    monkeypatch.setattr(harness, "run_single", lambda *args, **kwargs: calls.append("arm"))
    with pytest.raises(RuntimeError, match="canary HTTP 400"):
        harness.run_stage(harness.config.get_stage_config(0))
    assert calls == ["canary"]


def test_real_gepa_fatal_provider_error_is_one_attempt(monkeypatch, tmp_path):
    class BadRequest(Exception):
        status_code = 400

    class FailingLM(DummyLM):
        def __init__(self):
            super().__init__([])
            self.attempt_count = 0

        def __call__(self, prompt=None, *, messages=None, **kwargs):
            self.attempt_count += 1
            raise BadRequest("reasoning: expected object")

    task = FailingLM()
    config = ExperimentConfig(train_size=3, val_size=3, test_size=1, results_dir=str(tmp_path))
    harness = ExperimentHarness(config, lm_factory=lambda *args: {
        "task": task, "reflection": DummyLM([]), "critic": DummyLM([]),
    }, benchmark_factory=lambda name: FixtureBenchmark())
    result = harness.run_single("hotpotqa", "A", 50, 0)
    assert result["status"] == "error"
    assert result["test_mean"] is None
    assert "expected object" in result["message"]
    assert task.attempt_count == 1
    assert result["budget_stats"]["provider_failed"]
    rows = [json.loads(line) for line in (tmp_path / result["artifacts"]["lm_calls"]).read_text().splitlines()]
    attempts = [row for row in rows if row["role"] == "task"]
    assert len(attempts) == 1
    assert attempts[0]["status_code"] == 400


def test_setup_failure_still_finalizes_artifacts(tmp_path):
    config = ExperimentConfig(results_dir=str(tmp_path))
    def factory(*args):
        raise ValueError("missing provider credential")
    harness = ExperimentHarness(config, lm_factory=factory)
    result = harness.run_single("hotpotqa", "A", 50, 0)
    assert result["status"] == "error"
    assert "missing provider credential" in result["message"]
    assert result["budget_stats"]["total_units"] == 0
    events = [json.loads(line) for line in (tmp_path / result["artifacts"]["events"]).read_text().splitlines()]
    assert events[-1]["event_type"] == "finalization"


def test_confirm_cli_returns_nonzero_after_failed_run(monkeypatch):
    from experiments.theory_driven import run_staged
    monkeypatch.setattr(run_staged.PreFlightGate, "run_all_checks", lambda *args: True)
    def failed(config):
        raise RuntimeError("saved failure result; HTTP 401")
    monkeypatch.setattr(run_staged, "run_stage0", failed)
    assert run_staged.main(["--stage", "0", "--confirm"]) == 1
