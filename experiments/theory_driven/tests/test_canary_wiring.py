"""Offline stage-canary integration tests; all completions use real DummyLMs."""
import importlib
import json
from pathlib import Path

from dspy.utils import DummyLM

from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.config.canary import CanarySuite


def test_disabled_canary_does_not_create_lms_or_artifacts(tmp_path):
    config = ExperimentConfig(results_dir=str(tmp_path / "results"))
    stage = config.get_stage_config(0)
    stage.run_canary = False

    def factory(config, seed):
        raise AssertionError("disabled canary must not create LMs")

    module = importlib.import_module("experiments.theory_driven.canary_runner")
    result = module.run_stage_canary(config, stage, factory)
    assert result["passed"] is True
    assert result["status"] == "skipped"
    assert result["total_cost"] == 0
    assert result["artifacts"] == {}
    assert not Path(config.results_dir).exists()


def test_first_success_saves_labelled_baseline_with_canary_model(tmp_path):
    from experiments.theory_driven.canary_runner import run_stage_canary

    config = ExperimentConfig(results_dir=str(tmp_path))
    original_model = config.model_config.task_model
    stage = config.get_stage_config(0)
    stage.canary_model = "openrouter/mock/canary:free"
    seen = []
    lm = DummyLM([{"answer": "fixed"}] * len(CanarySuite.DEFAULT_CANARIES))

    def factory(model_config, seed):
        seen.append((model_config, seed))
        return {"task": lm}

    result = run_stage_canary(config, stage, factory)
    assert result["passed"] is True
    assert result["status"] == "baseline"
    assert "baseline" in result["message"].lower()
    assert seen[0][0].task_model == stage.canary_model
    assert seen[0][0].reflection_model == stage.canary_model
    assert seen[0][1] == 42
    assert config.model_config.task_model == original_model
    assert len(lm.history) == len(CanarySuite.DEFAULT_CANARIES)
    assert result["total_cost"] == 0.0
    for artifact in result["artifacts"].values():
        assert not Path(artifact).is_absolute()
        assert (tmp_path / artifact).is_file()
    saved = json.loads((tmp_path / result["artifacts"]["results"]).read_text())
    assert len(saved["results"]) == len(CanarySuite.DEFAULT_CANARIES)
    baseline = json.loads((tmp_path / result["artifacts"]["reference"]).read_text())
    assert baseline["model"] == stage.canary_model
    assert len(baseline["prompts"]) == len(CanarySuite.DEFAULT_CANARIES)
    summary = json.loads((tmp_path / result["artifacts"]["summary"]).read_text())
    assert summary == result


def test_provider_error_blocks_despite_drift_override_and_is_persisted(tmp_path, monkeypatch, capsys):
    import dspy
    from experiments.theory_driven.canary_runner import run_stage_canary

    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    stage.allow_canary_drift = True
    calls = []

    class FailingPredict:
        def __init__(self, signature):
            pass

        def __call__(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("400 invalid reasoning; Authorization: Bearer secret-test-key")

    monkeypatch.setattr(dspy, "Predict", FailingPredict)
    result = run_stage_canary(config, stage, lambda config, seed: {"task": DummyLM([])})
    assert result["passed"] is False
    assert result["status"] == "error"
    assert "400 invalid reasoning" in result["message"]
    assert len(calls) == 1
    saved = json.loads((tmp_path / result["artifacts"]["results"]).read_text())
    assert len(saved["results"]) == 1
    assert saved["results"][0]["error"]
    assert not list(tmp_path.rglob("canary_reference_*.json"))
    for path in tmp_path.rglob("*.json"):
        assert "secret-test-key" not in path.read_text()
    assert "secret-test-key" not in capsys.readouterr().out


def test_drift_gate_and_override_preserve_original_baseline(tmp_path):
    from experiments.theory_driven.canary_runner import run_stage_canary

    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)

    def factory(answer):
        return lambda config, seed: {"task": DummyLM(
            [{"answer": answer}] * len(CanarySuite.DEFAULT_CANARIES)
        )}

    baseline = run_stage_canary(config, stage, factory("original"))
    reference = tmp_path / baseline["artifacts"]["reference"]
    original = reference.read_text()
    matched = run_stage_canary(config, stage, factory("original"))
    assert matched["passed"] is True
    assert matched["status"] == "passed"
    blocked = run_stage_canary(config, stage, factory("totally different"))
    assert blocked["passed"] is False
    assert blocked["status"] == "drift"
    assert blocked["issues"]
    stage.allow_canary_drift = True
    allowed = run_stage_canary(config, stage, factory("totally different"))
    assert allowed["passed"] is True
    assert allowed["status"] == "drift_allowed"
    assert reference.read_text() == original
    assert blocked["artifacts"]["results"] != allowed["artifacts"]["results"]


def test_suite_applies_deterministic_generation_settings(tmp_path):
    lm = DummyLM([{"answer": "fixed"}] * len(CanarySuite.DEFAULT_CANARIES))
    lm.kwargs["temperature"] = 0.7
    suite = CanarySuite(tmp_path)
    suite.run_canaries(lm, "dummy", temperature=0.0, seed=17)
    assert len(lm.history) == len(CanarySuite.DEFAULT_CANARIES)
    for call in lm.history:
        assert call["kwargs"]["temperature"] == 0.0
        assert call["kwargs"]["seed"] == 17
        assert call["kwargs"]["cache"] is False
    assert lm.kwargs["temperature"] == 0.7


def test_cost_tracks_successful_calls_before_failure(tmp_path, monkeypatch):
    import dspy
    import pytest
    from experiments.theory_driven.canary_runner import run_stage_canary

    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    lm = DummyLM([{"answer": "fixed"}])
    predict = dspy.Predict

    class PartiallyFailingPredict:
        def __init__(self, signature):
            self.predict = predict(signature)

        def __call__(self, **kwargs):
            if lm.history:
                raise RuntimeError("400 provider failure")
            response = self.predict(**kwargs)
            lm.history[-1]["cost"] = 0.012
            return response

    monkeypatch.setattr(dspy, "Predict", PartiallyFailingPredict)
    result = run_stage_canary(config, stage, lambda config, seed: {"task": lm})
    assert result["passed"] is False
    assert result["total_cost"] == pytest.approx(0.012)


def test_canary_success_artifacts_redact_credentials(tmp_path, monkeypatch):
    from experiments.theory_driven.canary_runner import run_stage_canary
    monkeypatch.setenv("TEST_API_KEY", "synthetic-canary-credential")
    config = ExperimentConfig(results_dir=str(tmp_path))
    result = run_stage_canary(config, config.get_stage_config(0), lambda *args: {
        "task": DummyLM([{"answer": "synthetic-canary-credential"}] * 8),
    })
    assert result["passed"]
    for path in tmp_path.rglob("*.json"):
        assert "synthetic-canary-credential" not in path.read_text()


def test_historical_canary_reference_is_redacted_before_drift_excerpts(tmp_path, monkeypatch):
    import hashlib
    from experiments.theory_driven.canary_runner import run_stage_canary

    credential = "synthetic-historical-credential-" + "x" * 60
    monkeypatch.setenv("TEST_API_KEY", credential)
    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    suite = CanarySuite(tmp_path / "canaries" / "references")
    reference_path = suite.get_reference_file(stage.canary_model or stage.models[0])
    raw_output = "legacy " + credential
    raw_hash = hashlib.sha256(raw_output.encode()).hexdigest()
    historical = {
        "model": stage.canary_model or stage.models[0],
        "created_at": "historical timestamp",
        "metadata": [{credential: {"notes": [credential]}}],
        "prompts": {
            prompt.id: {"prompt": prompt.prompt, "output": raw_output, "output_hash": raw_hash}
            for prompt in CanarySuite.DEFAULT_CANARIES
        },
    }
    reference_path.write_text(json.dumps(historical))

    for allow_drift in (False, True):
        stage.allow_canary_drift = allow_drift
        result = run_stage_canary(config, stage, lambda *args: {
            "task": DummyLM([{"answer": "different " + credential}] * len(CanarySuite.DEFAULT_CANARIES)),
        })
        assert result["passed"] is allow_drift
        assert result["status"] == ("drift_allowed" if allow_drift else "drift")
        assert "synthetic-historical" not in json.dumps(result)
        for path in tmp_path.rglob("*.json"):
            assert "synthetic-historical" not in path.read_text()
        saved_reference = json.loads(reference_path.read_text())
        assert saved_reference["created_at"] == historical["created_at"]
        assert saved_reference["metadata"] == [{"[REDACTED]": {"notes": ["[REDACTED]"]}}]
        assert all(item["output_hash"] == raw_hash for item in saved_reference["prompts"].values())
        saved_results = json.loads((tmp_path / result["artifacts"]["results"]).read_text())
        assert all(item["output_hash"] == hashlib.sha256(("different " + credential).encode()).hexdigest()
                   for item in saved_results["results"])
        assert json.loads((tmp_path / result["artifacts"]["summary"]).read_text()) == result


def test_guarded_canary_persists_failed_attempts(tmp_path):
    from experiments.theory_driven.canary_runner import run_stage_canary
    from experiments.theory_driven.provider_runtime import ProviderRuntime

    class Failure(Exception):
        status_code = 400

    class FailingLM(DummyLM):
        def __call__(self, prompt=None, *, messages=None, **kwargs):
            raise Failure("reasoning expected object")

    runtime = ProviderRuntime()
    lm = runtime.wrap(FailingLM([]), "canary")
    config = ExperimentConfig(results_dir=str(tmp_path))
    result = run_stage_canary(config, config.get_stage_config(0), lambda *args: {"task": lm})
    assert not result["passed"]
    rows = [json.loads(line) for line in (tmp_path / result["artifacts"]["lm_calls"]).read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status_code"] == 400
    assert rows[0]["role"] == "canary"


def test_factory_failure_is_a_persisted_blocking_gate(tmp_path, monkeypatch):
    from experiments.theory_driven.canary_runner import run_stage_canary

    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    stage.allow_canary_drift = True
    monkeypatch.setenv("TEST_API_KEY", "synthetic-credential")

    def factory(config, seed):
        raise ValueError("401 invalid credential synthetic-credential")

    result = run_stage_canary(config, stage, factory)
    assert result["passed"] is False
    assert result["status"] == "error"
    assert result["total_cost"] == 0.0
    assert "401 invalid credential" in result["message"]
    assert "synthetic-credential" not in json.dumps(result)
    assert (tmp_path / result["artifacts"]["summary"]).is_file()
    assert not list(tmp_path.rglob("canary_reference_*.json"))
