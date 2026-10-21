"""Offline positive-control diagnostics; scripted LMs are not efficacy evidence."""
import importlib
import importlib.util
import json
from typing import Any

import pytest
import dspy
from dspy.utils import DummyLM


def scripted_factory(arm, condition, *, abstain=False):
    pc = diagnostic()
    train, hidden = pc.examples()
    defective = condition == "defective"
    initial = [json.loads(ex.question)["record"]["city"] if defective else ex.answer for ex in train]
    refs = [{"claim": "Always-city contradicts requested color in examples 0 and 1",
             "counterexample_ids": [0, 1], "severity": 0.9,
             "proposed_fix": "Return requested_field and preserve city requests"}] if defective and not abstain else []
    reflection: list[dict[str, Any]] = [{"refutations": refs}]
    if refs:
        reflection.append({"new_instruction": pc.CLEAN})
        if arm == "M":
            reflection.append({"new_instruction": pc.CLEAN + " Read the field name carefully.",
                               "rejected_refutations": "none"})
    defense = [{"rebuttals": [{"refutation_index": 0, "stance": "concede",
                              "argument": "City requests still work; preserve example 2", "protected_ids": [2]}] if refs else []}]
    survival = [ex.answer for ex in train] * ((2 if arm == "M" else 1) if refs else 0)
    parent_hidden = [json.loads(ex.question)["record"]["city"] if defective else ex.answer for ex in hidden]
    selected_hidden = [ex.answer for ex in hidden] if refs else parent_hidden
    return {"task": DummyLM([{"answer": answer} for answer in initial + survival + parent_hidden + selected_hidden]),
            "critic": DummyLM(reflection, adapter=dspy.JSONAdapter()),
            "reflection": DummyLM(defense, adapter=dspy.JSONAdapter())}


def test_real_proposers_repair_defect_preserve_clean_and_persist_full_trace(tmp_path):
    pc = diagnostic()
    assert hasattr(pc, "run_diagnostic"), "proposer runner is missing"
    result = pc.run_diagnostic(tmp_path / "run", lm_factory=scripted_factory)
    assert result["status"] == "success"
    assert result["scope"] == "proposer-level; no GEPA compile or final-selection claim"
    assert len(result["cells"]) == 4
    for cell in result["cells"]:
        defective = cell["condition"] == "defective"
        assert cell["parent_hidden"]["scores"] == ([0, 0, 1] if defective else [1, 1, 1])
        assert cell["selected_hidden"]["scores"] == [1, 1, 1]
        assert cell["instruction_changed"] is defective
        assert cell["proposal_events"][0]["survived"] is defective
        assert cell["calls"] <= 20
        assert cell["reflective_batch"][0]["Generated Outputs"]["answer"] == ("Lima" if defective else "red")
    calls = [json.loads(line) for line in (tmp_path / "run" / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == result["calls"] <= 80
    assert all(row["full_prompt"] and row["full_response"] for row in calls)
    for row in calls:
        if not row["role"].endswith(":task"):
            assert all(entity not in row["full_prompt"] for entity in ("Dahlia", "Elm", "Fir"))
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert saved == result


def test_runtime_enforces_cell_and_total_attempt_caps_without_retries():
    pc = diagnostic()
    assert hasattr(pc, "DiagnosticRuntime"), "bounded diagnostic runtime is missing"
    runtime = pc.DiagnosticRuntime()
    lm = runtime.wrap(DummyLM([{"answer": "x"}] * 100), "B:defective:task")
    for _ in range(20):
        lm("test")
    with pytest.raises(RuntimeError, match="call cap"):
        lm("blocked")
    assert len(runtime.attempts) == 20
    assert runtime.max_retries == 0
    total = pc.DiagnosticRuntime()
    for i in range(80):
        total.wrap(DummyLM([{"answer": "x"}]), f"cell{i}:task")("test")
    with pytest.raises(RuntimeError, match="call cap"):
        total.wrap(DummyLM([{"answer": "x"}]), "extra:task")("blocked")
    assert len(total.attempts) == 80


def test_soft_cost_crossing_is_rejected_even_on_final_call():
    pc = diagnostic()
    assert hasattr(pc, "DiagnosticRuntime"), "bounded diagnostic runtime is missing"

    class CostLM(DummyLM):
        def __call__(self, *args, **kwargs):
            response = super().__call__(*args, **kwargs)
            self.history[-1]["cost"] = 0.26
            return response

    runtime = pc.DiagnosticRuntime()
    lm = runtime.wrap(CostLM([{"answer": "x"}]), "B:clean:task")
    with pytest.raises(RuntimeError, match="soft cost"):
        lm("test")
    assert len(runtime.attempts) == 1
    with pytest.raises(RuntimeError):
        lm("blocked")
    assert len(runtime.attempts) == 1


def test_blanket_abstention_is_not_reported_as_repair(tmp_path):
    pc = diagnostic()
    result = pc.run_diagnostic(tmp_path / "abstain", lm_factory=lambda arm, condition: scripted_factory(arm, condition, abstain=True))
    assert result["status"] == "success"
    for cell in result["cells"]:
        assert "observation" in cell, "missing diagnostic interpretation"
        assert cell["observation"] == ("defect_observed_no_admitted_repair" if cell["condition"] == "defective"
                                        else "clean_preserved")
        assert not cell["instruction_changed"]


def test_strict_parse_failure_is_persisted_not_retried(tmp_path):
    pc = diagnostic()

    class RawLM(DummyLM):
        def _format_answer_fields(self, field_names_and_values):
            return '{"refutations": [], "refutations": []}'

    def factory(arm, condition):
        lms = scripted_factory(arm, condition)
        lms["critic"] = RawLM([{}], adapter=dspy.JSONAdapter())
        return lms

    result = pc.run_diagnostic(tmp_path / "malformed", lm_factory=factory)
    assert result["status"] == "failed"
    assert result["cells"][0]["status"] == "failed"
    assert result["cells"][0]["proposal_events"] == []
    calls = [json.loads(line) for line in (tmp_path / "malformed" / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 4
    assert calls[-1]["role"].endswith(":critic")
    assert json.loads((tmp_path / "malformed" / "result.json").read_text())["status"] == "failed"


def test_cli_defaults_to_no_calls_and_requires_confirmation(tmp_path, monkeypatch, capsys):
    pc = diagnostic()
    assert hasattr(pc, "main"), "safe CLI is missing"
    called = []
    monkeypatch.setattr(pc, "run_diagnostic", lambda *args, **kwargs: called.append(kwargs) or {"status": "failed"})
    assert pc.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run"
    assert plan["max_provider_calls"] == 80
    assert plan["max_retries"] == 0
    assert plan["soft_cost_usd"] == 0.25
    assert plan["cost_is_hard_cap"] is False
    assert not called
    assert pc.main(["--confirm", "--output", str(tmp_path / "live")]) == 1
    assert len(called) == 1


def test_live_factory_uses_existing_provider_configuration_and_tiny_limits(monkeypatch):
    pc = diagnostic()
    assert hasattr(pc, "live_factory"), "live factory is missing"
    configs = []

    def build(config, seed):
        configs.append((config, seed))
        return {role: DummyLM([]) for role in ("task", "reflection", "critic")}

    monkeypatch.setattr(pc, "create_lms", build)
    lms = pc.live_factory("openrouter/qwen/qwen3.5-9b")("B", "defective")
    config, seed = configs[0]
    assert config.task_model == config.reflection_model == "openrouter/qwen/qwen3.5-9b"
    assert config.provider_allow_fallbacks is False
    assert not config.enable_reasoning
    assert seed == 0
    for role, lm in lms.items():
        assert lm.kwargs["max_tokens"] == (128 if role == "task" else 1024)
        assert lm.kwargs["temperature"] == 0
        assert lm.kwargs["n"] == 1
        assert not lm.cache
        assert lm.num_retries == 0


def test_failed_rewrite_is_evaluated_and_not_selected(tmp_path):
    pc = diagnostic()
    train, hidden = pc.examples()

    def factory(arm, condition):
        lms = scripted_factory(arm, condition)
        if condition == "defective":
            initial = [json.loads(ex.question)["record"]["city"] for ex in train]
            held = [json.loads(ex.question)["record"]["city"] for ex in hidden]
            answers = initial + initial * (2 if arm == "M" else 1) + held * 2
            lms["task"] = DummyLM([{"answer": answer} for answer in answers])
        return lms

    result = pc.run_diagnostic(tmp_path / "reject", lm_factory=factory)
    assert result["status"] == "success"
    assert "settings" in result, "effective settings must travel with the result"
    assert result["settings"]["max_provider_calls"] == 80
    assert result["settings"]["cost_is_hard_cap"] is False
    assert result["settings"]["data_split"] == {"reflection": 3, "hidden": 3}
    for cell in result["cells"]:
        if cell["condition"] == "defective":
            assert not cell["instruction_changed"]
            assert cell["selected_instruction"] == pc.DEFECTIVE
            event = cell["proposal_events"][0]
            assert not event["survived"]
            evidence = event["evidence"] if cell["arm"] == "B" else event["prosecution"]["evidence"]
            assert any(row["rewrite_score"] == row["parent_score"] == 0 for row in evidence)


MODULE = "experiments.theory_driven.positive_control"


def diagnostic():
    assert importlib.util.find_spec(MODULE) is not None, "positive-control module is missing"
    return importlib.import_module(MODULE)


def test_synthetic_records_are_disjoint_and_gold_is_not_an_input():
    pc = diagnostic()
    train, hidden = pc.examples()
    assert len(train) == len(hidden) == 3
    entities = []
    for split in (train, hidden):
        entities.append({json.loads(ex.question)["entity"] for ex in split})
        for ex in split:
            assert set(ex.inputs()) == {"question"}
            record = json.loads(ex.question)
            assert record["source"].startswith("synthetic")
            assert "answer" not in record
            assert ex.answer == record["record"][record["requested_field"]]
            assert pc.exact_metric(ex, dspy.Prediction(answer=ex.answer)).score == 1
            assert pc.exact_metric(ex, dspy.Prediction(answer=ex.answer + ".")).score == 0
            assert pc.exact_metric(ex, dspy.Prediction(answer="")).score == 0
    assert entities[0].isdisjoint(entities[1])
    assert {json.loads(ex.question)["requested_field"] for ex in train} == {"color", "city"}
    for instruction in (pc.CLEAN, pc.DEFECTIVE):
        program = pc.RecordQA(instruction)
        assert program.answer.signature.instructions == instruction
        assert set(program.answer.signature.input_fields) == {"question"}
