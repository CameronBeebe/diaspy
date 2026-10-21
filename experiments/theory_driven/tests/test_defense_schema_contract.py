"""Defense prompt and guarded schema contract; offline wire replay only."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import dspy
import httpx
import pytest
from dspy.utils import DummyLM
from dspy.utils.exceptions import AdapterParseError

from experiments.theory_driven.arms.millian import Defense, MillianArm, Rebuttal
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.provider_runtime import ProviderRuntime
from experiments.theory_driven.tests.test_millian import AnswerProgram, metric
from experiments.theory_driven.tests.test_strict_json_adapter import REFUTATIONS, raw_lm
from experiments.theory_driven.arms.strict_json import StrictJSONAdapter


def test_defense_raw_prompt_explains_every_required_rebuttal_field():
    messages = StrictJSONAdapter().format(
        Defense, [], dict(current_instruction="Parent", refutations=[], examples_with_feedback="")
    )
    system = messages[0]["content"]
    instruction = Defense.instructions
    assert "Every rebuttal must include refutation_index, stance, argument, and protected_ids" in instruction
    assert "protected_ids: []" in instruction
    assert "Keep each argument concise" in instruction
    assert " ".join(instruction.split()) in " ".join(system.split())
    schema = Rebuttal.model_json_schema()
    assert set(schema["required"]) == {"refutation_index", "stance", "argument", "protected_ids"}
    for name, field in schema["properties"].items():
        assert field["description"] in system, name
        assert "default" not in field
    assert "[]" in schema["properties"]["protected_ids"]["description"]
    assert "concise" in schema["properties"]["argument"]["description"]
    assert all(field.is_required() for field in Rebuttal.model_fields.values())


RECORDED = json.loads((Path(__file__).parent / "fixtures" / "defense_missing_protected_ids.json").read_text())


def explicit_empty_response():
    """Synthetic valid counterpart, not a repair of a live response."""
    data = json.loads(RECORDED["raw_response"])
    data["rebuttals"][1]["protected_ids"] = []
    return json.dumps(data)


def test_recorded_missing_field_aborts_real_compile_without_regeneration():
    raw = RECORDED["raw_response"]
    assert hashlib.sha256(raw.encode()).hexdigest() == RECORDED["raw_response_sha256"]
    reflection = raw_lm(json.dumps({"refutations": REFUTATIONS}), '{"refutations": []}')
    defender = raw_lm(raw, explicit_empty_response())
    task = DummyLM([{"answer": "0"}] * 20)
    budget = UnifiedBudget(100, 1, task, reflection, defender)
    events = []
    arm = MillianArm("strict", "offline", budget, max_metric_calls=10,
                     reflection_minibatch_size=1, max_attempts=1, defender_lm=defender)
    gold = dspy.Example(question="q", answer="1").with_inputs("question")
    with pytest.raises(AdapterParseError, match=r"1\.protected_ids") as caught:
        arm.compile(AnswerProgram(), metric, [gold], [gold], task_lm=task,
                    reflection_lm=reflection, event_sink=events.append)
    assert caught.value.lm_response == raw
    assert len(defender.history) == len(reflection.history) == 1
    assert len(task.history) == 2
    assert not events
    assert arm.proposer is not None
    assert arm.proposer.structured_adapter.failure is caught.value
    assert not arm.proposer.log and not arm.proposer.archive


def test_explicit_empty_protected_ids_preserves_both_rebuttals():
    raw = explicit_empty_response()
    lm = raw_lm(raw)
    with dspy.context(lm=lm, adapter=StrictJSONAdapter()):
        result = dspy.Predict(Defense)(current_instruction="Parent", refutations=[], examples_with_feedback="")
    assert [item.model_dump() for item in result.rebuttals] == json.loads(raw)["rebuttals"]
    assert result.rebuttals[1].protected_ids == []
    assert len(lm.history) == 1


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("wrapper", ["bare", "guarded", "copy", "deepcopy"])
def test_defense_actual_http_request_mode_and_prompt_offline(monkeypatch, wrapper, asynchronous):
    requests = []

    def send(client, request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json={
            "id": "offline-defense", "object": "chat.completion", "created": 0,
            "model": "qwen/qwen3.5-9b", "provider": "DeepInfra",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": explicit_empty_response()},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": .00000125},
        })

    async def asend(client, request, **kwargs):
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    lm = dspy.LM("openrouter/qwen/qwen3.5-9b", engine="litellm", api_key="offline-fixture-not-a-key",
                 temperature=0, max_tokens=2048, n=1, seed=0, timeout=30, cache=False, num_retries=0,
                 extra_body={"reasoning": {"enabled": False},
                             "provider": {"order": ["DeepInfra"], "allow_fallbacks": False,
                                          "require_parameters": True}})
    raw = lm
    runtime = ProviderRuntime(max_retries=0)
    if wrapper != "bare":
        lm = runtime.wrap(lm, "defender")
        if wrapper == "copy":
            lm = lm.copy(num_retries=99, max_retries=99)
        elif wrapper == "deepcopy":
            lm = copy.deepcopy(lm)
        assert lm.runtime is runtime
        assert lm.history is raw.history
        assert lm.num_retries == 0
        assert "max_retries" not in lm.kwargs
    if asynchronous:
        from pytest_socket import disable_socket, enable_socket
        # asyncio needs local self-pipes; Internet sockets remain disabled.
        enable_socket()
        disable_socket(allow_unix_socket=True)
    inputs = dict(current_instruction="Parent", refutations=[], examples_with_feedback="")
    with dspy.context(lm=lm, adapter=StrictJSONAdapter()):
        predict = dspy.Predict(Defense)
        result = asyncio.run(predict.acall(**inputs)) if asynchronous else predict(**inputs)
    assert result.rebuttals[1].protected_ids == []
    assert len(requests) == 1
    assert len(raw.history) == 1
    if wrapper != "bare":
        assert len(runtime.attempts) == 1
        trace = runtime.attempts[0]
        assert trace["role"] == "defender"
        assert trace["attempt"] == 1 and trace["error"] is None
        assert trace["usage"]["total_tokens"] == 10
        assert trace["cost"] == pytest.approx(.00000125)
        assert trace["provider"] == "DeepInfra"
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    body = json.loads(request.content)
    assert body["response_format"]["type"] == "json_schema"
    wire_schema = body["response_format"]["json_schema"]
    assert wire_schema["strict"] is True
    rebuttal_schema = wire_schema["schema"]["$defs"]["Rebuttal"]
    assert set(rebuttal_schema["required"]) == set(Rebuttal.model_fields)
    assert "protected_ids" in rebuttal_schema["required"]
    assert rebuttal_schema["additionalProperties"] is False
    for name, field in Rebuttal.model_json_schema()["properties"].items():
        assert rebuttal_schema["properties"][name]["description"] == field["description"]
    assert body["provider"] == {"order": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True}
    assert body["reasoning"] == {"enabled": False}
    assert (body["max_tokens"], body["temperature"], body["n"], body["seed"]) == (2048, 0, 1, 0)
    assert request.extensions["timeout"]["read"] == 30
    system = body["messages"][0]["content"]
    assert "Every rebuttal must include refutation_index, stance, argument, and protected_ids" in system
    assert "protected_ids: []" in system
    assert "Keep each argument concise" in system
    for field in Rebuttal.model_json_schema()["properties"].values():
        assert field["description"] in system
