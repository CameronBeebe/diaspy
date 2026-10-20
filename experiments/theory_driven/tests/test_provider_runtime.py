"""Real DSPy calls, offline engines only; provider errors must survive swallowing."""
import importlib
import importlib.util

import dspy
import pytest


def runtime_module():
    spec = importlib.util.find_spec("experiments.theory_driven.provider_runtime")
    assert spec is not None, "Provider runtime must intercept actual LM calls"
    return importlib.import_module("experiments.theory_driven.provider_runtime")


class StatusError(Exception):
    def __init__(self, status, message="reasoning: expected object, received boolean",
                 *, response=None, headers=None):
        super().__init__(message)
        self.status_code = status
        self.response = response
        self.headers = headers


class FailingEngine:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise self.error


@pytest.mark.parametrize("status", [400, 401, 404])
def test_fatal_error_latches_across_roles_and_swallowed_calls(status):
    module = runtime_module()
    runtime = module.ProviderRuntime(sleep=lambda _: pytest.fail("fatal status retried"))
    first = FailingEngine(StatusError(status, headers={"Retry-After": "100"}))
    second = FailingEngine(StatusError(status))
    task = runtime.wrap(dspy.LM("openai/offline", engine=first, cache=False), "task")
    critic = runtime.wrap(dspy.LM("openai/offline", engine=second, cache=False), "critic")
    for lm in (task, task, critic, task.copy(temperature=0.7)):
        with pytest.raises(module.ProviderFailure, match="expected object"):
            lm("hello")
    assert first.calls == 1
    assert second.calls == 0
    assert task.history == []
    assert runtime.should_stop(None)
    with pytest.raises(module.ProviderFailure, match="expected object"):
        runtime.raise_if_failed()
    assert len(runtime.attempts) == 1
    assert runtime.attempts[0]["status_code"] == status
    assert runtime.attempts[0]["role"] == "task"


def test_success_traces_are_compact_sanitized_and_keep_history(tmp_path):
    import json
    from dspy.utils.dummies import DummyLM

    module = runtime_module()
    secret = "opaque-provider-credential"
    runtime = module.ProviderRuntime(secrets=[secret], text_limit=120)
    raw = DummyLM([{"answer": f"Bearer other-secret {secret} " + "a" * 200}])
    lm = runtime.wrap(raw, "task")
    result = lm(messages=[{"role": "user", "content": f"api_key={secret} " + "x" * 200}])
    assert result
    assert lm.history is raw.history
    assert len(raw.history) == 1
    assert len(runtime.attempts) == 1
    trace = runtime.attempts[0]
    assert trace["role"] == "task"
    assert trace["model"] == "dummy"
    assert trace["error"] is None
    assert "usage" in trace and "cost" in trace
    assert len(trace["prompt"]) <= 120
    assert len(trace["response"]) <= 120
    artifact = tmp_path / "calls.jsonl"
    runtime.write_jsonl(artifact)
    text = artifact.read_text()
    assert secret not in text and "other-secret" not in text
    assert "kwargs" not in text
    assert json.loads(text) == trace


@pytest.mark.parametrize("location", ["provider", "hidden_provider", "hidden_provider_name", "native"])
def test_serving_provider_is_read_from_real_history_response_and_sanitized(location, tmp_path):
    import json
    from dataclasses import replace
    from litellm import ModelResponse

    module = runtime_module()
    provider = "DeepInfra opaque-provider-credential"
    response = ModelResponse(
        model="offline", choices=[{"message": {"role": "assistant", "content": "ok"}}],
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        **({"provider": provider} if location == "provider" else {}),
    )
    response._hidden_params = {"response_cost": 0.01, "custom_llm_provider": "openrouter"}
    if location.startswith("hidden_"):
        response._hidden_params[location.removeprefix("hidden_")] = provider

    class ResponseLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return response

    class ProviderEngine(ScriptedEngine):
        def complete(self, request):
            return replace(super().complete(request), provider_data={"provider": provider})

    raw = (dspy.LM("openai/offline", engine=ProviderEngine([]), cache=False)
           if location == "native" else ResponseLM("openrouter/offline", cache=False))
    runtime = module.ProviderRuntime(secrets=["opaque-provider-credential"])
    lm = runtime.wrap(raw, "task")
    assert "ok" in lm("hello")[0]
    assert runtime.attempts[-1]["provider"] == "DeepInfra [REDACTED]"
    assert lm.history is raw.history
    assert len(raw.history) == 1
    if location != "native":
        assert raw.history[-1]["response"] is response
        assert runtime.attempts[-1]["cost"] == 0.01
        assert runtime.attempts[-1]["usage"]["total_tokens"] == 3
    artifact = tmp_path / "calls.jsonl"
    runtime.write_jsonl(artifact)
    assert json.loads(artifact.read_text())["provider"] == "DeepInfra [REDACTED]"


@pytest.mark.parametrize("response", [
    {}, {"provider": {"order": ["DeepInfra"]}}, {"provider": ""},
    {"_hidden_params": {"custom_llm_provider": "openrouter"}},
    {"model": "deepinfra/offline"},
])
def test_serving_provider_does_not_infer_routing_or_model(response):
    from dspy.utils.dummies import DummyLM

    module = runtime_module()

    class ResponseLM(DummyLM):
        def update_history(self, entry):
            super().update_history({**entry, "response": response})

    raw = ResponseLM([{"answer": "ok"}])
    raw.kwargs["provider"] = {"order": ["DeepInfra"]}
    runtime = module.ProviderRuntime()
    lm = runtime.wrap(raw, "task")
    lm("hello")
    assert runtime.attempts[-1].get("provider") is None
    assert raw.history[-1]["response"] == response


def test_failure_text_and_exception_are_sanitized(monkeypatch):
    module = runtime_module()
    monkeypatch.setenv("OPENROUTER_API_KEY", "opaque-env-key")
    runtime = module.ProviderRuntime(text_limit=300)
    engine = FailingEngine(StatusError(401, 'bad key opaque-env-key sk-or-v1-supersecret api_key="opaque-inline" Authorization: Bearer bearer-secret'))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "critic")
    with pytest.raises(module.ProviderFailure) as failure:
        lm("prompt opaque-env-key")
    text = str(failure.value) + str(runtime.attempts)
    for secret in ("opaque-env-key", "sk-or-v1-supersecret", "opaque-inline", "bearer-secret"):
        assert secret not in text
    assert "bad key" in text


class ScriptedEngine:
    def __init__(self, errors):
        from dspy.utils.dummies import DummyLM
        self.errors = iter(errors)
        self.calls = 0
        self.success = DummyLM([{"answer": "ok"}])._engine_spec

    def complete(self, request):
        self.calls += 1
        error = next(self.errors, None)
        if error is not None:
            raise error
        return self.success.complete(request)


@pytest.mark.parametrize("backoff,expected", [
    (2.0, [2.25, 4.25, 8.25, 16.25, 32.25]),
    (30.0, [30.25, 60.0, 60.0, 60.0, 60.0]),
    (60.0, [60.0, 60.0, 60.0, 60.0, 60.0]),
])
def test_default_five_retries_recover_with_capped_injected_jitter(backoff, expected):
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(
        backoff_seconds=backoff, sleep=delays.append, random=lambda: 0.25,
    )
    engine = ScriptedEngine([StatusError(429) for _ in range(5)])
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    assert "ok" in lm("hello")[0]
    assert engine.calls == 6
    assert delays == expected
    assert [row["attempt"] for row in runtime.attempts] == [1, 2, 3, 4, 5, 6]
    assert [row["status_code"] for row in runtime.attempts] == [429] * 5 + [None]
    assert not runtime.failed


def test_rate_limit_only_bounded_exponential_retry_records_every_attempt():
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(max_retries=2, backoff_seconds=0.5, sleep=delays.append,
                                     random=lambda: 0.0)
    engine = ScriptedEngine([StatusError(429), StatusError(429)])
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False, num_retries=9), "task")
    assert "ok" in lm("hello")[0]
    assert engine.calls == 3
    assert delays == [0.5, 1.0]
    assert [r["status_code"] for r in runtime.attempts] == [429, 429, None]
    assert [r["attempt"] for r in runtime.attempts] == [1, 2, 3]
    assert not runtime.failed


@pytest.mark.parametrize("header,expected", [
    ("7", 7.0), ("1", 2.25), ("120", 120.0), ("2.5", 2.5),
    ("Thu, 01 Jan 1970 00:16:50 GMT", 10.0),
    ("Thu, 01 Jan 1970 00:16:39 GMT", 2.25),
    ("invalid", 2.25), ("-1", 2.25), ("nan", 2.25), ("inf", 2.25),
    (None, 2.25),
])
@pytest.mark.parametrize("location", ["response", "exception"])
def test_retry_after_is_minimum_delay_from_chained_error(header, expected, location):
    from types import SimpleNamespace

    module = runtime_module()
    error = StatusError(429)
    headers = {"rEtRy-AfTeR": header}
    if location == "response":
        error.response = SimpleNamespace(headers=headers)
    else:
        error.headers = headers
    wrapped = RuntimeError("sdk wrapper")
    wrapped.__cause__ = error
    delays = []
    runtime = module.ProviderRuntime(
        sleep=delays.append, random=lambda: 0.25, clock=lambda: 1000.0,
    )
    engine = ScriptedEngine([wrapped])
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    assert "ok" in lm("hello")[0]
    assert delays == [expected]
    assert engine.calls == 2
    assert not runtime.failed


def test_exhausted_rate_limit_is_latched_not_retried_by_outer_optimizer():
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(sleep=delays.append, random=lambda: 0.5)
    engine = FailingEngine(StatusError(429, "free pool exhausted"))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    critic_engine = ScriptedEngine([])
    critic = runtime.wrap(dspy.LM("openai/offline", engine=critic_engine, cache=False), "critic")
    for target in (lm, lm, lm.copy(), critic):
        with pytest.raises(module.ProviderFailure, match="free pool exhausted"):
            target("hello")
    assert engine.calls == 6
    assert critic_engine.calls == 0
    assert delays == [2.5, 4.5, 8.5, 16.5, 32.5]
    assert len(runtime.attempts) == 6
    assert runtime.failed


@pytest.mark.parametrize("error_class,status", [
    ("LMInvalidRequestError", 400), ("LMAuthError", 401),
    ("LMUnsupportedModelError", 404), ("LMRateLimitError", 429),
])
def test_dspy_error_classes_without_status(error_class, status):
    import dspy.utils.exceptions as errors
    module = runtime_module()
    runtime = module.ProviderRuntime(max_retries=0)
    engine = FailingEngine(getattr(errors, error_class)("provider failure"))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    with pytest.raises(module.ProviderFailure):
        lm("hello")
    assert runtime.attempts[0]["status_code"] == status
    assert engine.calls == 1


@pytest.mark.parametrize("status", [403, 408, 500, 503])
def test_other_http_statuses_are_not_retried_or_latched(status):
    module = runtime_module()
    runtime = module.ProviderRuntime(sleep=lambda _: pytest.fail("unexpected retry"))
    engine = FailingEngine(StatusError(status, headers={"Retry-After": "10"}))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    with pytest.raises(Exception):
        lm("hello")
    assert engine.calls == 1
    assert not runtime.failed


def test_provider_metadata_does_not_reuse_history_when_history_is_disabled():
    from dspy.utils.dummies import DummyLM

    module = runtime_module()
    raw = DummyLM([{"answer": "ok"}])
    raw.history.append({"response": {"provider": "stale"}, "cost": 10.0})
    runtime = module.ProviderRuntime()
    lm = runtime.wrap(raw, "task")
    with dspy.context(disable_history=True):
        lm("hello")
    assert runtime.attempts[-1].get("provider") is None
    assert runtime.attempts[-1]["cost"] is None
    assert len(raw.history) == 1


def test_non_provider_programming_errors_are_not_latched_or_retried():
    module = runtime_module()
    runtime = module.ProviderRuntime(sleep=lambda _: pytest.fail("unexpected retry"))
    engine = FailingEngine(ValueError("local bug"))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    with pytest.raises(Exception, match="local bug") as failure:
        lm("hello")
    assert not isinstance(failure.value, module.ProviderFailure)
    assert not runtime.failed
    assert engine.calls == 1


def test_real_predictor_and_optimizer_copy_share_guard_and_history():
    import copy
    from dspy.utils.dummies import DummyLM
    module = runtime_module()
    runtime = module.ProviderRuntime()
    raw = DummyLM([{"answer": "ok"}, {"answer": "again"}])
    lm = runtime.wrap(raw, "task")
    program = dspy.Predict("question -> answer")
    with dspy.context(lm=lm):
        assert program(question="hello").answer == "ok"
    cloned = copy.deepcopy(lm)
    assert cloned.runtime is runtime
    with dspy.context(lm=cloned):
        assert program(question="hello").answer == "again"
    assert len(lm.history) == 2
    assert len(runtime.attempts) == 2


def test_parallel_evaluator_swallowing_still_only_makes_one_provider_attempt():
    module = runtime_module()
    runtime = module.ProviderRuntime()
    engine = FailingEngine(StatusError(400))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    data = [dspy.Example(question="hi", answer="ok").with_inputs("question") for _ in range(6)]
    evaluator = dspy.Evaluate(devset=data, metric=lambda *args: 0,
                              num_threads=3, max_errors=1000, display_progress=False)
    with dspy.context(lm=lm):
        evaluator(dspy.Predict("question -> answer"))
    assert engine.calls == 1
    assert runtime.should_stop(None)
    assert len(runtime.attempts) == 1


def test_async_call_cannot_bypass_guard():
    import asyncio
    from pytest_socket import disable_socket, enable_socket
    # Reset the fixture's wrapper, then immediately permit only local self-pipes.
    enable_socket()
    disable_socket(allow_unix_socket=True)
    module = runtime_module()
    runtime = module.ProviderRuntime()
    engine = FailingEngine(StatusError(401))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    for _ in range(2):
        with pytest.raises(module.ProviderFailure):
            asyncio.run(lm.acall("hello"))
    assert engine.calls == 1


def test_call_and_copy_cannot_restore_internal_retry_options():
    module = runtime_module()
    runtime = module.ProviderRuntime(max_retries=0)
    engine = FailingEngine(StatusError(429, "stop"))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    cloned = lm.copy(num_retries=99, max_retries=99)
    with pytest.raises(module.ProviderFailure, match="stop"):
        cloned("hello", num_retries=99, max_retries=99, retry_strategy="constant_retry")
    assert engine.calls == 1
    assert cloned.num_retries == 0
    assert "max_retries" not in cloned.kwargs


@pytest.mark.parametrize("options", [{"max_retries": -1}, {"max_retries": True},
                                     {"backoff_seconds": -1},
                                     {"backoff_seconds": 60.1}, {"backoff_seconds": float("nan")},
                                     {"backoff_seconds": float("inf")}, {"text_limit": 0},
                                     {"timeout_seconds": 0}, {"timeout_seconds": -1},
                                     {"max_retry_after_seconds": 0}, {"max_retry_after_seconds": -1}])
def test_runtime_rejects_unbounded_or_invalid_settings(options):
    module = runtime_module()
    with pytest.raises(ValueError):
        module.ProviderRuntime(**options)


def test_timeout_is_injected_for_real_providers():
    """Timeout should be injected only for real HTTP providers, not custom engines."""
    module = runtime_module()
    runtime = module.ProviderRuntime(timeout_seconds=90.0)
    
    # Real provider (string engine) should get timeout
    lm = runtime.wrap(dspy.LM("openai/gpt-4o-mini", cache=False), "task")
    assert lm.kwargs["timeout"] == 90.0
    
    # Custom engine should NOT get timeout
    lm_custom = runtime.wrap(dspy.LM("openai/offline", engine=ScriptedEngine([]), cache=False), "task")
    assert "timeout" not in lm_custom.kwargs


def test_retry_after_is_capped_at_max_retry_after_seconds():
    from types import SimpleNamespace

    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(
        sleep=delays.append, random=lambda: 0.0, clock=lambda: 1000.0,
        max_retry_after_seconds=60.0,
    )
    error = StatusError(429)
    error.response = SimpleNamespace(headers={"Retry-After": "3600"})
    engine = ScriptedEngine([error])
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    assert "ok" in lm("hello")[0]
    assert delays == [60.0]
    assert engine.calls == 2


def test_exponential_backoff_not_capped_only_retry_after_capped():
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(
        backoff_seconds=30.0, sleep=delays.append, random=lambda: 0.0,
        max_retry_after_seconds=100.0,
    )
    engine = ScriptedEngine([StatusError(429) for _ in range(5)])
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    assert "ok" in lm("hello")[0]
    assert delays == [30.0, 60.0, 120.0, 240.0, 480.0]
    assert engine.calls == 6


def test_full_prompt_log_includes_unsanitized_full_prompt():
    from dspy.utils.dummies import DummyLM

    module = runtime_module()
    runtime = module.ProviderRuntime(text_limit=20, full_prompt_log=True)
    raw = DummyLM([{"answer": "response"}])
    lm = runtime.wrap(raw, "task")
    long_prompt = "x" * 1000
    lm(long_prompt)
    assert len(runtime.attempts[-1]["prompt"]) == 20
    assert len(runtime.attempts[-1]["full_prompt"]) == 1000
    assert runtime.attempts[-1]["full_prompt"] == long_prompt


def test_full_prompt_log_false_does_not_include_full_prompt():
    from dspy.utils.dummies import DummyLM

    module = runtime_module()
    runtime = module.ProviderRuntime(text_limit=20, full_prompt_log=False)
    raw = DummyLM([{"answer": "response"}])
    lm = runtime.wrap(raw, "task")
    long_prompt = "x" * 1000
    lm(long_prompt)
    assert len(runtime.attempts[-1]["prompt"]) == 20
    assert "full_prompt" not in runtime.attempts[-1]
