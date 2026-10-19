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
    def __init__(self, status, message="reasoning: expected object, received boolean"):
        super().__init__(message)
        self.status_code = status


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
    runtime = module.ProviderRuntime()
    first = FailingEngine(StatusError(status))
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


def test_rate_limit_only_bounded_exponential_retry_records_every_attempt():
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(max_retries=2, backoff_seconds=0.5, sleep=delays.append)
    engine = ScriptedEngine([StatusError(429), StatusError(429)])
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False, num_retries=9), "task")
    assert "ok" in lm("hello")[0]
    assert engine.calls == 3
    assert delays == [0.5, 1.0]
    assert [r["status_code"] for r in runtime.attempts] == [429, 429, None]
    assert [r["attempt"] for r in runtime.attempts] == [1, 2, 3]
    assert not runtime.failed


def test_exhausted_rate_limit_is_latched_not_retried_by_outer_optimizer():
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(max_retries=2, sleep=delays.append)
    engine = FailingEngine(StatusError(429, "free pool exhausted"))
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    for _ in range(4):
        with pytest.raises(module.ProviderFailure, match="free pool exhausted"):
            lm("hello")
    assert engine.calls == 3
    assert len(delays) == 2
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


@pytest.mark.parametrize("options", [{"max_retries": -1}, {"max_retries": 100},
                                     {"max_retries": True}, {"backoff_seconds": -1},
                                     {"text_limit": 0}])
def test_runtime_rejects_unbounded_or_invalid_settings(options):
    module = runtime_module()
    with pytest.raises(ValueError):
        module.ProviderRuntime(**options)
