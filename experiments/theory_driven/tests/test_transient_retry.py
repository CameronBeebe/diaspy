"""Transient transport failures share the bounded 429 retry policy (fake clock, offline)."""
import json

import dspy
from litellm.exceptions import APIConnectionError, Timeout
import pytest
from dspy.utils import DummyLM

from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.harness import ExperimentHarness
from experiments.theory_driven.tests.test_harness import FixtureBenchmark
from experiments.theory_driven.tests.test_provider_runtime import ScriptedEngine, StatusError, runtime_module


def litellm_timeout():
    # What the stage-0 run actually raised: litellm.Timeout carries status 408.
    return Timeout(message="Connection timed out after 120.0 seconds.",
                   model="qwen/qwen3.5-9b", llm_provider="openrouter")


class FakeClock:
    """Time advances on sleep and by `call_seconds` per provider attempt."""

    def __init__(self, call_seconds=0.0):
        self.now = 0.0
        self.sleeps = []
        self.call_seconds = call_seconds

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class SlowScriptedEngine(ScriptedEngine):
    def __init__(self, errors, clock):
        super().__init__(errors)
        self.fake = clock

    def complete(self, request):
        self.fake.now += self.fake.call_seconds
        return super().complete(request)


def make(errors, fake=None, **options):
    module = runtime_module()
    fake = fake or FakeClock()
    runtime = module.ProviderRuntime(sleep=fake.sleep, clock=fake.clock, random=lambda: 0.0, **options)
    engine = SlowScriptedEngine(errors, fake)
    return module, runtime, engine, runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task"), fake


def test_408_timeout_then_success_is_retried():
    _, runtime, engine, lm, fake = make([litellm_timeout()])
    assert "ok" in lm("hello")[0]
    assert engine.calls == 2
    assert fake.sleeps == [2.0]
    assert [row["status_code"] for row in runtime.attempts] == [408, None]
    assert runtime.attempts[0]["retry_kind"] == "transient"
    assert runtime.attempts[0]["transient_error"] == "Timeout"
    assert not runtime.failed
    assert runtime.retry_summary() == {"failed_attempts": 1, "rate_limit_errors": 0, "transient_errors": 1,
                                       "timeouts": 1, "calls_recovered_after_retry": 1}


@pytest.mark.parametrize("error", [ConnectionResetError("Connection reset by peer"),
                                   EOFError("EOF occurred in violation of protocol"),
                                   APIConnectionError(message="Server disconnected",
                                                      llm_provider="openrouter", model="m")])
def test_connection_reset_or_eof_then_success_is_retried(error):
    _, runtime, engine, lm, fake = make([error])
    assert "ok" in lm("hello")[0]
    assert engine.calls == 2
    assert len(fake.sleeps) == 1
    assert runtime.attempts[0]["retry_kind"] == "transient"
    assert runtime.retry_summary()["transient_errors"] == 1
    assert not runtime.failed


def test_wrapped_reset_is_found_in_the_cause_chain():
    try:
        try:
            raise ConnectionResetError("reset")
        except ConnectionResetError as inner:
            raise RuntimeError("engine failed") from inner
    except RuntimeError as outer:
        wrapped = outer
    _, runtime, engine, lm, _ = make([wrapped])
    assert "ok" in lm("hello")[0]
    assert engine.calls == 2


def test_repeated_timeouts_stop_at_the_total_wall_cap_and_latch():
    fake = FakeClock(call_seconds=120.0)  # each attempt hangs for the full read timeout
    module, runtime, engine, lm, fake = make([litellm_timeout() for _ in range(50)], fake,
                                             max_retries=10, max_delay_seconds=120.0,
                                             max_retry_wall_seconds=900.0)
    with pytest.raises(module.ProviderFailure, match="retry wall time exceeded"):
        lm("hello")
    assert runtime.failed
    assert engine.calls < 11
    assert fake.now <= 900.0 + 120.0  # the wall cap bounds sleeps; at most one in-flight attempt beyond it
    assert all(row["retry_kind"] == "transient" for row in runtime.attempts)
    with pytest.raises(module.ProviderFailure):  # latched for every role/copy afterwards
        lm("again")
    assert engine.calls == len(runtime.attempts)


def test_exhausted_transient_retries_latch_instead_of_silently_raising():
    module, runtime, engine, lm, _ = make([ConnectionResetError("reset")] * 3, max_retries=2)
    with pytest.raises(module.ProviderFailure):
        lm("hello")
    assert engine.calls == 3
    assert runtime.failed


@pytest.mark.parametrize("status", [400, 401, 404])
def test_fatal_status_fails_fast_even_inside_a_transport_error(status):
    try:
        try:
            raise StatusError(status, "bad key")
        except StatusError as inner:
            raise ConnectionResetError("closed after auth failure") from inner
    except ConnectionResetError as outer:
        wrapped = outer
    for error in (StatusError(status, "bad key"), wrapped):
        module, runtime, engine, lm, fake = make([error])
        with pytest.raises(module.ProviderFailure):
            lm("hello")
        assert engine.calls == 1
        assert fake.sleeps == []
        assert runtime.failed


def test_model_is_never_switched_on_retry():
    seen = []

    class Recording(ScriptedEngine):
        def complete(self, request):
            seen.append(request.model)
            return super().complete(request)

    module = runtime_module()
    fake = FakeClock()
    runtime = module.ProviderRuntime(sleep=fake.sleep, clock=fake.clock, random=lambda: 0.0)
    lm = runtime.wrap(dspy.LM("openai/offline", engine=Recording([litellm_timeout(), ConnectionResetError()]),
                              cache=False), "task")
    lm("hello")
    assert len(seen) == 3 and len(set(seen)) == 1


class FlakyTaskLM(DummyLM):
    """Fails its first call with a dropped connection, then answers."""

    def __init__(self, error):
        super().__init__([{"answer": "gold"}] * 20)
        self.error = error
        self.attempt_count = 0

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        self.attempt_count += 1
        if self.attempt_count == 1:
            raise self.error
        return super().__call__(prompt, messages=messages, **kwargs)


def test_test_evaluation_retries_a_transient_failure_and_counts_it(monkeypatch, tmp_path):
    task = FlakyTaskLM(ConnectionResetError("reset by peer"))
    config = ExperimentConfig(train_size=1, val_size=1, test_size=2, results_dir=str(tmp_path))
    harness = ExperimentHarness(config, lm_factory=lambda *args: {
        "task": task, "reflection": DummyLM([]), "critic": DummyLM([])},
        benchmark_factory=lambda name: FixtureBenchmark())

    class NoOpArm:  # no optimization calls: the first task call is a test-evaluation call
        def compile(self, program, **kwargs):
            return program

        def get_stats(self):
            return {}

    monkeypatch.setattr(harness, "create_arm", lambda *args, **kwargs: NoOpArm())
    result = harness.run_single("hotpotqa", "A", 10, 0, backoff_seconds=0.0)
    assert result["status"] == "success"
    assert result["test_scores"] == [1.0, 1.0]
    assert task.attempt_count == 3
    retries = result["budget_stats"]["provider_retries"]
    assert retries["transient_errors"] == 1 and retries["calls_recovered_after_retry"] == 1
    rows = [json.loads(line) for line in (tmp_path / result["artifacts"]["lm_calls"]).read_text().splitlines()]
    assert [row.get("retry_kind") for row in rows if row["role"] == "task"] == ["transient", None, None]


def test_test_evaluation_still_aborts_after_bounded_retries(monkeypatch, tmp_path):
    class DeadLM(DummyLM):
        def __init__(self):
            super().__init__([])
            self.attempt_count = 0

        def __call__(self, prompt=None, *, messages=None, **kwargs):
            self.attempt_count += 1
            raise ConnectionResetError("reset by peer")

    task = DeadLM()
    config = ExperimentConfig(train_size=1, val_size=1, test_size=2, results_dir=str(tmp_path))
    harness = ExperimentHarness(config, lm_factory=lambda *args: {
        "task": task, "reflection": DummyLM([]), "critic": DummyLM([])},
        benchmark_factory=lambda name: FixtureBenchmark())

    class NoOpArm:
        def compile(self, program, **kwargs):
            return program

        def get_stats(self):
            return {}

    monkeypatch.setattr(harness, "create_arm", lambda *args, **kwargs: NoOpArm())
    result = harness.run_single("hotpotqa", "A", 10, 0, backoff_seconds=0.0, max_retries=2)
    assert result["status"] == "error"
    assert result["test_mean"] is None
    assert task.attempt_count == 3
    assert result["budget_stats"]["provider_failed"]


def test_canary_runtime_uses_stage_retry_policy_and_retries_transients(tmp_path):
    config = ExperimentConfig(results_dir=str(tmp_path))
    stage = config.get_stage_config(0)
    stage.backoff_seconds = 0.0
    task = FlakyTaskLM(litellm_timeout())
    harness = ExperimentHarness(config, lm_factory=lambda *args: {
        "task": task, "reflection": DummyLM([]), "critic": DummyLM([])})
    lms = harness._canary_lms(config.model_config, 42, stage)
    runtime = lms["task"].runtime
    assert (runtime.max_retries, runtime.max_delay_seconds, runtime.max_retry_wall_seconds) == (
        stage.max_retries, stage.max_delay_seconds, stage.max_retry_wall_seconds)
    lms["task"]("hello")
    assert task.attempt_count == 2
    assert runtime.retry_summary()["timeouts"] == 1
