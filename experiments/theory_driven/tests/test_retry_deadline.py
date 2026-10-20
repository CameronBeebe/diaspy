"""Offline regression tests for the per-call retry deadline."""

import pytest

from experiments.theory_driven.provider_runtime import ProviderFailure, ProviderRuntime
from experiments.theory_driven.tests.test_transient_retry import FakeClock, make


class TimedProvider:
    """Offline callable with the HTTP-provider interface and timeout semantics."""

    _engine_spec: object = "openai"
    model = "openai/offline"

    def __init__(self, fake, durations, errors=(), **kwargs):
        self.fake = fake
        self.durations = iter(durations)
        self.errors = iter(errors)
        self.kwargs = kwargs
        self.history = []
        self.calls = []

    def __call__(self, prompt, *, messages=None, **kwargs):
        self.calls.append({"model": self.model, **kwargs})
        duration = next(self.durations)
        timeout = kwargs.get("timeout", self.kwargs.get("timeout", float("inf")))
        self.fake.now += min(duration, timeout)
        if duration >= timeout:
            raise TimeoutError("provider timed out")
        error = next(self.errors, None)
        if error:
            raise error
        return ["ok"]


def test_each_attempt_timeout_is_bounded_by_remaining_deadline():
    fake = FakeClock()
    runtime = ProviderRuntime(clock=fake.clock, sleep=fake.sleep, random=lambda: 0.0,
                              backoff_seconds=0.0, max_retry_wall_seconds=5.0)
    raw = TimedProvider(fake, [4.0, 4.0], [ConnectionResetError("reset")])
    lm = runtime.wrap(raw, "task")

    with pytest.raises(ProviderFailure):
        lm("hello")

    assert [call["timeout"] for call in raw.calls] == [5.0, 1.0]
    assert {call["model"] for call in raw.calls} == {"openai/offline"}
    assert fake.now == 5.0
    assert len(runtime.attempts) == 2


@pytest.mark.parametrize("http_provider", [False, True])
def test_no_dispatch_after_sleep_overshoots_deadline(http_provider):
    fake = FakeClock()

    def oversleep(seconds):
        fake.sleep(seconds + 5.0)

    runtime = ProviderRuntime(clock=fake.clock, sleep=oversleep, random=lambda: 0.0,
                              backoff_seconds=1.0, max_retry_wall_seconds=5.0)
    raw = TimedProvider(fake, [0.0, 0.0], [ConnectionResetError("reset")])
    if not http_provider:
        raw._engine_spec = object()
    lm = runtime.wrap(raw, "task")

    with pytest.raises(ProviderFailure, match="retry wall time exceeded"):
        lm("hello")

    assert len(raw.calls) == 1
    assert len(runtime.attempts) == 1  # no synthetic provider attempt
    assert runtime.failed


@pytest.mark.parametrize("lm_timeout,call_timeout,expected", [
    (0.5, None, 0.5), (None, 0.5, 0.5), (0.5, 1.0, 0.5), (None, 10.0, 2.0),
])
def test_explicit_timeout_is_never_extended(lm_timeout, call_timeout, expected):
    fake = FakeClock()
    runtime = ProviderRuntime(clock=fake.clock, timeout_seconds=2.0,
                              max_retry_wall_seconds=5.0)
    raw = TimedProvider(fake, [0.0], **({"timeout": lm_timeout} if lm_timeout else {}))
    lm = runtime.wrap(raw, "task")
    assert lm("hello", **({"timeout": call_timeout} if call_timeout else {})) == ["ok"]
    assert raw.calls[0]["timeout"] == expected
    assert raw.kwargs["timeout"] == (lm_timeout or 2.0)


def test_default_deadline_uses_monotonic_time(monkeypatch):
    from experiments.theory_driven import provider_runtime

    fake = FakeClock()
    monkeypatch.setattr(provider_runtime.time, "monotonic", fake.clock)
    runtime = ProviderRuntime(sleep=fake.sleep, random=lambda: 0.0,
                              backoff_seconds=0.0, max_retry_wall_seconds=5.0)
    raw = TimedProvider(fake, [4.0, 4.0], [ConnectionResetError("reset")])

    with pytest.raises(ProviderFailure):
        runtime.wrap(raw, "task")("hello")

    assert [call["timeout"] for call in raw.calls] == [5.0, 1.0]
    assert fake.now == 5.0


@pytest.mark.parametrize("max_retries", [1, 10])
def test_success_after_deadline_is_rejected_even_if_backend_ignores_timeout(max_retries):
    fake = FakeClock(call_seconds=4.0)
    _, runtime, engine, lm, _ = make(
        [ConnectionResetError("reset")], fake,
        backoff_seconds=0.0, max_retry_wall_seconds=5.0, max_retries=max_retries,
    )

    with pytest.raises(ProviderFailure, match="retry wall time exceeded"):
        lm("hello")

    assert fake.now == 8.0
    assert engine.calls == 2
    assert runtime.failed
    assert len(runtime.attempts) == 2
    assert runtime.attempts[-1]["error"]
    assert runtime.retry_summary()["calls_recovered_after_retry"] == 0
    with pytest.raises(ProviderFailure):
        lm("again")
    assert engine.calls == 2


@pytest.mark.parametrize("wall_jump", [-1000.0, 1000.0])
def test_retry_after_dates_use_wall_clock_but_budget_includes_monotonic_waits(monkeypatch, wall_jump):
    from experiments.theory_driven import provider_runtime
    from experiments.theory_driven.tests.test_provider_runtime import StatusError

    fake = FakeClock()
    wall = [1000.0]
    monkeypatch.setattr(provider_runtime.time, "monotonic", fake.clock)
    monkeypatch.setattr(provider_runtime.time, "time", lambda: wall[0])

    def sleep(seconds):
        fake.sleep(seconds)
        wall[0] += wall_jump

    runtime = ProviderRuntime(sleep=sleep, random=lambda: 0.0,
                              backoff_seconds=0.0, max_retry_wall_seconds=5.0)
    raw = TimedProvider(fake, [1.0, 4.0], [StatusError(429, headers={
        "Retry-After": "Thu, 01 Jan 1970 00:16:42 GMT",
    })])
    with pytest.raises(ProviderFailure, match="retry wall time exceeded"):
        runtime.wrap(raw, "task")("hello")

    assert fake.sleeps == [2.0]
    assert [call["timeout"] for call in raw.calls] == [5.0, 2.0]
    assert fake.now == 5.0
