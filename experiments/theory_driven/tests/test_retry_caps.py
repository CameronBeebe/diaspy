"""Tests for retry delay caps and total wall-time caps."""

import importlib.util
import pytest


def runtime_module():
    spec = importlib.util.find_spec("experiments.theory_driven.provider_runtime")
    assert spec is not None
    return importlib.import_module("experiments.theory_driven.provider_runtime")


class StatusError(Exception):
    def __init__(self, status, message="error", *, response=None, headers=None):
        super().__init__(message)
        self.status_code = status
        self.response = response
        self.headers = headers


class RateLimitWithRetryAfter(StatusError):
    def __init__(self, retry_after_seconds):
        from types import SimpleNamespace
        super().__init__(429, "rate limited")
        self.response = SimpleNamespace(headers={"Retry-After": str(retry_after_seconds)})


class FailingEngine:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise self.error


class ScriptedEngine:
    """Engine that returns scripted results (errors or responses)."""
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


def test_retry_after_is_capped_at_max_delay_seconds():
    """Retry-After headers are capped at max_delay_seconds (120s default)."""
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(
        max_delay_seconds=100.0,
        backoff_seconds=2.0,
        sleep=delays.append,
        random=lambda: 0.0,
        clock=lambda: 1000.0,
    )
    # Retry-After says wait 3600s, but should be capped at 100s
    engine = ScriptedEngine([RateLimitWithRetryAfter(3600)])
    import dspy
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    response = lm("hello")
    assert "ok" in str(response)
    assert engine.calls == 2
    assert delays == [100.0]  # 3600s capped to 100s
    print("✓ Retry-After capped at max_delay_seconds")


def test_exponential_backoff_is_capped_at_max_delay_seconds():
    """Exponential backoff delays are capped at max_delay_seconds."""
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(
        backoff_seconds=60.0,
        max_delay_seconds=100.0,
        sleep=delays.append,
        random=lambda: 0.0,
    )
    # Without cap: 60, 120, 240, ...
    # With 100s cap: 60, 100, 100, ...
    engine = ScriptedEngine([StatusError(429) for _ in range(3)])
    import dspy
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    response = lm("hello")
    assert "ok" in str(response)
    assert engine.calls == 4
    assert delays == [60.0, 100.0, 100.0]  # 120 and 240 capped to 100
    print("✓ Exponential backoff capped at max_delay_seconds")


def test_total_retry_wall_time_cap_stops_further_retries():
    """Total retry wall time is capped at max_retry_wall_seconds (900s default)."""
    module = runtime_module()
    delays = []
    clock_time = [0.0]
    
    def fake_clock():
        return clock_time[0]
    
    def fake_sleep(duration):
        delays.append(duration)
        clock_time[0] += duration
    
    runtime = module.ProviderRuntime(
        backoff_seconds=100.0,
        max_delay_seconds=120.0,
        max_retry_wall_seconds=300.0,  # Cap at 5 minutes
        sleep=fake_sleep,
        random=lambda: 0.0,
        clock=fake_clock,
    )
    
    # Each retry would want to sleep 100, 120 (capped), 120, 120, ...
    # After 100 + 120 + 120 = 340s > 300s cap, it should abort
    engine = FailingEngine(StatusError(429, "persistent rate limit"))
    import dspy
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    
    with pytest.raises(module.ProviderFailure, match="retry wall time exceeded"):
        lm("hello")
    
    # Should have done: 100s, 120s, then stopped (80s remaining < 120s needed)
    assert len(delays) == 3  # 100, 120, 80 (partial)
    assert delays[0] == 100.0
    assert delays[1] == 120.0
    assert delays[2] == 80.0  # Remaining time before cap
    assert clock_time[0] == 300.0  # Hit the cap
    assert engine.calls == 3  # 1 initial + 2 retries
    print("✓ Total retry wall time capped at max_retry_wall_seconds")


def test_retry_after_plus_backoff_both_capped():
    """When both Retry-After and backoff apply, both respect max_delay_seconds."""
    module = runtime_module()
    delays = []
    runtime = module.ProviderRuntime(
        backoff_seconds=50.0,
        max_delay_seconds=80.0,
        sleep=delays.append,
        random=lambda: 0.0,
        clock=lambda: 1000.0,
    )
    
    engine = ScriptedEngine([
        StatusError(429),  # No Retry-After: backoff 50s
        RateLimitWithRetryAfter(200),  # Retry-After 200s → capped to 80s
        StatusError(429),  # No Retry-After: backoff 100s → capped to 80s
    ])
    import dspy
    lm = runtime.wrap(dspy.LM("openai/offline", engine=engine, cache=False), "task")
    response = lm("hello")
    assert "ok" in str(response)
    assert engine.calls == 4
    assert delays == [50.0, 80.0, 80.0]  # All capped at 80s where needed
    print("✓ Both Retry-After and backoff respect max_delay_seconds")
