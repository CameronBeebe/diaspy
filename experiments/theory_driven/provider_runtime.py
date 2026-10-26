"""Per-arm provider circuit breaker, independent of DSPy's advisory callbacks.

Integration: create ``ProviderRuntime()``, replace each role LM with
``runtime.wrap(lm, role)``, add ``runtime.should_stop`` to GEPA stop callbacks,
and call ``runtime.raise_if_failed()`` after compile and evaluation. Persist
``runtime.attempts`` rather than dumping raw LM history/kwargs. Wrappers retain
the underlying LM history. DSPy-owned retries are disabled so the runtime owns
attempt accounting. No process-global DSPy state is changed.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
from random import random
import re
from threading import RLock
import time
from typing import Any

import dspy


class ProviderFailure(RuntimeError):
    """A latched provider failure; safe for the harness to persist."""


def provider_status(error: Exception) -> int | None:
    """Inspect chained SDK/DSPy errors without guessing from message text."""
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for name in ("status_code", "status"):
            status = getattr(current, name, None)
            if isinstance(status, int):
                return status
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
        names = {cls.__name__ for cls in type(current).__mro__}
        for status, classes in (
            (404, {"NotFoundError", "LMUnsupportedModelError", "UnsupportedModelError"}),
            (401, {"AuthenticationError", "LMAuthError", "AuthError"}),
            (400, {"BadRequestError", "LMInvalidRequestError", "InvalidRequestError"}),
            (429, {"RateLimitError", "LMRateLimitError"}),
        ):
            if names & classes:
                return status
        current = current.__cause__ or current.__context__
    return None


# Transport-level failures that look like dropped connections (e.g. a VM snapshot
# pausing the box mid-request). Matched by exception class name across the
# cause/context chain, so SDK wrappers (DSPy LM*, LiteLLM, httpx, lm15) all qualify.
# Local lock timeouts are deliberately absent. A bare HTTP status (e.g. a server
# 408 or 5xx response object) is not transient by itself.
TRANSIENT_ERROR_NAMES = frozenset({
    "LMTimeoutError", "LMTransportError",                      # DSPy
    "Timeout", "APITimeoutError", "APIConnectionError",        # LiteLLM / OpenAI SDK
    "TimeoutError", "TransportError",                          # lm15 and builtin TimeoutError
    "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "ConnectError", "ReadError", "WriteError", "RemoteProtocolError",  # httpx/httpcore
    "ServerDisconnectedError", "ClientConnectionError", "ClientOSError",  # aiohttp
    "ConnectionError", "ConnectionResetError", "ConnectionAbortedError",
    "BrokenPipeError", "IncompleteRead", "EOFError", "SSLEOFError", "SSLZeroReturnError",
})


def transient_kind(error: Exception) -> str | None:
    """Return the matching transient class name in the error chain, if any."""
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for cls in type(current).__mro__:
            if cls.__name__ in TRANSIENT_ERROR_NAMES:
                return cls.__name__
        current = current.__cause__ or current.__context__
    return None


def retry_after_seconds(error: Exception, now: float) -> float | None:
    """Read Retry-After across SDK wrappers; malformed hints are advisory only."""
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for source in (current, getattr(current, "response", None)):
            headers = getattr(source, "headers", None)
            if not isinstance(headers, Mapping):
                continue
            for key, value in headers.items():
                if str(key).lower() != "retry-after" or not isinstance(value, (str, int, float)):
                    continue
                try:
                    seconds = float(value)
                except (ValueError, TypeError):
                    try:
                        date = parsedate_to_datetime(str(value))
                        if date.tzinfo is None:
                            date = date.replace(tzinfo=timezone.utc)
                        seconds = date.timestamp() - now
                    except (ValueError, TypeError, OverflowError):
                        continue
                if math.isfinite(seconds) and seconds >= 0:
                    return seconds
        current = current.__cause__ or current.__context__
    return None


def serving_provider(response) -> str | None:
    """Read exposed response metadata, never requested routing/model prefixes.

    BaseLM history retains the LiteLLM ModelResponse; newer DSPy engines
    retain a tuple of canonical responses with the wire body in provider_data.
    LiteLLM's custom_llm_provider describes the API adapter (e.g. OpenRouter),
    not necessarily the provider that served the model, so ignore it.
    """
    if isinstance(response, (list, tuple)):
        providers = {serving_provider(item) for item in response}
        return providers.pop() if len(providers) == 1 else None

    def field(source, key):
        return source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)

    for source in (response, field(response, "provider_data"), field(response, "_hidden_params")):
        for key in ("provider", "provider_name"):
            value = field(source, key)
            if isinstance(value, str) and value.strip():
                return value
    return None


class ProviderRuntime:
    """One shared latch across task/reflection/critic, including LM copies.

    429s and transient transport failures (timeouts, connection resets/EOF;
    see TRANSIENT_ERROR_NAMES) retry with exponential backoff under one shared
    policy; 400/401/404 fail fast, and exhausted retries latch. Each delay is capped at
    max_delay_seconds, and the total retry wall time is capped at
    max_retry_wall_seconds, including attempts and waits. HTTP attempt timeouts
    are bounded by the remaining budget; backends ignoring timeouts cannot be
    interrupted here, but their late results are rejected. Retry-After is a
    minimum, also capped at max_delay_seconds. Elapsed time uses a monotonic
    clock; HTTP dates use UNIX wall time. Inject sleep, random and a single
    UNIX clock for backward-compatible deterministic offline tests (the injected
    clock then drives both clocks). No internal SDK retries are permitted.
    """

    def __init__(self, *, secrets=(), text_limit: int = 1000,
                 max_retries: int = 10, backoff_seconds: float = 2.0,
                 timeout_seconds: float = 120.0, max_delay_seconds: float = 120.0,
                 max_retry_wall_seconds: float = 900.0,
                 sleep=time.sleep, random=random, clock=None, full_prompt_log: bool = False):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not 0 <= backoff_seconds <= 3600:
            raise ValueError("backoff_seconds must be between 0 and 3600")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_delay_seconds <= 0:
            raise ValueError("max_delay_seconds must be positive")
        if max_retry_wall_seconds <= 0:
            raise ValueError("max_retry_wall_seconds must be positive")
        if text_limit < 1:
            raise ValueError("text_limit must be positive")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.max_delay_seconds = max_delay_seconds
        self.max_retry_wall_seconds = max_retry_wall_seconds
        self._sleep = sleep
        self._random = random
        self._clock = time.time if clock is None else clock
        # Preserve legacy single-clock injection for deterministic offline tests.
        self._elapsed_clock = time.monotonic if clock is None else clock
        self.text_limit = text_limit
        self.full_prompt_log = full_prompt_log
        self._secrets = set(filter(None, secrets))
        self._secrets.update(value for key, value in os.environ.items()
                             if value and any(word in key.upper()
                                              for word in ("API_KEY", "API_TOKEN", "SECRET", "PASSWORD")))
        self.attempts: list[dict[str, Any]] = []
        self.failure: ProviderFailure | None = None
        # Serialize entry/check/call: another worker cannot race past a failure.
        self._lock = RLock()

    @property
    def failed(self) -> bool:
        return self.failure is not None

    def should_stop(self, state=None) -> bool:
        return self.failed

    def raise_if_failed(self) -> None:
        if self.failure is not None:
            raise ProviderFailure(str(self.failure)) from None

    def wrap(self, lm: Any, role: str) -> "GuardedLM":
        lm.num_retries = 0
        for key in ("num_retries", "max_retries", "retry_strategy"):
            lm.kwargs.pop(key, None)
        # Only set timeout for real HTTP providers, not custom engines
        if isinstance(getattr(lm, "_engine_spec", None), str):
            lm.kwargs["timeout"] = min(self.timeout_seconds,
                                       lm.kwargs.get("timeout") or self.timeout_seconds)
        return GuardedLM(lm, self, role)

    def sanitize(self, value, *, truncate: bool = True) -> str:
        text = str(value)
        for secret in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"\bsk-[\w-]+", "[REDACTED]", text)
        text = re.sub(r"(?i)Bearer\s+[^\s'\",;]+", "Bearer [REDACTED]", text)
        text = re.sub(r'''(?i)(["']?(?:api[_-]?key|access[_-]?token|password|secret)["']?\s*[:=]\s*)["']?[^\s'",;}]+["']?''',
                      r"\1[REDACTED]", text)
        return text[:self.text_limit] if truncate else text

    def _learn_secrets(self, values):
        for key, value in values.items():
            if isinstance(value, dict):
                self._learn_secrets(value)
            elif isinstance(value, str) and value and any(
                word in key.lower() for word in ("api_key", "token", "secret", "password", "authorization")
            ):
                self._secrets.add(value)

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target.write_text("".join(json.dumps(row) + "\n" for row in self.attempts))

    def retry_summary(self) -> dict[str, int]:
        """Counts over failed attempt rows; each retried row is followed by another attempt."""
        with self._lock:
            rows = [row for row in self.attempts if row.get("error") and row.get("role") != "harness"]
            return {
                "failed_attempts": len(rows),
                "rate_limit_errors": sum(row.get("retry_kind") == "rate_limit" for row in rows),
                "transient_errors": sum(row.get("retry_kind") == "transient" for row in rows),
                "timeouts": sum("Timeout" in str(row.get("transient_error", "")) for row in rows),
                "calls_recovered_after_retry": sum(
                    1 for row in self.attempts
                    if not row.get("error") and isinstance(row.get("attempt"), int) and row["attempt"] > 1),
            }

    def call(self, lm, role, prompt, messages, kwargs):
        with self._lock:
            self.raise_if_failed()
            self._learn_secrets(lm.kwargs)
            self._learn_secrets(kwargs)
            kwargs = {key: value for key, value in kwargs.items()
                      if key not in ("num_retries", "max_retries", "retry_strategy")}
            # Only set timeout for real HTTP providers, not custom engines
            if isinstance(getattr(lm, "_engine_spec", None), str):
                kwargs["timeout"] = min(self.timeout_seconds,
                                        lm.kwargs.get("timeout") or self.timeout_seconds,
                                        kwargs.get("timeout") or self.timeout_seconds)
            full_prompt = self.sanitize(messages if messages is not None else prompt, truncate=False)
            trace = {"role": self.sanitize(role), "model": self.sanitize(lm.model),
                     "prompt": self.sanitize(messages if messages is not None else prompt),
                     "response": "", "usage": {}, "cost": None,
                     "status_code": None, "error": None}
            if self.full_prompt_log:
                trace["full_prompt"] = full_prompt
            history_size = len(lm.history)
            retry_start_time = self._elapsed_clock()
            for attempt in range(self.max_retries + 1):
                # Fresh per-attempt row: never carry a previous attempt's retry labels.
                trace = {**{key: value for key, value in trace.items()
                            if key not in ("retry_kind", "transient_error")}, "attempt": attempt + 1}
                attempt_kwargs = dict(kwargs)
                remaining = self.max_retry_wall_seconds - (self._elapsed_clock() - retry_start_time)
                if remaining <= 0:
                    self.failure = ProviderFailure(self.sanitize(
                        f"{role} {lm.model}: retry wall time exceeded before attempt {attempt + 1}"))
                    self.raise_if_failed()
                if isinstance(getattr(lm, "_engine_spec", None), str):
                    attempt_kwargs["timeout"] = min(kwargs["timeout"], remaining)
                try:
                    result = lm(prompt, messages=messages, **attempt_kwargs)
                    # A backend may ignore its timeout; never accept a late success.
                    if self._elapsed_clock() - retry_start_time >= self.max_retry_wall_seconds:
                        raise TimeoutError("retry wall time exceeded during provider call")
                    trace.update(status_code=None, error=None)
                    break
                except Exception as error:
                    status = provider_status(error)
                    trace.update(status_code=status, error=self.sanitize(error))
                    # 400/401/404 fail fast even if the SDK wraps them in a transport class.
                    kind = None if status in (400, 401, 404) else (
                        "rate_limit" if status == 429 else "transient" if transient_kind(error) else None)
                    if kind is not None:
                        trace["retry_kind"] = kind
                        if kind == "transient":
                            trace["transient_error"] = transient_kind(error)
                    self.attempts.append(trace)
                    if kind is not None and attempt < self.max_retries:
                        # 429 and transient transport failures share one bounded policy:
                        # same attempt counter, per-wait cap and per-call total wall cap.
                        elapsed = self._elapsed_clock() - retry_start_time
                        if elapsed >= self.max_retry_wall_seconds:
                            self.failure = ProviderFailure(
                                self.sanitize(f"{role} {lm.model}: retry wall time exceeded ({elapsed:.1f}s)"))
                            self.raise_if_failed()

                        # Calculate delay with exponential backoff and jitter
                        delay = self.backoff_seconds * 2 ** attempt + self._random()
                        # Cap the delay
                        delay = min(delay, self.max_delay_seconds)
                        # Retry-After is also capped at max_delay_seconds
                        retry_after = retry_after_seconds(error, self._clock())
                        if retry_after is not None:
                            retry_after = min(retry_after, self.max_delay_seconds)
                        actual_delay = max(delay, retry_after or 0.0)

                        # Check if sleeping would exceed wall time
                        if elapsed + actual_delay >= self.max_retry_wall_seconds:
                            remaining = self.max_retry_wall_seconds - elapsed
                            if remaining > 0:
                                self._sleep(remaining)
                            self.failure = ProviderFailure(
                                self.sanitize(f"{role} {lm.model}: retry wall time exceeded after {attempt + 1} attempts"))
                            self.raise_if_failed()

                        self._sleep(actual_delay)
                        continue
                    if status in (400, 401, 404, 429) or kind == "transient":
                        # Exhausted transient retries latch like an exhausted 429: the
                        # run aborts instead of GEPA silently scoring the call as 0.
                        self.failure = ProviderFailure(self.sanitize(f"{role} {lm.model}: {error}"))
                        self.raise_if_failed()
                    raise
            else:
                raise ValueError("max_retries must be nonnegative")
            trace["response"] = self.sanitize(result)
            if self.full_prompt_log:
                trace["full_response"] = self.sanitize(result, truncate=False)
            if len(lm.history) > history_size:
                entry = lm.history[-1]
                provider = serving_provider(entry.get("response"))
                if provider is not None:
                    trace["provider"] = self.sanitize(provider)
                trace["usage"] = {key: value for key, value in entry.get("usage", {}).items()
                                  if key in ("prompt_tokens", "completion_tokens", "total_tokens",
                                             "input_tokens", "output_tokens")
                                  and isinstance(value, (int, float))}
                cost = entry.get("cost")
                trace["cost"] = cost if isinstance(cost, (int, float)) else None
            self.attempts.append(trace)
            return result


class GuardedLM(dspy.BaseLM):
    """Delegate public LM calls, before callback swallowing or engine retries."""

    def __init__(self, lm, runtime: ProviderRuntime, role: str):
        self._lm = lm
        self.runtime = runtime
        self.role = role

    def __getattr__(self, name):
        return getattr(self._lm, name)

    # BaseLM defines these properties, so __getattr__ cannot delegate them.
    # Forward adapter hints only; inference must still enter runtime.call.
    @property
    def supports_function_calling(self) -> bool:
        return self._lm.supports_function_calling

    @property
    def supports_reasoning(self) -> bool:
        return self._lm.supports_reasoning

    @property
    def supports_response_schema(self) -> bool:
        return self._lm.supports_response_schema

    @property
    def supported_params(self) -> set[str]:
        return self._lm.supported_params

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        return self.runtime.call(self._lm, self.role, prompt, messages, kwargs)

    async def acall(self, prompt=None, *, messages=None, **kwargs):
        # Run the same guarded sync path in a worker; never bypass its latch.
        import asyncio
        return await asyncio.to_thread(self, prompt, messages=messages, **kwargs)

    def __deepcopy__(self, memo):
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    def copy(self, **kwargs):
        copied = self._lm.copy(**kwargs)
        # Copies made by optimizers still belong to the same role/budget.
        copied.history = self._lm.history
        return self.runtime.wrap(copied, self.role)
