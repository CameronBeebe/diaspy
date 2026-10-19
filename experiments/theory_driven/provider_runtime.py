"""Per-arm provider circuit breaker, independent of DSPy's advisory callbacks.

Integration: create ``ProviderRuntime()``, replace each role LM with
``runtime.wrap(lm, role)``, add ``runtime.should_stop`` to GEPA stop callbacks,
and call ``runtime.raise_if_failed()`` after compile and evaluation. Persist
``runtime.attempts`` rather than dumping raw LM history/kwargs. Wrappers retain
the underlying LM history. DSPy-owned retries are disabled so the runtime owns
attempt accounting. No process-global DSPy state is changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
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


class ProviderRuntime:
    """One shared latch across task/reflection/critic, including LM copies."""

    def __init__(self, *, secrets=(), text_limit: int = 1000,
                 max_retries: int = 2, backoff_seconds: float = 1.0,
                 sleep=time.sleep):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer between 0 and 5")
        if not 0 <= backoff_seconds <= 30:
            raise ValueError("backoff_seconds must be between 0 and 30")
        if text_limit < 1:
            raise ValueError("text_limit must be positive")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self.text_limit = text_limit
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
        return GuardedLM(lm, self, role)

    def sanitize(self, value) -> str:
        text = str(value)
        for secret in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"\bsk-[\w-]+", "[REDACTED]", text)
        text = re.sub(r"(?i)Bearer\s+[^\s'\",;]+", "Bearer [REDACTED]", text)
        text = re.sub(r'''(?i)(["']?(?:api[_-]?key|access[_-]?token|password|secret)["']?\s*[:=]\s*)["']?[^\s'",;}]+["']?''',
                      r"\1[REDACTED]", text)
        return text[:self.text_limit]

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

    def call(self, lm, role, prompt, messages, kwargs):
        with self._lock:
            self.raise_if_failed()
            self._learn_secrets(lm.kwargs)
            self._learn_secrets(kwargs)
            kwargs = {key: value for key, value in kwargs.items()
                      if key not in ("num_retries", "max_retries", "retry_strategy")}
            trace = {"role": self.sanitize(role), "model": self.sanitize(lm.model),
                     "prompt": self.sanitize(messages if messages is not None else prompt),
                     "response": "", "usage": {}, "cost": None,
                     "status_code": None, "error": None}
            history_size = len(lm.history)
            for attempt in range(self.max_retries + 1):
                trace = {**trace, "attempt": attempt + 1}
                try:
                    result = lm(prompt, messages=messages, **kwargs)
                    trace.update(status_code=None, error=None)
                    break
                except Exception as error:
                    status = provider_status(error)
                    trace.update(status_code=status, error=self.sanitize(error))
                    self.attempts.append(trace)
                    if status == 429 and attempt < self.max_retries:
                        self._sleep(min(self.backoff_seconds * 2 ** attempt, 30.0))
                        continue
                    if status in (400, 401, 404, 429):
                        self.failure = ProviderFailure(self.sanitize(f"{role} {lm.model}: {error}"))
                        self.raise_if_failed()
                    raise
            else:
                raise ValueError("max_retries must be nonnegative")
            trace["response"] = self.sanitize(result)
            if len(lm.history) > history_size:
                entry = lm.history[-1]
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
