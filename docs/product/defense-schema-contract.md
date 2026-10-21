# Defense serialization failure: pre-edit evidence

Baseline: clean `3fb1206`, originally `experiment/positive-control`; investigation branch `fix/defense-schema-contract`. Recorded before production/test edits. No live rerun authorized.

Source directory: `/Users/sciphi/diaspy-experiments/gepa-pilot/live-v2-artifacts` (read-only).

- `summary.json`: status `incomplete`, 121 attempts, $0.01444595 accounted, 2 completed compilations (seed 0 A and B@half_no_regress), 0 successful hidden items. This is an aborted pilot, not a comparative result.
- `ledger.jsonl`: 368 rows; 121 reserved, 121 accounted, 121 runtime_trace, 2 program_frozen, 1 opened, 1 failure, 1 finalized. Decimal sum of accounted `cost_usd` is exactly `0.01444595`; 121 unique accounted call IDs.
- Last call: `call-0121`, seed 0, M@half_no_regress, optimization/defender, `openrouter/qwen/qwen3.5-9b`, provider DeepInfra, one recorded attempt, $0.00045115, 3694 prompt / 545 completion / 4239 total tokens, output limit 2048.
- Ledger failure occurs immediately after that call's runtime_trace. The only subsequent event is `finalized`; no reservations, accounting events, or model traces after failure.
- `harness/lm_calls/343bc74a482a451882bb4c0f99edd2df_M@half_no_regress.jsonl`: final model response is complete, syntactically valid JSON with two rebuttals. First has `protected_ids: [2]`; second omits `protected_ids`. Next row is the harness `AdapterParseError`, specifically `1.protected_ids` / `Field required`. The strict boundary correctly rejected it.
- Raw prompt already includes all four fields in the nested JSON schema's `required` array. However, Rebuttal fields have no descriptions; Defense prose does not explicitly say every rebuttal needs all four fields or to serialize `protected_ids: []` when none are cited. Arguments have no brevity guidance.
- No saved artifact contains `finish_reason` or the request's `response_format`. Thus the exact live finish_reason and actual live HTTP schema mode cannot be read back. The captured response closes normally and is below the output limit; these facts do not establish the provider's finish_reason. Do not attribute omission to truncation or claim `stop` was observed.

Scope: improve serialization instructions/descriptions only; leave parser, requiredness, retries, role/model settings, decision policy, scoring, and budgets unchanged. Verify request construction offline and distinguish that evidence from captured live HTTP data.

## Offline findings and change

Intercepted actual HTTP construction with the pilot's model, explicit `engine="litellm"`, routing, decoding, and inherited `ProviderRuntime.wrap` path. **Guarded path sends no `response_format` at all** (neither `json_schema` nor `json_object`); it relies on prompt schema instructions plus local strict validation. `GuardedLM` inherits `BaseLM.supported_params` (empty set) and `supports_response_schema` (false); those properties shadow `__getattr__` delegation. The bare-LM comparison sends `json_schema` with `strict: true`, all four required fields, and descriptions. This is reconstructed offline transport evidence, not a saved live HTTP request. No capability/transport change was made: switching schema enforcement would broaden this serialization-only patch.

`Rebuttal` now describes each field without introducing defaults; `Defense` explicitly requires all four fields on every rebuttal, says `protected_ids: []` when no examples are cited, and asks for concise arguments without repeating full feedback. No decision-policy guidance was added or changed. The cause of the model's omission is not established; clearer serialization guidance is not proof a provider will obey it.

The runner checks for `length`, `content_filter`, or `error` before adapter parsing. The saved run reached missing-field parsing instead; no truncation failure was recorded. Since raw provider finish_reason was not persisted, its exact value cannot be recovered. Complete JSON / 545 completion tokens against 2048 are consistent with field omission, not evidence of a cut-off response.

## Verification (offline only, uncommitted)

- RED: raw-prompt/schema regression failed on the missing explicit per-rebuttal instruction before production edits (`1 failed`).
- GREEN: `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest experiments/theory_driven/tests/test_defense_schema_contract.py --disable-socket -q`: **5 passed**. Includes exact captured raw-response/hash replay through real M compile; missing second-list field remains `AdapterParseError`, only one defense call, no retry/rewrite/later task call or successful debate persistence. A separately labeled synthetic counterpart with explicit empty list is accepted losslessly. HTTP responses are mocked at `httpx.Client.send`, with sockets disabled and credentials stripped.
- `.venv/bin/ruff check experiments/theory_driven`: **All checks passed!**
- `.venv/bin/pyright experiments/theory_driven`: **0 errors, 0 warnings, 0 informations**.
- `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest experiments/theory_driven/tests --disable-socket -q`: **471 passed, 455 warnings** (deprecation/collection/TypedDict warnings, not a warning-free run).
- `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest tests --disable-socket -q`: **18 passed**.
- `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -c 'from pathlib import Path; from experiments.theory_driven.preflight import PreFlightGate; raise SystemExit(0 if PreFlightGate(Path("experiments/theory_driven")).run_all_checks(0) else 1)'`: **exit 0**, lint/type/offline tests and planning passed.
- `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/diaspy experiment --stage 0 --dry-run`: **exit 0**, reports no LLM calls or dataset downloads.
- `git diff --check`: **exit 0**.

No live rerun, commit, push, parser relaxation, implicit empty-list default, or retry added. Original pilot artifacts remain untouched. No live reliability or comparative efficacy claim follows from these tests.

## Capability root-cause fix (supersedes serialization-only scope above)

The earlier sections preserve the pre-fix investigation and its verification. The authorized follow-up fixes the wrapper, not just the prompt. Installed DSPy **3.4.1** defines four adapter-facing properties on `BaseLM`: `supported_params`, `supports_response_schema`, `supports_function_calling`, and `supports_reasoning`. Its JSON adapter checks the first two before sending a response schema; the base and reasoning adapters consume the other two. `__getattr__` cannot override these inherited properties.

`GuardedLM` now explicitly delegates only these four capability properties to the wrapped LM. No inference methods, decision-request support, private capability APIs, retry configuration, runtime accounting, async guard, or copy mechanics were changed. Direct delegation preserves custom-engine and legacy BaseLM overrides and DummyLM's unsupported defaults. Sync/async inference still enters `runtime.call`, and copies retain the shared runtime/history.

Evidence from actual request construction with mocked `httpx` transport (not live provider inference):

- RED: `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest experiments/theory_driven/tests/test_defense_schema_contract.py -k actual_http --disable-socket -q --tb=short` returned **6 failed, 2 passed, 3 deselected**. Guarded, copy, and deepcopy each failed in sync and async with `KeyError: 'response_format'`; both bare-LM baselines passed.
- GREEN after explicit schema capability delegation: `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest experiments/theory_driven/tests/test_defense_schema_contract.py --disable-socket -q --tb=short` returned **11 passed, 6 warnings**. All eight wire variants send `response_format.type=json_schema`, `strict=true`, and all four required Rebuttal fields including `protected_ids`; nested additional properties are forbidden. Routing, reasoning settings, decoding, timeout, one HTTP request, and usage/cost/provider accounting are asserted. Copies cannot restore SDK retries.
- Additional RED: `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest experiments/theory_driven/tests/test_provider_runtime.py -k guarded_adapter_capabilities --disable-socket -q --tb=short` returned **2 failed, 4 passed, 70 deselected**, exposing the remaining function/reasoning capability shadowing for enabled custom/legacy implementations. Both additional properties were then delegated. This test covers all four properties, disabled support, DummyLM, copies/deepcopies, and zero inference during inspection.
- Final full offline suite: `LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest tests experiments/theory_driven/tests --disable-socket -q --tb=short`: **501 passed, 460 warnings**. Existing provider latch/retry/async safety tests and strict missing-field compile rejection remain green. Warnings are deprecation/collection/TypedDict warnings, not a warning-free run.
- `.venv/bin/ruff check experiments/theory_driven`: **All checks passed!**
- `.venv/bin/pyright experiments/theory_driven`: **0 errors, 0 warnings, 0 informations**.
- The preflight and no-call dry-run commands listed above were rerun: both **exit 0**; preflight reports **483 passed, 460 warnings** and clean experiment lint/type checks.
- Expanded checks outside the experiment expose existing issues in unchanged files: `.venv/bin/ruff check src tests experiments/theory_driven` reports **14 errors** (unused imports/variables in `src/diaspy/responders.py`, `src/diaspy/utils.py`, `tests/test_qc.py`, `tests/test_responders.py`); `.venv/bin/pyright src tests experiments/theory_driven` reports **4 errors, 4 warnings** (`src/diaspy/utils.py` attribute accesses and `src/diaspy/__init__.py` exports). These unrelated files were not changed.
- `git diff --check`: **exit 0**.

Previous prompt edits and the recorded response fixture are preserved. No live calls, commits, pushes, SDK/provider retry changes, or parser relaxation were made. Offline schema negotiation is verified; live provider adherence is not.
