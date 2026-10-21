# Synthetic positive-control diagnostic

This is a **proposer-level diagnostic**, not a GEPA compile, benchmark replacement,
efficacy study, or generalization result. It calls the existing `PopperianProposer`
(B) and `MillianProposer` (M), strict JSON adapter, score-based survival evaluator,
`create_lms`, and `ProviderRuntime`. No HotpotQA settings or production prompts change.

From the repository root, the default is an offline, no-call plan:

```sh
.venv/bin/python -m experiments.theory_driven.positive_control
```

Only after explicit authorization, with `OPENROUTER_API_KEY` already configured:

```sh
.venv/bin/python -m experiments.theory_driven.positive_control --confirm \
  --model openrouter/qwen/qwen3.5-9b \
  --output experiments/theory_driven/results/positive-control-live-01
```

The output directory must not already exist. No live call is needed for tests:

```sh
.venv/bin/python -m pytest experiments/theory_driven/tests/test_positive_control.py --disable-socket -q
```

## Matched controls and interpretation

Each arm gets two fresh cells with identical manually authored synthetic records,
model, seed, temperature, call ceiling, and one rewrite attempt. Only the parent
instruction differs: clean returns `record[requested_field]`; defective ignores the
requested field and always returns `city`. Three reflection records contain two
color requests (intended failures) and one city request (behavior to preserve).
Three fixed hidden records have disjoint entities, colors, and cities. Hidden
records are never supplied to the proposer, its gold map, or survival selection.
Gold is excluded from task inputs; reflection receives training-only metric feedback.

Parent reflection answers are actual task-LM rollouts, not prefilled failures.
The exact metric compares strings without normalization. The real survival gate
scores proposed rewrites on reflection records, admitting only according to its
existing strict policy. M retains its two rival rewrites and defender; B has one
rewrite. Both have the same ceilings, **not forced equal realized call counts**.
Duplicate/no-refutation paths legitimately consume fewer calls. M's duplicate
rivals may reuse evaluations under the existing implementation.

After freezing the proposer result, parent and selected instruction each receive
fresh evaluations on all hidden records—even if the selected instruction is
unchanged. Rejected rewrites have survival evaluations, not hidden evaluations.
No hidden score feeds admission. If the task model ignores the deliberately wrong
instruction, the positive control is inconclusive, not a repair success. Inspect
per-question outputs and scores; a healthy diagnostic separates admitted repair
of observed defects from preservation of the clean control. Empty critiques in
both conditions remain visible as failure to repair observed defective behavior.
A completed run's `status=success` means execution completed, **not** that repair
was successful. There are no statistical or cross-task conclusions.

## Bounds and artifacts

- Maximum 20 provider attempts per cell, 80 total, including task evaluations and
  adapter fallback attempts; no SDK/runtime transient retries, one proposal round.
- Same model for task, prosecution, rewriting and defense; temperature 0, seed 0,
  `n=1`, cache off, reasoning off, provider fallback off. Provider nondeterminism
  is still possible; serving metadata is retained when available.
- Task completion limit 128 tokens; reflection/defense/rewrite limit 1024 tokens;
  request timeout 30 seconds and runtime wall check 40 seconds.
- **$0.25 is a soft reported-cost stop, not a hard billing cap.** Missing/delayed
  provider cost metadata and the call crossing the threshold can exceed it.
  Calls are checked before and after completion, including the last evaluation.
- `result.json`: source label, settings, original/selected instructions, actual
  parent outputs, full refutations/rebuttals/rival instructions and admission
  records, hidden scores/outputs, and diagnostic observations.
- `calls.jsonl`: sanitized full prompts/responses plus role, usage, cost and
  exposed serving-provider metadata, including partial runs. No API keys logged.
  The `full_prompt`/`full_response` fields are lossless text representations (apart
  from secret redaction); the ordinary short fields are previews.
- Failed provider or strict structured parsing aborts and persists partial results;
  the CLI exits nonzero. No parser relaxation or synthesized measured output.

Offline integration tests use explicitly scripted `DummyLM` responses through the
real provider wrapper/proposers. Their repair/rejection results prove wiring and
gates only; they are **not measured model performance**. Run one authorized live
comparison and inspect the artifacts before drawing any diagnostic conclusion.
