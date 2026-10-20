# Live smoke findings and corrective boundary

## Scope

Two small live Qwen3.5-9B/OpenRouter wiring smokes: six training examples, three validation examples, two held-out examples, one seed, nominal optimization budget 12, A/B/M. Not a statistically meaningful comparison. Provider-reported costs, including canaries, were $0.00622314 and $0.00564858 respectively; not reconciled against billing. Both created fresh canary baselines rather than testing historical drift.

## Observed problem

The first smoke's correct temporal comparison received partial token-F1 because it included an explanation. Critics diagnosed a reasoning failure despite seeing the correct answer and supporting dates. Rewrites sometimes made correct answers longer and therefore lower-scoring. Other failures involved missing retrieved support or ambiguous source wording, not established instruction defects.

Raw structured model responses exposed a separate correctness defect: stock DSPy/json-repair could silently replace consecutive similarly shaped lists with the last list. One Millian critique disappeared before the defender received it. Another malformed response triggered a fresh JSON-adapter generation, changing the criticism rather than merely repairing formatting.

## Changes

- Common minimal-answer contract for all HotpotQA arms; numerical F1 unchanged.
- Metric feedback explains verbosity penalties and distinguishes retrieval coverage from semantic correctness.
- Criticism/defense treat diagnoses as hypotheses and permit empty refutations.
- Strict structured-role JSON boundary rejects multiple values, duplicate keys, dropped nested members, and invalid native types rather than repairing away information.
- Per-compile parsing-error latch prevents GEPA's internal retry/suppression from turning malformed criticism into an ordinary no-change or successful result. No dependency patch; task adapters unchanged.

## Rerun observation

On the second smoke, the temporal/AFC examples produced concise correct answers across all arms. A proposed a rewrite that did not improve its minibatch; it was rejected. B and M returned valid empty lists, retaining the original instruction. All held-out score vectors were [0, 1]. This shows cleaner execution on these cases, not method superiority. The simultaneous contract, feedback and adapter changes do not isolate their causal contributions.

The live rerun did not exercise nonempty strict-JSON debate/rewrite paths. Real compile-level DummyLM regressions exercise those paths and parsing failures; offline correctness is not a substitute for live coverage. The final parsing-error latch was added after the rerun and verified offline.

## Limits / next experiment

- A nominal optimization budget is checked at iteration boundaries and can overshoot; it is not an exact matched-call budget or a hard billing ceiling.
- Empty criticism may reflect appropriate restraint or excessive conservatism. Test a bounded, evidence-complete, deliberately flawed instruction as a positive control before interpreting benchmark abstention.
- Preserve both the control and ordinary tasks; do not replace inconvenient cases or tune on held-out outcomes.
- No larger paid benchmark was launched.

## Local evidence

- Original run: `/Users/sciphi/diaspy-smoke-results/20261020T210158Z/`
- Rerun: `/Users/sciphi/diaspy-smoke-results/20261020T215412Z/`
- Each directory contains `inspection.md`, result JSON, LM-call JSONL and epistemic events.
- Rerun directory additionally preserves the smoke script and tracked/untracked source snapshot. Raw run artifacts remain local, not published in the repository.
