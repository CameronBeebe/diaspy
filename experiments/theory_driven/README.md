# Theory-driven low-budget experiment

## How to run stage 0

From the repository root, use Python 3.12 and install:

```sh
python -m pip install -e .
python -m pip install -r experiments/theory_driven/requirements.txt
python -m pytest experiments/theory_driven/tests --disable-socket
python -m experiments.theory_driven.preflight --stage 0
python -m experiments.theory_driven.run_staged --stage 0 --dry-run
```

Stage 0 compares stock GEPA (A) with typed refutation and candidate survival
checks (B) and Millian adversarial discussion (M), using `dspy==3.4.1` and `gepa==0.1.4`. HotpotQA distractor validation
examples are split deterministically by seed: 20 train, 20 validation, 50 test.
Retrieval uses BM25 over each question's own ten paragraphs; scoring is answer
F1. The first real run downloads and caches the subset from
`hotpotqa/hotpot_qa` (CC BY-SA 4.0); tests use attributed checked-in examples.

To explicitly allow network/model execution, provide `OPENROUTER_API_KEY` in
your environment, then use the same entry point with `--confirm` instead of
`--dry-run`. The configured model is
`openrouter/qwen/qwen3.5-9b` for both task and canary, with a $1 stage cap.
This model is paid; dry-run reports unknown usage cost, not $0. No live model
calls are part of the offline checks. Live runs execute the canary before the arms; the
first successful canary establishes a baseline. Provider failures block the
stage, while `allow_canary_drift` controls reference drift only. OpenRouter
reasoning uses `{"enabled": false}`; provider pinning is unchanged. The shared
provider guard retries HTTP 429 up to five times (2s exponential backoff plus
jitter, capped at 60s; longer Retry-After hints are honored), never retries 400/401/404,
and never changes models. DSPy retries are disabled to avoid nested retries.
Any failed arm stops the stage with exit code 1; use `set -o pipefail` if piping
through `tee`. Result JSON points to sanitized compact LM-attempt logs and
sparse epistemic events (B decisions mean survival admission, not GEPA's final
acceptance; M records both rival rewrites and the winning proposal). Canary artifacts are under `results/canaries/`.

Each run has a shared 100-call optimization budget across task, reflection and
critic histories. GEPA checks the stopper between iterations, so one iteration
may overshoot. Held-out evaluation has a separate 50-task-call allocation;
reported totals include it and show optimization overshoot separately. The
$1 stage safety cap uses costs present in LM histories; absent provider costs
count as zero, not as independently verified billing. LM caching is disabled.
Adaptive role counts are shown as shared ranges, not exact predictions.

B and M default to `strict`; stage 0 compares `A`, `B@half_no_regress`,
`M@half_no_regress`, and `B@majority`. Non-vacuous survival checks target a
minimum of three questions; thin sets are reported rather than fabricated. Corroboration and admission use observed
parent/rewrite F1, not self-reported severity. `veto` adds archive protection.
`severity_tau` is a deprecated explicit legacy-filter opt-in, not a stage default.
M tests prosecution and dialectical rewrites on the same set; score gain then
instruction length breaks ties. Stances steer prompts, not mechanical decisions.

Retry rewrites receive typed failed-attempt scores; duplicate instructions skip
survival rollouts. Zero-reference checks are marked vacuous and do not vote.
Arm A events retain full instructions and actual GEPA candidate-pool decisions.

Use `--resume experiments/theory_driven/results` with `--confirm` to skip
completed matching arm/config hashes; the canary still runs. Skipped results
retain their original costs against the stage cap. Results are git-ignored.
`OPENROUTER_ALLOW_FALLBACKS=true` and comma-separated `OPENROUTER_PROVIDER_ORDER`
configure same-model provider routing, never another model. The preflight cost
estimate is assumption-based, not a live quote or guaranteed billing bound.

Only stage 0 is runnable here; pilot arms/benchmarks and statistical analyses
are not validated for real experiments.
