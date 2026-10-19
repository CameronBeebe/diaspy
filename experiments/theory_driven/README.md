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
checks (B), using `dspy==3.4.1` and `gepa==0.1.4`. HotpotQA distractor validation
examples are split deterministically by seed: 20 train, 20 validation, 50 test.
Retrieval uses BM25 over each question's own ten paragraphs; scoring is answer
F1. The first real run downloads and caches the subset from
`hotpotqa/hotpot_qa` (CC BY-SA 4.0); tests use attributed checked-in examples.

To explicitly allow network/model execution, provide `OPENROUTER_API_KEY` in
your environment, then use the same entry point with `--confirm` instead of
`--dry-run`. The configured model is
`openrouter/google/gemma-4-26b-a4b-it:free`. No real-model run is part of the
offline checks. Free-model availability and response quality are not validated
by these checks.

Each run has a shared 50-call optimization budget across task, reflection and
critic histories. GEPA checks the stopper between iterations, so one iteration
may overshoot. Held-out evaluation has a separate 50-task-call allocation;
reported totals include it and show optimization overshoot separately. The
$1 stage safety cap uses costs present in LM histories; absent provider costs
count as zero, not as independently verified billing. LM caching is disabled.
Adaptive role counts are shown as shared ranges, not exact predictions.

Only stage 0 is runnable here; pilot arms/benchmarks and statistical analyses
are not validated for real experiments.
