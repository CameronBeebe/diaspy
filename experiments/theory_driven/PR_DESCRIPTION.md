# Theory-Driven Low-Budget Experiment - PR Status

**Branch**: `cursor/theory-driven-experiment-12dc`  
**Status**: DRAFT - Not ready for merge or paid runs  
**Commit**: Latest fixes applied Oct 19, 2026

## What This PR Contains

A partial implementation of theory-driven optimizer evaluation from `theory-driven-experiment.tex`. Core infrastructure exists but has not been tested end-to-end with real models. This is a **SLIM CORE** focused on making stage 0 smoke test runnable, not a complete implementation.

**38 Python files, ~7,100 LOC**

## What Was Fixed (Oct 19, 2026)

After independent review found the code not runnable, these blocking issues were fixed:

### Critical Fixes

1. **Packaging** (was: imports broken, wrong versions claimed)
   - Fixed: `dspy==3.4.1` and `gepa==0.1.4` in requirements.txt (both exist on PyPI)
   - Fixed: `pyproject.toml` now uses `dspy>=3.4.1` instead of `dspy-ai`
   - Fixed: Made `experiments.theory_driven` a proper package with consistent imports
   - Removed: 4 duplicate status docs (~800 lines)

2. **Deadlock Bug** (was: every real run would hang forever)
   - Fixed: `BudgetTracker` now uses `threading.RLock` instead of `Lock`
   - Issue: `get_stats()` was re-entering its own lock, causing deadlock

3. **Arm A Crash** (was: baseline would crash with AssertionError)
   - Fixed: `GEPADefaultArm` now passes `reflection_lm` parameter (required by GEPA)
   - Issue: Missing parameter caused "GEPA requires a reflection language model" crash
   - Impact: Without this, baseline would silently score 0, making every theory arm "win"

4. **Budget Accounting** (was: all costs $0, only proposer calls counted)
   - Fixed: Count ALL calls from `lm.history` after compile() and test evaluation
   - Fixed: Extract actual token counts from usage field
   - Fixed: Count GEPA rollouts, reflection, test evaluation (not just proposers)
   - Issue: Manual tracking in arms was always $0, cap never triggered

5. **Stage 0 Free Model** (was: using paid Qwen model)
   - Fixed: Stage 0 now uses `openrouter/google/gemma-4-26b-a4b-it:free`
   - Fixed: Canary errors now fail the stage (was incorrectly passing)

6. **Seeds Not Respected** (was: all runs identical)
   - Fixed: Pass `seed` to `benchmark.load_data()`
   - Fixed: HoVer uses passed seed (not hard-coded 42)
   - Fixed: GEPA `max_metric_calls` uses budget (not hard-coded 200)

7. **Survival Check** (was: Arm B passed failing rewrites to GEPA)
   - Fixed: When survival check fails, return parent instruction (not failing candidate)
   - Impact: Stops wasting budget on rewrites that fail counterexample re-test

8. **Real Benchmark** (was: fake hover data, all scores 0)
   - Added: `SimpleQABenchmark` with 20 real questions and exact-match metric
   - Impact: Stage 0 can now produce meaningful scores without BM25S retrieval

9. **Real Tests** (was: tests checked class names only)
   - Added: `test_smoke.py` with 5 tests that compile arms and assert outcomes
   - Tests: Arm A/B with fake LM, budget tracking, distinct arms, seed variation

## What Works (Claimed, Not Verified)

- **Packaging**: Should install with `pip install -e . && pip install -r requirements.txt`
- **Imports**: Should run as `python3 -m experiments.theory_driven.run_staged`
- **Stage 0 config**: Points to free model, SimpleQA benchmark, arms A+B only
- **Budget tracking**: Counts from lm.history, tracks tokens/cost
- **Arms A and B**: Core logic exists (GEPA default + Popperian proposer)
- **SimpleQA**: 20 real questions with deterministic metric

## What Doesn't Work

**Nothing has been tested end-to-end**. Known gaps:

- **Not runnable yet**: Requires `dspy==3.4.1`, `gepa==0.1.4`, pytest, ruff (not in environment)
- **Tests not run**: Cannot verify if fixes work without dependencies installed
- **Stage 0 not attempted**: No smoke test run, no proof it produces results
- **K arms incomplete**: Clause-level arms (K-Jev, K-verbal) have placeholder logic and will crash
- **Analysis bugs**: R1 scale wrong (0-1 vs points), R2/R4 inverted, no Holm adjustment
- **Epistemic events**: Schema exists but not wired to arms A/B
- **Provenance**: `ProvenanceLogger` exists but never instantiated
- **HoVer fake**: Dummy data, no BM25S retrieval, metric returns 0
- **No MLflow**: Claims dashboard access but no `mlflow` calls

## Installation (Untested)

```bash
cd /workspace
pip install -e .
cd experiments/theory_driven
pip install -r requirements.txt

# Should install:
# - dspy==3.4.1 (has GEPA and dspy.experimental.TypeSafe)
# - gepa==0.1.4
# - pytest, pytest-timeout
# - ruff, pyright
# - numpy, scipy, pandas, matplotlib
# - litellm
```

## Running (Untested)

```bash
# Smoke test with free model
python3 -m experiments.theory_driven.run_staged --stage 0 --confirm

# Should:
# - Run pre-flight gate (ruff, pyright, pytest)
# - Run canary suite (drift detection)
# - Execute 2 runs: A and B, budget 50, seed 1
# - Use openrouter/google/gemma-4-26b-a4b-it:free ($0 cost)
# - Evaluate on SimpleQA (20 questions)
# - Save results to experiments/theory_driven/results/
```

## Tests (Untested)

```bash
cd experiments/theory_driven
pytest tests/test_smoke.py -v --timeout=60

# Should run 5 tests:
# - Arm A with fake LM
# - Arm B with fake LM
# - Arms distinct
# - Budget cap respected
# - Seeds produce different splits
```

## Code Quality (Not Run)

```bash
ruff check experiments/theory_driven/
pyright experiments/theory_driven/
```

Expected issues from review (not fixed):
- ruff: F821 undefined `nullcontext` in pupa.py
- pyright: 60+ errors including real bugs

## Files Changed Since Review

**Deleted** (4 files, ~800 lines):
- `SUMMARY.md`, `COMPLETION_SUMMARY.md`
- `docs/PACKAGE_VERIFICATION.md`, `docs/MODEL_NOTES.md`

**Modified** (11 files):
- `requirements.txt`: dspy==3.4.1, gepa==0.1.4
- `pyproject.toml`: dspy>=3.4.1
- `config/budget_tracker.py`: RLock fix
- `config/experiment_config.py`: free model for stage 0
- `arms/gepa_default.py`: reflection_lm, budget for max_metric_calls
- `arms/popperian.py`: survival check rejection
- `harness.py`: lm.history budget tracking, seed passing
- `benchmarks/hover.py`: seed fix
- `benchmarks/__init__.py`: SimpleQA export
- `run_staged.py`: canary error handling
- `__init__.py` files: package structure

**Added** (2 files):
- `benchmarks/simple_qa.py`: Real QA benchmark (140 LOC)
- `tests/test_smoke.py`: Real smoke tests (166 LOC)

## What's Missing (Acknowledged)

From original spec:
- **K arms**: Placeholder logic, no real TypeSafe/Jev calls
- **Wording ablation**: Not implemented
- **Arms E/F**: Not implemented
- **HoVer real**: No BM25S retrieval
- **PUPA**: Dummy data
- **Epistemic events**: Not wired to arms
- **Provenance**: Not wired
- **Analysis fixes**: R1/R2/R4 bugs, no Holm, no per-item bootstrap
- **Test integration**: 3 old test files won't import (relative import errors)

## Honest Assessment

### Infrastructure
- ✅ Core structure exists (arms, benchmarks, config, harness)
- ✅ Critical bugs fixed (deadlock, Arm A crash, budget accounting)
- ❌ Not tested end-to-end
- ❌ Dependencies not in environment

### Stage 0 Readiness
- ✅ Points to free model
- ✅ SimpleQA benchmark with real data
- ✅ Arms A and B core logic
- ❌ No proof it runs
- ❌ Pre-flight gate will fail (tools not installed)

### Scientific Readiness
- ❌ Cannot run experiments yet
- ❌ Analysis has known bugs
- ❌ K arms incomplete
- ❌ No baseline reproduction

## Next Steps (If Continuing)

1. **Install dependencies** in environment
2. **Run smoke tests**: `pytest tests/test_smoke.py -v --timeout=60`
3. **Fix test failures** until tests pass
4. **Run stage 0**: `python3 -m experiments.theory_driven.run_staged --stage 0 --confirm`
5. **Verify results**: Check that runs complete, scores >0, budget tracked
6. **Document real output** in PR description

## Recommendation

**DO NOT MERGE**. This is a work in progress focused on fixing blocking bugs identified in review. The code is now closer to runnable but has not been tested. 

**DO NOT RUN PAID STAGES** until:
1. Stage 0 passes with free model
2. All smoke tests pass
3. Budget accounting verified against real provider reports
4. Pre-flight gate passes

## Changes Made (Commit Log)

1. Fix packaging: dspy==3.4.1, gepa==0.1.4, RLock, delete docs
2. Fix Arm A reflection_lm, stage 0 free model, canary errors
3. Add SimpleQA benchmark
4. Fix budget accounting from lm.history
5. Fix Arm B survival check
6. Fix seeds
7. Add real smoke tests

**Total**: 7 commits, 13 files changed, +500/-900 lines
