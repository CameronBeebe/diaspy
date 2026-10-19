# Theory-Driven GEPA Experiment Workspace

## Overview

This PR adds a complete experiment workspace for the pre-registered low-data DSPy optimization study comparing typed Popperian critic vs GEPA default and other optimizers (see `uploads/theory-driven-experiment.tex` for full research design).

## What's Delivered

### 1. Clean, Importable Experiment Harness

**Location:** `experiments/`

- **`arms.py`**: Arm definitions (A0, A, E, C, B, D, F) including:
  - `PopperianProposer`: Typed Popperian conjecture-and-refutation with survival check
  - `CheckOnlyProposer`: Filter-only control (no critic)
  - `UntypedCriticProposer`: Prose critic without mechanical check
  - `MillianProposer`: Two rival critics with debate
  
- **`config.py`**: Configuration structures and budget tracking
  - `ExperimentConfig`: Full grid specification
  - `BudgetConfig`: Per-run constraints (1,000 rollouts, 60 reflection calls, 300k tokens)
  - `BudgetTracker`: Enforcement and reporting

- **`runner.py`**: Main experiment runner with equal-budget accounting
  - Runs full arm × N × seed grid
  - Tracks budget usage (simplified for now - post-hoc from logs)
  - Handles failures gracefully

- **`store.py`**: JSONL/gzip results storage
  - `RunResult`: Per-run outcome with test predictions for bootstrap
  - `ResultsStore`: Append-only JSONL with compression

- **`analysis.py`**: Paired bootstrap and refutation criteria R1-R5
  - `paired_bootstrap_ci`: 95% CIs over seeds and test items
  - `compare_arms`: Pairwise comparisons
  - `check_refutation_criteria`: Pre-registered hypothesis tests

- **`task.py`**: IFBench task setup (simplified for testing)
- **`mlflow_utils.py`**: MLflow integration
- **`cli.py`**: Command-line interface

### 2. Iteration Tools

**Notebooks:** `notebooks/explore_results.ipynb`
- Load results from store
- Summary statistics
- Accuracy by arm and N
- Refutation criteria evaluation
- Budget analysis
- Proposer log inspection

**Dashboard:** MLflow integration via native DSPy support
- Launch: `python3 experiments/mlflow_utils.py ui`
- Export: `python3 experiments/mlflow_utils.py export results`

### 3. Robustness Audit

**Doc:** `docs/dspy_api_audit.md`

Key findings:
- ✓ `instruction_proposer` used correctly (verified against DSPy 3.4.1 source)
- ✓ `reflection_lm` provided correctly
- ✓ Metric returns `dspy.Prediction(score, feedback)` as expected
- ✓ SIMBA configured with `metric`, `prompt_model`, `max_demos=0`
- ✓ Pydantic models work in DSPy 3.4+ without TypeSafe
- ✓ NOT reimplementing what DSPy provides
- ✓ NOT misusing any APIs

Risks identified:
- Format failures if LM returns malformed JSON (tracked in logs)
- Budget accounting approximation (5% tolerance, actuals reported)
- Pydantic schema overhead (measured post-hoc)

### 4. Tests

**Tests:** `tests/test_api_contracts.py`, `tests/test_experiment.py`

- API contract tests verify DSPy/GEPA signatures
- Experiment tests structured (DummyLM formatting needs work for DSPy 3.4.1)
- All API contracts pass

**Run tests:**
```bash
pytest tests/test_api_contracts.py -v
```

## Repository Layout

```
/workspace/
├── experiments/          # Experiment harness
│   ├── __init__.py
│   ├── arms.py          # A0-F definitions with proposers
│   ├── config.py        # Config and budget tracking
│   ├── runner.py        # Main experiment runner
│   ├── task.py          # IFBench task (simplified)
│   ├── store.py         # JSONL results storage
│   ├── analysis.py      # Bootstrap and R1-R5
│   ├── mlflow_utils.py  # MLflow integration
│   ├── cli.py           # CLI
│   └── README.md        # Full usage guide
├── notebooks/
│   └── explore_results.ipynb  # Exploration notebook
├── docs/
│   └── dspy_api_audit.md      # API verification
├── tests/
│   ├── test_experiment.py     # Smoke tests (structure in place)
│   └── test_api_contracts.py  # API contract tests (✓ pass)
├── requirements.txt           # Pinned dependencies
└── uploads/
    ├── theory-driven-experiment.tex  # Research design
    └── popperian_gepa_skeleton.py   # Original skeleton
```

## How to Run

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Run API Contract Tests

```bash
pytest tests/test_api_contracts.py -v
```

### Run Experiment (Requires Model API Key)

```bash
# Option 1: Cloud provider
export OPENROUTER_API_KEY='your-key'
python3 experiments/cli.py \
  --arms A0 A E C B F \
  --n 10 25 50 100 \
  --seeds 0 1 2 3 4 \
  --task-model openrouter/qwen/qwen3-8b \
  --reflection-model openrouter/qwen/qwen3-8b

# Option 2: Self-hosted
# (start vLLM server separately)
python3 experiments/cli.py --arms A0 A --n 10 --seeds 0 \
  --task-model openai/Qwen3-8B \
  --reflection-model openai/Qwen3-8B
```

### Explore Results

```bash
jupyter notebook notebooks/explore_results.ipynb
```

### View in MLflow

```bash
python3 experiments/mlflow_utils.py export results
python3 experiments/mlflow_utils.py ui
# Open http://127.0.0.1:5000
```

## Audit Highlights

See `docs/dspy_api_audit.md` for full details.

**Extension points used correctly:**
1. `dspy.GEPA(instruction_proposer=...)` ✓
2. `dspy.GEPA(reflection_lm=...)` ✓  
3. `dspy.GEPA(metric=...)` returns `dspy.Prediction(score, feedback)` ✓
4. `dspy.SIMBA(metric=..., prompt_model=..., max_demos=0)` ✓
5. Pydantic `list[Refutation]` in signatures ✓

**Not reimplemented:**
- GEPA's Pareto sampling, candidate generation, acceptance logic ✓
- DSPy primitives (`dspy.Predict`, signature handling, LM routing) ✓

**Added:**
- Custom proposers (critic → rewrite → survival check)
- Refutation archive (regression tests)
- Budget tracking (post-hoc from logs for now)

## What Still Needs Work

1. **DummyLM smoke tests:** DSPy 3.4.1 output formatting needs refinement for full offline tests
2. **Real IFBench data:** Current task.py uses dummy data
3. **Budget wrapper:** Simplified for now - tracks from DSPy logs post-hoc
4. **Real runs:** No paid model calls yet - everything is plumbing

## Honest Assessment

**What works:**
- All arm definitions are complete and match the research design
- API usage verified against installed DSPy/GEPA source
- Configuration, storage, analysis modules ready
- Notebook and MLflow integration in place
- API contract tests pass

**What needs more work:**
- DummyLM formatting for offline smoke tests (DSPy 3.4.1 expects specific JSON)
- Real IFBench data integration
- Full end-to-end test with tiny real model

**Status:** Plumbing complete and verified. Ready for real data and real models.

## Cost Estimate (from Design)

- Pilot: ~$10 (12 runs)
- Main grid: ~$82 (100 runs)
- Optional arm D: ~$16 (20 runs)
- **Total: ~$110** (or $15-30 self-hosted)

## No Secrets Added

Hard constraint honored: no API keys or secrets anywhere in this work. Real-provider runs documented as opt-in via environment variables.

## References

- Research design: `uploads/theory-driven-experiment.tex`
- Working skeleton: `uploads/popperian_gepa_skeleton.py`
- DSPy 3.4.1: installed source verified
- GEPA 0.1.4: installed source verified
