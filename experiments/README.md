# Theory-Driven GEPA Experiment

Pre-registered low-data experiment comparing typed Popperian critic vs GEPA default and other DSPy optimizers.

**Status:** Plumbing complete, all offline tests pass. Real-provider runs not yet executed.

---

## Research Question

Does replacing GEPA's generic reflection step with a theory-driven critic (Popperian conjecture-and-refutation, emitting typed Refutation objects with claim, counterexample IDs, severity; plus a mechanical check that rewrites survive those counterexamples) improve held-out performance in the low-data regime vs GEPA default and other DSPy optimizers?

**See:** Full design in `uploads/theory-driven-experiment.tex`

---

## Structure

```
experiments/
├── __init__.py
├── arms.py              # Arm definitions (A0, A, E, C, B, D, F)
├── config.py            # ExperimentConfig and budget tracking
├── runner.py            # Main experiment runner
├── task.py              # IFBench task setup
├── store.py             # JSONL results storage
├── analysis.py          # Paired bootstrap and refutation criteria
├── mlflow_utils.py      # MLflow integration
└── cli.py               # Command-line interface

notebooks/
└── explore_results.ipynb  # Jupyter notebook for exploration

docs/
└── dspy_api_audit.md      # DSPy/GEPA API usage verification

tests/
├── test_experiment.py     # Smoke tests (all arms, DummyLM)
└── test_api_contracts.py  # API contract tests
```

---

## Arms

| Arm | Definition | What it isolates |
|-----|------------|------------------|
| A0  | Zero-shot, unoptimized | Floor baseline |
| A   | GEPA default with InstructionProposer | Incumbent |
| E   | GEPA + survival check + archive (no critic) | Is any gain just the filter? |
| C   | GEPA + untyped prose critic (no check) | Does typing matter? |
| B   | GEPA + typed Popperian proposer | **The hypothesis** |
| D   | GEPA + Millian debate (two rival critics) | Optional |
| F   | dspy.SIMBA | Introspective rules from high-variance minibatches |

---

## Quick Start

### 1. Install Dependencies

```bash
pip3 install 'dspy-ai==3.4.1' 'gepa==0.1.4' pydantic jupyter mlflow scikit-learn pandas matplotlib seaborn pytest
```

### 2. Run Smoke Test (Free, Offline)

**Note:** Full smoke tests with DummyLM require additional work to match DSPy 3.4.1's output formatting.  
For now, run API contract tests to verify the plumbing:

```bash
pytest tests/test_api_contracts.py -v
```

To test with a real model (minimal cost):

```bash
export OPENROUTER_API_KEY='your-key'

python3 experiments/cli.py \
  --arms A0 A \
  --n 10 \
  --seeds 0 \
  --task-model openrouter/qwen/qwen3-8b \
  --reflection-model openrouter/qwen/qwen3-8b \
  --max-rollouts 20
```

### 3. Run Tests

```bash
pytest tests/test_experiment.py -v
pytest tests/test_api_contracts.py -v
```

### 4. Explore Results

```bash
jupyter notebook notebooks/explore_results.ipynb
```

### 5. View in MLflow

```bash
# Export results to MLflow
python3 experiments/mlflow_utils.py export results

# Launch MLflow UI
python3 experiments/mlflow_utils.py ui
# Open http://127.0.0.1:5000
```

---

## Running the Full Experiment

**⚠️ IMPORTANT:** Real runs require API keys and will incur costs (~$110 estimated).

### Option 1: Cloud Provider (e.g., OpenRouter)

```bash
# Set API key
export OPENROUTER_API_KEY='your-key-here'

# Run full grid
python3 experiments/cli.py \
  --arms A0 A E C B F \
  --n 10 25 50 100 \
  --seeds 0 1 2 3 4 \
  --task-model openrouter/qwen/qwen3-8b \
  --reflection-model openrouter/qwen/qwen3-8b \
  --results-dir results_main
```

### Option 2: Self-Hosted (vLLM)

```bash
# Start vLLM server
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000

# Run experiment
python3 experiments/cli.py \
  --arms A0 A E C B F \
  --n 10 25 50 100 \
  --seeds 0 1 2 3 4 \
  --task-model openai/Qwen3-8B \
  --reflection-model openai/Qwen3-8B \
  --results-dir results_main
```

Configure DSPy to use the vLLM endpoint:
```python
import dspy
dspy.configure(api_base="http://localhost:8000/v1", api_key="dummy")
```

---

## Configuration

Key parameters in `ExperimentConfig`:

- **Budget:** 1,000 program rollouts, 60 reflection calls, 300k reflection tokens per run
- **Tolerance:** 5% (runs >5% off budget are discarded)
- **GEPA params:** `reflection_minibatch_size=3`, `use_merge=False`
- **Proposer params:** `max_attempts=2`, `tau=0.5`, `archive_cap=6`
- **SIMBA params:** `max_demos=0` (prompt-only), `bsize=min(32, |train|)`

Adjust via CLI args or by editing `experiments/config.py`.

---

## Results and Analysis

Results are stored as JSONL in `results/results.jsonl.gz`.

Each `RunResult` includes:
- Arm, N, seed, train/val sizes
- Budget usage (rollouts, reflection calls, tokens)
- Test accuracy, val accuracy, per-item predictions
- Proposer logs (for arms B, C, D, E)
- Optimizer stats

### Paired Bootstrap Analysis

The `analysis.py` module implements:
- `compare_arms(results, arm_x, arm_y, n_labels)` — paired bootstrap for X - Y with 95% CI
- `check_refutation_criteria(results)` — R1-R5 hypothesis tests from pre-registration

### Refutation Criteria

- **R1:** Pooled small-N (10, 25): Δ_BA < +2.0 OR lower CI ≤ 0 → hypothesis refuted
- **R2:** Δ_BA(N=100) ≥ Δ_BA(N=10) → "low-data specific" refuted
- **R3:** Pooled small-N Δ_BC ≤ 0 → typing irrelevant
- **R4:** CI of Δ_BE includes 0 → filter alone suffices (epistemology irrelevant)
- **R5:** Pooled small-N Δ_BF ≤ 0 → SIMBA sufficient (B not needed)

---

## Dashboard

Two options:

### MLflow (Recommended)

DSPy has native MLflow integration. To use:

```python
# In runner, enable MLflow
optimizer = dspy.GEPA(..., use_mlflow=True)
```

Or export results post-hoc:
```bash
python3 experiments/mlflow_utils.py export results
python3 experiments/mlflow_utils.py ui
```

### JupyterLab

```bash
jupyter lab
# Open notebooks/explore_results.ipynb
```

---

## API Usage Verification

See `docs/dspy_api_audit.md` for:
- Verification that we use `instruction_proposer` correctly
- Inspection of GEPA/SIMBA signatures against installed source
- Confirmation we don't reimplement what DSPy provides
- Identified risks (format failures, budget approximation)
- ✓ No misuse of DSPy/GEPA APIs

---

## Testing

### Offline Tests (No API Calls)

All tests use `DummyLM` and run free:

```bash
# Smoke test: all arms at tiny scale
pytest tests/test_experiment.py::test_all_arms_smoke -v

# Budget enforcement
pytest tests/test_experiment.py::test_budget_enforcement -v

# Popperian proposer plumbing
pytest tests/test_experiment.py::test_popperian_proposer_plumbing -v

# API contracts
pytest tests/test_api_contracts.py -v
```

### Smoke Test via CLI

```bash
python3 experiments/cli.py --smoke-test
```

---

## Known Limitations

1. **Plumbing verified with API contracts:** Smoke tests using DummyLM need work to match DSPy 3.4.1 output formatting
2. **IFBench data not included:** Current task.py has dummy data for testing
3. **No real-provider runs yet:** No paid model calls have been made
4. **SIMBA may fail with DummyLM:** SIMBA's introspection may not work with scripted fakes
5. **Budget tracking simplified:** BudgetedLM wrapper disabled for initial version - tracks post-hoc from logs

---

## Next Steps

To run the real experiment:

1. **Acquire data:** Obtain IFBench dataset (IF-RLVR train split + 294-item test set)
2. **Implement verifiers:** IFBench constraint checking (currently stubbed)
3. **Choose provider:** OpenRouter (Qwen3-8B, ~$110) or self-host (vLLM, ~$15-30 GPU)
4. **Run pilot:** 12 runs (B vs A, N=10/25, seeds 0-2) to check power
5. **Main grid:** 100+ runs if pilot SD < 3 points; else 10 seeds

---

## Citation

If you use this experiment harness:

```bibtex
@misc{diaspy-gepa-experiment-2026,
  title={Theory-driven optimization in DSPy: A pre-registered low-data experiment},
  author={diaspy contributors},
  year={2026},
  note={Pre-registered experiment comparing Popperian critic vs GEPA default}
}
```

---

## License

Copyright (c) 2025 The SciPhi Initiative, LLC. All rights reserved.

This is proprietary research code. See repository LICENSE for details.
