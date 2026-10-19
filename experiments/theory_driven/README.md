# Theory-Driven Low-Budget Experiment

**Status**: DRAFT - Critical bugs fixed Oct 19 2026, not tested end-to-end.

See `PR_DESCRIPTION.md` for full status and what was fixed.

---

Testing whether theory-driven optimizers beat GEPA's default reflection at LOW optimization budgets (50/100/200 total calls).

## Overview

This experiment implements the design from `theory-driven-experiment.tex`:

- **Hypothesis**: At budgets of 50-200 calls, budget spent on theory-driven critique beats budget spent on re-scoring candidates
- **Primary benchmark**: HoVer (deterministic recall, local BM25S retrieval)
- **Secondary benchmark**: PUPA (LLM judge)
- **Arms**:
  - A: GEPA default
  - B: Typed Popperian proposer (Refutation objects + survival check)
  - C: Untyped prose critic
  - K-Jev: Clause-level Bayesian/Popperian with Jev credences (TypeSafe)
  - K-verbal: Clause-level with verbalized credences
  - E, F: Full grid additions (check-only, SIMBA)

## Quick Start

### Installation

```bash
cd /workspace
pip install -e .
cd experiments/theory_driven
pip install -r requirements.txt
```

### Offline Tests

Run offline tests with DummyLM (no API calls):

```bash
pytest tests/ -v
```

### Staged Execution

The experiment runs in stages, each with a hard cost cap:

#### Stage 0: Smoke Test ($1 cap)

End-to-end validation with 1 benchmark, 2 arms, 1 seed, 20-item test:

```bash
python run_staged.py --stage 0
```

This validates:
- Call accounting matches provider reports
- Cost cap aborts correctly
- Results/dashboard populate
- Metrics parse

**Stage gate**: Stage 1 cannot run until Stage 0 passes all checks.

#### Stage 1: Pilot ($5 cap)

Full pilot after smoke test passes:

```bash
python run_staged.py --stage 1
```

- HoVer benchmark
- Arms: A, C, B, K-Jev, K-verbal
- Budgets: 50, 100
- Seeds: 2
- Test size: 100 items

#### Stage 2: Cross-Model ($25 cap)

Test on multiple models (optional):

```bash
python run_staged.py --stage 2
```

Tests pilot configuration on 2-3 cheap models (Qwen3-8B, GPT-4o-mini, Claude-3-Haiku).

### Configuration

Model and provider settings via environment variables:

```bash
# Primary model: Qwen3.5-9B via OpenRouter (thinking disabled)
export OPENROUTER_API_KEY="your-openrouter-key"
export TASK_MODEL="openrouter/qwen/qwen3.5-9b"  # Default
export REFLECTION_MODEL="openrouter/qwen/qwen3.5-9b"  # Default
export ENABLE_REASONING="false"  # Disable thinking tokens (default)

# Alternative models
export BASELINE_CHECK_MODEL="openrouter/openai/gpt-4.1-mini"  # For baseline check
export STRONG_MODEL="xai/grok-4.3"  # Strong-model comparison
export XAI_API_KEY="your-xai-key"  # For Grok
export STRONG_MODEL_REASONING_EFFORT="none"  # For Grok

# TypeSafe for K-Jev arm
export TYPESAFE_API_KEY="your-jev-key"

# Cost controls
export MAX_COST_DOLLARS="5.0"

# OR use local vLLM endpoint
export VLLM_ENDPOINT="http://localhost:8000/v1"
```

**Note**: Qwen3-8B (used in GEPA paper) is no longer available pay-per-token. We use Qwen3.5-9B with thinking disabled. GEPA paper likely ran Qwen3-8B in thinking mode (temp 0.6/top-p 0.95/top-k 20), so our baseline numbers are not directly comparable. See `docs/MODEL_NOTES.md` for details.

### Notebooks and Dashboard

Start JupyterLab for exploration:

```bash
jupyter lab notebooks/
```

Key notebooks:
- `explore_results.ipynb`: Load and visualize results
- `refutation_analysis.ipynb`: Run refutation criteria (R1-R7)
- `calibration_analysis.ipynb`: Brier/ECE for clause credences

Launch MLflow dashboard:

```bash
mlflow ui --backend-store-uri experiments/theory_driven/mlruns --port 5000
```

Then open http://localhost:5000

## Architecture

```
experiments/theory_driven/
├── arms/               # Optimizer implementations
│   ├── base.py
│   ├── gepa_default.py
│   ├── popperian.py    # Arm B: typed Refutation + survival check
│   ├── prose_critic.py # Arm C: untyped prose critique
│   └── bayesian_popperian.py  # Arms K: clause-level credences
├── benchmarks/         # Task implementations
│   ├── hover.py        # Primary: deterministic recall
│   └── pupa.py         # Secondary: LLM judge
├── config/             # Configuration and budget tracking
│   ├── experiment_config.py
│   ├── budget_tracker.py   # Unified call accounting
│   └── stage_gate.py       # Stage gate mechanism
├── analysis/           # Statistical analysis
│   ├── bootstrap.py         # Paired bootstrap
│   ├── refutation_criteria.py  # R1-R7, H_cal
│   └── calibration.py       # Brier/ECE
├── tests/              # Offline tests
│   ├── test_arms_offline.py
│   └── test_end_to_end_dummy.py
├── notebooks/          # Jupyter notebooks
├── harness.py          # Main experiment harness
├── run_staged.py       # Staged execution script
└── README.md          # This file
```

## Robustness Audit

See `ROBUSTNESS_AUDIT.md` for:
- DSPy/GEPA extension points used
- Version pins and breakage risks
- What we reimplement vs what DSPy provides
- Verification against installed source

Key findings:
- GEPA `instruction_proposer`: Custom proposal function (documented, verified)
- Budget tracker as `stop_callbacks`: Works, can overshoot by 1 iteration (documented)
- Popperian survival check: Our extension, not redundant
- TypeSafe for Jev: Third-party API, version pinned

## Cost Estimates

Approximate (Qwen3.5-9B, thinking disabled):

- **Smoke test**: ~$0.20 (actual cap $1)
- **Pilot**: ~$4.20 (cap $5)
- **Cross-model**: Variable by model (cap $25)
  - Qwen3.5-9B: ~$5
  - GPT-4.1-mini: ~$10
  - Grok-4.3: ~$10
- **Full grid**: ~$108 (cap $130)

**Cost tracking**: Prefers provider-reported costs from OpenRouter/xAI when available, falls back to token-based estimates.

Self-hosting with vLLM: ~$10-20 total.

## Results Interpretation

Once runs complete, see `notebooks/refutation_analysis.ipynb` for:

- **R1 (primary)**: Did K-Jev or B beat A by +3.0 points with CI > 0?
- **R2**: Does low budget favor theory-driven more than high budget?
- **R3**: Does typing (B) beat prose (C)?
- **H_cal**: Are Jev credences calibrated (Brier skill > 0)?

Per spec: "Hypothesis refuted if BOTH K-Jev and B fail R1."

## Development Notes

### Adding a New Arm

1. Subclass `BaseArm` in `arms/`
2. Implement `compile()` method
3. Register in `harness.create_arm()`
4. Add config to `ExperimentConfig.arms`
5. Add offline test in `tests/`

### Adding a New Benchmark

1. Subclass `BaseBenchmark` in `benchmarks/`
2. Implement `load_data()`, `get_program()`, `get_metric()`, `setup()`
3. Register in `harness.create_benchmark()`
4. Add to `ExperimentConfig.benchmarks`

## References

- Spec: `uploads/theory-driven-experiment.tex`
- Popperian skeleton: `uploads/popperian_gepa_skeleton.py`
- GEPA paper: Agrawal et al., arXiv:2507.19457
- DSPy: dspy-ai/dspy v3.4.1
- GEPA: gepa-ai/gepa v0.1.4

## License

Part of diaspy package. Copyright (c) 2025 The SciPhi Initiative, LLC. Proprietary.
