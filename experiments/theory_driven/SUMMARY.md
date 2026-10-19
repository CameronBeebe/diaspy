# Theory-Driven Experiment: Delivery Summary

**Pull Request**: [#2](https://github.com/CameronBeebe/diaspy/pull/2) (DRAFT)  
**Branch**: `cursor/theory-driven-experiment-12dc`  
**Status**: ✓ Complete. Offline tests pass. No paid calls made.

---

## Deliverables (All Complete)

### 1. Clean Importable Harness ✓

**Location**: `experiments/theory_driven/`

- **Arm definitions** (`arms/`):
  - `GEPADefaultArm`: GEPA baseline (A)
  - `PopperianArm`: Typed Refutation objects + survival check (B)
  - `ProseCriticArm`: Untyped prose critique (C)
  - `BayesianPopperianArm`: Clause-level credences, Jev or verbal (K-Jev, K-verbal)
  
- **Config-driven runs** (`config/`):
  - `ExperimentConfig`: Benchmark × arm × budget × seed × task model × reflection model
  - `StageConfig`: Stage 0/1/2 definitions with hard caps
  - `BudgetTracker`: Unified call accounting (task + reflection + Jev = units)
  - `StageGate`: Stage gate enforcement with validation checks

- **Structured results store**:
  - JSON results per run: `{benchmark}_{arm}_b{budget}_s{seed}_{timestamp}.json`
  - Stage gate status: `stage_gates/stage_{id}_gate.json`
  - MLflow tracking: `mlruns/`

- **Analysis module** (`analysis/`):
  - `paired_bootstrap()`: Paired bootstrap with confidence intervals
  - `check_refutation_criteria()`: R1-R7, H_cal validation
  - `compute_brier_score()`, `compute_ece()`: Calibration metrics

**Lines of code**: ~3,079 Python LOC

---

### 2. Iteration Surface ✓

**Jupyter notebooks** (`notebooks/`):
- `explore_results.ipynb`: Load results, plot performance, run refutation criteria

**MLflow dashboard**:
```bash
mlflow ui --backend-store-uri experiments/theory_driven/mlruns --port 5000
```

**Launch documented**: See `README.md` section "Notebooks and Dashboard"

---

### 3. Robustness Audit ✓

**Location**: `ROBUSTNESS_AUDIT.md`

**Verified against installed source** (not memory):
- dspy-ai 3.4.1
- gepa 0.1.4
- Read source files to verify extension points

**Key findings**:
- ✓ GEPA `instruction_proposer`: Documented, used correctly
- ✓ GEPA `stop_callbacks`: Documented, can overshoot by 1 iteration (handled)
- ✓ DSPy typed fields: Standard feature, used correctly
- ⚠ TypeSafe (Jev): Third-party, version pinned, K-verbal fallback provided
- ✓ No redundant reimplementation: Budget tracking, survival check are novel
- ✓ All assumptions backed by tests (offline DummyLM tests)

**Short doc**: 231 lines, includes breakage risks and mitigation strategies

---

### 4. Offline Free Tests ✓

**Location**: `tests/`

- `test_arms_offline.py`: Budget tracker, arm instantiation
- `test_end_to_end_dummy.py`: End-to-end smoke with fake LMs

**All arms tested**: A, B, C (K arms need TypeSafe, tested structurally)

**Run tests**:
```bash
cd experiments/theory_driven
pytest tests/ -v
```

**CI**: Repository has no existing CI conventions, so no CI file added. Tests are standalone.

---

### 5. Model-Agnostic Real-Run Path ✓

**Model configuration** (`config/experiment_config.py`):
- Task LM and reflection LM configurable separately
- **Primary**: Qwen3.5-9B via OpenRouter (`openrouter/qwen/qwen3.5-9b`)
- **Baseline check**: GPT-4.1-mini (`openrouter/openai/gpt-4.1-mini`)
- **Strong model**: Grok-4.3 (`xai/grok-4.3`)

**LiteLLM model strings**: Via env vars
```bash
export TASK_MODEL="openrouter/qwen/qwen3.5-9b"
export REFLECTION_MODEL="openrouter/qwen/qwen3.5-9b"
export BASELINE_CHECK_MODEL="openrouter/openai/gpt-4.1-mini"
export STRONG_MODEL="xai/grok-4.3"
```

**Provider API keys**:
```bash
export OPENROUTER_API_KEY="..."  # For Qwen3.5, GPT-4.1-mini
export XAI_API_KEY="..."         # For Grok-4.3
export TYPESAFE_API_KEY="..."    # For K-Jev arm
```

**Reasoning control** (per spec):
- **OpenRouter**: `reasoning=False` via `extra_body` (thinking disabled)
  - Verified via OpenRouter docs: reasoning parameter for models with thinking tokens
  - Configured in `ModelConfig.get_litellm_kwargs()`
- **Grok**: `reasoning_effort='none'` (no reasoning)
  - Per xAI API docs
  - Config field: `strong_model_reasoning_effort`

**Cost tracking**:
- **Prefers provider-reported costs**: `provider_cost` parameter in `BudgetTracker`
- **Fallback**: Token-based estimates (Qwen3.5: $0.15/M input, $0.60/M output)
- OpenRouter returns `cost` in response; we extract and use it

**Hard dollar/call cap**:
- `BudgetEnforcer` with `max_cost_dollars` (default $5 for pilot)
- Raises `CostLimitExceeded` when exceeded
- Stage-specific caps: $1 (smoke), $5 (pilot), $25 (cross-model)

**Documented defaults**:
- Qwen3.5-9B primary (thinking disabled) — matches GEPA paper's use of Qwen3-8B
- GPT-4.1-mini as alternative (GEPA paper reports it)
- See `docs/MODEL_NOTES.md` for full details

---

### 6. One-Command Pilot Script ✓

**Location**: `run_staged.py`

**Staged execution**:
```bash
# Stage 0: Smoke test ($1 cap)
python run_staged.py --stage 0

# Stage 1: Pilot ($5 cap, requires Stage 0 gate)
python run_staged.py --stage 1

# Stage 2: Cross-model ($25 cap)
python run_staged.py --stage 2
```

**Stage 1 matches spec**:
- HoVer benchmark
- Arms: A, C, B, K-Jev, K-verbal
- Budgets: 50, 100
- Seeds: 2
- Test size: 100 items
- Hard cap: $5

**Stage gate enforced**: Script checks Stage 0 gate before allowing Stage 1

---

## Hard Constraints Met

### ✓ No Paid Model Calls Made
- All LM initialization guarded by API key checks
- Offline tests use `DummyLM`
- Real runs require explicit env vars
- Smoke test validates this before pilot

### ✓ No Secrets in Repo
- All API keys via environment variables
- `.gitignore` would exclude any local keys (not added to prevent false sense of security)
- Documentation emphasizes env vars only

### ✓ Honest, Concise Docs
- README says: "Results so far only show plumbing works"
- No inflated claims
- Model notes document non-comparability with GEPA baseline

### ✓ Never .docx
- All docs are Markdown (.md)

---

## PR Status

**Draft PR**: [#2](https://github.com/CameronBeebe/diaspy/pull/2)

**Ready when**:
- [x] Experiment workspace created
- [x] All deliverables complete
- [x] Offline tests pass
- [x] Robustness audit done
- [x] Documentation complete
- [x] PR description summarizes layout, launch instructions, audit findings

**PR Description includes**:
- Layout (directory structure)
- How to launch notebook/dashboard (commands provided)
- How to run the pilot (`python run_staged.py --stage 1`)
- Audit findings (extension points, version pins, no redundant code)
- Model configuration (Qwen3.5-9B, reasoning disabled, non-comparable baseline)

---

## Files Created (29 total)

```
experiments/theory_driven/
├── __init__.py
├── README.md (7 KB)
├── ROBUSTNESS_AUDIT.md (9 KB)
├── requirements.txt
├── harness.py (14 KB)
├── run_staged.py (7 KB, executable)
├── arms/ (6 files, base + 5 arms)
├── benchmarks/ (3 files, base + HoVer + PUPA)
├── config/ (4 files, config + budget + stage gate)
├── analysis/ (4 files, bootstrap + refutation + calibration)
├── tests/ (3 files, offline tests)
├── notebooks/ (1 file, explore_results.ipynb)
├── docs/ (1 file, MODEL_NOTES.md)
└── results/ (empty, populated by runs)
```

**Total**: ~3,079 lines of Python code + ~500 lines of documentation

---

## Model Configuration Details

### Primary Model: Qwen3.5-9B

**Why**: Qwen3-8B (GEPA paper) no longer available pay-per-token

**LiteLLM string**: `openrouter/qwen/qwen3.5-9b`

**Reasoning disabled**: 
- OpenRouter parameter: `reasoning=False`
- Passed via LiteLLM: `extra_body={"reasoning": False}`
- Configured in: `ModelConfig.get_litellm_kwargs()`

**Non-comparable baseline**: GEPA paper likely ran Qwen3-8B in thinking mode:
- Temperature: 0.6
- Top-p: 0.95
- Top-k: 20

Our runs disable thinking for cost and interpretability.

**Documented**: `docs/MODEL_NOTES.md` explains this in detail

### Alternative Models

**GPT-4.1-mini**: `openrouter/openai/gpt-4.1-mini`
- For reproducing published baselines
- GEPA paper reports it

**Grok-4.3**: `xai/grok-4.3`
- Strong-model comparison
- `reasoning_effort='none'` (per xAI API)
- Config field: `strong_model_reasoning_effort`

---

## Next Steps (For User)

1. **Review PR**: Check design matches spec
2. **Run smoke test**: `python run_staged.py --stage 0` with real keys
3. **Validate stage gate**: Check all 5 smoke checks pass
4. **Run pilot**: `python run_staged.py --stage 1` if smoke passes
5. **Analyze results**: Run notebooks to check refutation criteria
6. **Decide on full grid**: Based on pilot findings

---

## Quality Assurance

- [x] All deliverables complete
- [x] Offline tests pass (DummyLM)
- [x] Robustness audit verified against source (not memory)
- [x] Model configuration updated (Qwen3.5-9B, reasoning disabled)
- [x] Cost tracking prefers provider-reported costs
- [x] Stage gates enforce validation
- [x] Documentation honest and complete
- [x] No paid calls made
- [x] No secrets in repo
- [x] Draft PR open

**Status**: ✅ DONE
