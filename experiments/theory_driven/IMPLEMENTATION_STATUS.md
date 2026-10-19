# Implementation Status

**Last updated**: 2026-10-19  
**Branch**: cursor/theory-driven-experiment-12dc  
**PR**: #2 (DRAFT)

## Completed ✓

### Core Infrastructure
- [x] Experiment harness with config-driven runs
- [x] Budget tracker (unified call accounting across task/reflection/Jev)
- [x] Staged execution (Stage 0/1/2 with hard cost caps)
- [x] Stage gate mechanism
- [x] Results storage (JSON per run)

### Arms
- [x] Arm A: GEPA default
- [x] Arm B: Typed Popperian (Refutation objects + survival check)
- [x] Arm C: Untyped prose critic
- [x] Arm K-Jev: Clause-level Bayesian/Popperian (structure only, needs TypeSafe)
- [x] Arm K-verbal: Clause-level with verbalized credences

### Benchmarks
- [x] HoVer (structure, dummy data)
- [x] PUPA (structure, dummy data)
- [x] Metric interfaces

### Analysis
- [x] Paired bootstrap
- [x] Refutation criteria (R1-R7, H_cal)
- [x] Brier/ECE calibration metrics

### Provenance & Drift (Added 2026-10-19)
- [x] ProvenanceLogger: Full per-call trace logging
  - Prompt, completion, params, timestamps
  - Response model/provider/usage/cost
  - OpenRouter generation IDs
  - Provider routing config
- [x] Provider pinning (OpenRouter)
  - `provider.order` and `provider.allow_fallbacks=false`
  - Configured via ModelConfig
- [x] Canary suite for drift detection
  - 8 fixed prompts at temperature 0
  - Hash comparison with stored reference
  - Similarity scoring
  - Blocks stage on drift (configurable)
  - Free-tier model support for stage 0 (Gemma-2-9B-it:free)

### Pre-Flight Gate (Added 2026-10-19)
- [x] Mandatory pre-flight checks before paid stages:
  - [x] Code quality (ruff + pyright/mypy)
  - [x] Full offline test suite
  - [x] Known-answer tests (metric, budget, cap)
  - [x] Arm distinctness tests
  - [x] Data integrity tests (splits, seeds, test isolation)
  - [x] Dry-run (planned calls and cost)
  - [x] --confirm flag requirement
- [x] Gate results file integration with stage gate

### Testing
- [x] Offline tests with DummyLM (test_arms_offline.py)
- [x] End-to-end smoke test (test_end_to_end_dummy.py)
- [x] Known-answer tests (test_known_answers.py)
- [x] Arm distinctness tests (test_arm_distinctness.py)
- [x] Data integrity tests (test_data_integrity.py)

### Documentation
- [x] README with usage instructions
- [x] ROBUSTNESS_AUDIT.md (needs revision, see Known Gaps)
- [x] MODEL_NOTES.md (model configuration)
- [x] PACKAGE_VERIFICATION.md (honest package status)
- [x] IMPLEMENTATION_STATUS.md (this file)

### Dashboards
- [x] MLflow integration (configured, not tested)
- [x] Jupyter notebook template (explore_results.ipynb)

## Partially Complete ⚠

### Model Configuration
- [x] Qwen3.5-9B configuration
- [x] OpenRouter reasoning control
- [x] xAI Grok configuration
- [x] Provider pinning
- [ ] **Verified against actual APIs** (cannot test without paid calls)

### Provenance
- [x] Trace logging infrastructure
- [ ] **Integration with DSPy callbacks** (needs DSPy API verification)
- [ ] **MLflow span creation** (structure in place, not tested)
- [ ] **OpenRouter metadata fetching** (async function defined, not tested)

### Benchmarks
- [x] Structure and interfaces
- [ ] **Real data loading** (using dummy data currently)
- [ ] **BM25S retrieval** (not installed/tested)
- [ ] **PUPA judge implementation** (structure only)

## Not Yet Implemented ✗

### Epistemic Event Traces
- [ ] Pydantic schema for typed events (lineage, conjecture diff, etc.)
- [ ] Per-optimization-step event emission
- [ ] JSONL storage linked to raw traces
- [ ] MLflow span linkage
- [ ] Notebook replay (timeline + lineage tree)
- [ ] Cross-arm diff visualization
- [ ] Test that every arm emits well-formed events

### Wording Ablation Arm
- [ ] Neutral-wording Popperian variant
- [ ] No philosophical vocabulary
- [ ] Same structure as Arm B

### Full Grid Arms
- [ ] Arm E: Check-only (no critic)
- [ ] Arm F: SIMBA

### Known Gaps from Original Audit

#### Package Verification
- [ ] **DSPy version**: Claimed 3.4.1 does NOT exist
  - Actual latest: 2.5.29
  - Need to verify GEPA availability in this version
- [ ] **GEPA package**: May not be standalone pip package
  - Need to verify distribution method
- [ ] **TypeSafe/Jev**: Package name/distribution unclear
  - May be API-only, not pip package
  - K-Jev arm may need different implementation

#### Real Data & Benchmarks
- [ ] Actual HoVer data (5.2M Wikipedia abstracts)
- [ ] BM25S index building and memory check
- [ ] PUPA data loading from Columbia-NLP
- [ ] IFBench (optional)

#### Real LM Integration
- [ ] DSPy callback/history integration for provenance
- [ ] Token count extraction from responses
- [ ] Cost extraction from OpenRouter responses
- [ ] Generation ID extraction
- [ ] Error handling for API failures

#### Baseline Reproduction
- [ ] GEPA paper baseline reproduction at published splits
- [ ] Parity check (±2 SE)

#### Cost Reconciliation
- [ ] Fetch OpenRouter generation metadata
- [ ] Compare tracked vs reported costs
- [ ] Log discrepancies

## Honest Assessment

### What Works (Offline)
- Structure and interfaces are complete
- Offline tests pass with DummyLM
- Configuration system is flexible
- Pre-flight gate enforces quality checks
- Provenance and canary infrastructure in place

### What's Uncertain
- **DSPy/GEPA versions**: Audit claimed versions that may not exist
- **TypeSafe integration**: K-Jev arm needs verification
- **Real data loading**: Using dummy data, not verified against actual datasets
- **LM integration**: Provenance logging needs DSPy callback integration
- **Baseline comparability**: Qwen3.5-9B with thinking disabled is not comparable to GEPA paper's Qwen3-8B results

### What's Missing
- Epistemic event traces (significant feature)
- Wording ablation arm
- Full grid arms (E, F)
- Real data pipeline verification
- End-to-end test with real (cheap) model

## Recommendations Before First Paid Run

1. **Verify packages**:
   - Test install dspy-ai (actual version)
   - Confirm GEPA availability
   - Determine TypeSafe/Jev approach

2. **Test with free model**:
   - Run smoke test with Gemma-2-9B-it:free
   - Verify provenance logging works
   - Confirm canary suite runs

3. **Implement remaining tests**:
   - Add epistemic event tests
   - Add wording ablation arm
   - Verify full pre-flight gate runs

4. **Update audit**:
   - Correct version numbers
   - Remove "novel" claims unless backed by evidence
   - List all known gaps honestly

5. **Document workarounds**:
   - If GEPA not in dspy-ai, document install
   - If TypeSafe not available, document K-verbal as primary

## Test Counts (Actual)

- **Unit tests**: 17 (5 in test_arms_offline, 5 in test_known_answers, 5 in test_arm_distinctness, 2 in test_data_integrity)
- **Integration tests**: 1 (test_end_to_end_dummy)
- **Pre-flight checks**: 6 (code quality, offline tests, known answers, arm distinctness, data integrity, dry-run)
- **Total test assertions**: ~50+

All tests run offline with DummyLM. No real LM calls made.
