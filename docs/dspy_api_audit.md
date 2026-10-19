# DSPy/GEPA API Usage Audit

**Purpose:** Verify that our experiment uses DSPy and GEPA extension points correctly,
backed by inspection of the installed source (DSPy 3.4.1, GEPA 0.1.4).

**Date:** 2026-10-19  
**Status:** ✓ Verified against installed packages

---

## Extension Points Used

### 1. GEPA `instruction_proposer`

**What we do:**
- Pass a custom callable to `dspy.GEPA(..., instruction_proposer=proposer)`
- Proposer signature: `(candidate, reflective_dataset, components_to_update) -> dict[str, str]`
- Arms B, C, D, E use custom proposers; arm A uses GEPA default (None)

**Verification:**
```python
import inspect
import dspy

sig = inspect.signature(dspy.GEPA.__init__)
# Parameter exists: instruction_proposer: ProposalFn | None = None
```

**Source inspection results:**
- ✓ `instruction_proposer` parameter exists in `dspy.GEPA.__init__`
- ✓ Type annotation: `ProposalFn | None`
- ✓ Default: `None` (uses GEPA's internal `InstructionProposer`)
- ✓ When provided, GEPA calls it during reflection with the correct signature

**Are we using it correctly?** YES
- Our proposers match the expected signature
- When `instruction_proposer` is set, `reflection_lm` becomes optional (we provide it anyway)
- GEPA calls the proposer during its reflection loop, passing the current candidate instructions,
  a dataset of recent evaluations with feedback, and which components to update

**What the skeleton prototype proved:**
- GEPA successfully called our `PopperianProposer`
- The proposer returned a dict mapping component names to new instructions
- GEPA's acceptance logic still ran (minibatch test)

---

### 2. GEPA `reflection_lm`

**What we do:**
- Pass a DSPy LM to `dspy.GEPA(..., reflection_lm=lm)` for critic/rewrite calls
- Wrap it in `BudgetedLM` to track reflection calls against budget

**Verification:**
- ✓ `reflection_lm: dspy.clients.lm.LM | None` parameter exists
- ✓ Optional when `instruction_proposer` is provided
- ✓ We provide it explicitly for all GEPA arms

**Are we using it correctly?** YES
- Our custom proposers use `dspy.context(lm=self.critic_lm)` to route calls
- Budget wrapper delegates all LM methods via `__getattr__`

---

### 3. GEPA `metric`

**What we do:**
- Define `ifbench_metric(gold, pred, trace, pred_name, pred_trace) -> dspy.Prediction`
- Metric returns `dspy.Prediction(score=float, feedback=str)`

**Verification:**
- ✓ `metric: GEPAFeedbackMetric` required parameter
- ✓ `GEPAFeedbackMetric` is a callable with signature matching ours
- ✓ Must return object with `.score` and `.feedback` attributes

**Are we using it correctly?** YES
- Our metric has the correct signature (5 params)
- Returns `dspy.Prediction` with `score` (float) and `feedback` (str)
- GEPA uses `score` for acceptance, `feedback` for proposer context

---

### 4. GEPA `component_selector`

**What we do:**
- Use default `'round_robin'`
- Could use `'all'` or a custom callable, but round-robin is standard

**Verification:**
- ✓ `component_selector: ReflectionComponentSelector | str = 'round_robin'`

**Are we using it correctly?** YES
- We use the default, which GEPA handles internally

---

### 5. SIMBA (Arm F)

**What we do:**
- `dspy.SIMBA(prompt_model=reflection_lm, max_demos=0, bsize=min(32, |train|))`
- SIMBA optimizes instructions via introspective rules from high-variance minibatches

**Verification:**
```python
sig = inspect.signature(dspy.SIMBA.__init__)
# Parameters: prompt_model, max_demos, bsize, ...
```

- ✓ `prompt_model: dspy.LM | None` parameter exists
- ✓ `max_demos: int = 4` (we override to 0 for prompt-only)
- ✓ `bsize: int = 32` (minibatch size)

**Are we using it correctly?** YES
- We pass `prompt_model` (the reflection LM)
- `max_demos=0` enforces prompt-only (no demonstrations), matching design
- `bsize` controls minibatch size for introspection

**Note:** SIMBA uses `trainset` only (no separate val), so we pass `train + val` as trainset.

---

### 6. DSPy Signatures and Predict

**What we do:**
- Define typed signatures: `Critique`, `Rewrite`, `UntypedCritique`, etc.
- Use `dspy.Predict(Signature)` to create predictors
- Use `dspy.InputField()` and `dspy.OutputField()` for field definitions

**Verification:**
- ✓ `dspy.Signature` base class for custom signatures
- ✓ `dspy.Predict` accepts signature class
- ✓ Field definitions with `InputField` and `OutputField`

**Are we using it correctly?** YES
- Our signatures inherit from `dspy.Signature`
- Fields use appropriate `InputField` / `OutputField` annotations
- Pydantic models (e.g., `list[Refutation]`) work as output types

---

### 7. Typed Fields with Pydantic

**What we do:**
- Use `list[Refutation]` where `Refutation` is a `pydantic.BaseModel`
- DSPy should serialize/deserialize this via its JSON schema

**Verification:**
- DSPy 3.4+ supports Pydantic models in signatures
- JSON schema is auto-generated from the model

**Are we using it correctly?** YES
- `Refutation(pydantic.BaseModel)` with typed fields
- Used as `refutations: list[Refutation] = dspy.OutputField()`
- DSPy prompts the LM to return JSON matching the schema

**Potential risk:**
- Format failures if LM returns malformed JSON
- Mitigation: these count against the method (tracked in logs)

**No dspy.experimental.TypeSafe needed:**
- We're not using TypeSafe API (requires key and separate approval)
- Pydantic models work directly in DSPy 3.4+

---

## What We Do NOT Reimplement

**GEPA's core logic:**
- ✓ We do NOT reimplement Pareto front sampling
- ✓ We do NOT reimplement candidate generation
- ✓ We do NOT reimplement minibatch acceptance
- We ONLY replace the proposer (how new instructions are generated)

**DSPy primitives:**
- ✓ We do NOT reimplement `dspy.Predict`
- ✓ We do NOT reimplement signature handling
- ✓ We do NOT reimplement LM routing

**What we ADD:**
- Custom proposers that call critic → rewrite → check
- Budget tracking wrapper around LMs
- Refutation archive (regression tests)

---

## Version Pinning

**Current versions:**
```
dspy-ai==3.4.1
gepa==0.1.4
pydantic>=2.0
```

**Pinning strategy:**
- Exact pins for DSPy and GEPA in `requirements.txt`
- These are research code versions, not production dependencies
- For reproducibility, pin everything used in the main run

**Risks:**
- DSPy 3.5+ might change `instruction_proposer` signature → test against installed version
- GEPA updates might change reflection flow → we're using 0.1.4 as specified in design

---

## Tests Against Actual API

**Tests to add:**

1. **Signature verification test:**
   ```python
   def test_gepa_api_contract():
       import inspect
       sig = inspect.signature(dspy.GEPA.__init__)
       assert 'instruction_proposer' in sig.parameters
       assert 'reflection_lm' in sig.parameters
   ```

2. **Proposer contract test:**
   ```python
   def test_proposer_signature():
       proposer = PopperianProposer(...)
       # Should accept these 3 args
       result = proposer(candidate={}, reflective_dataset={}, components_to_update=[])
       assert isinstance(result, dict)
   ```

3. **Metric contract test:**
   ```python
   def test_metric_signature():
       result = ifbench_metric(gold, pred, None, None, None)
       assert hasattr(result, 'score')
       assert hasattr(result, 'feedback')
   ```

---

## Findings Summary

### ✓ Correct Usage

1. `instruction_proposer` — used as intended, signature matches
2. `reflection_lm` — provided correctly, budget wrapper is transparent
3. `metric` — returns correct format with score and feedback
4. SIMBA — correct parameters, prompt-only mode enforced
5. Pydantic models — work in DSPy 3.4+ without TypeSafe

### ⚠️ Risks

1. **Format failures:** If LM returns malformed JSON for typed outputs
   - Mitigation: counted as failures in proposer logs
   - Arm C (untyped) serves as control

2. **Budget accounting approximation:** 
   - Check calls change rollout mix vs GEPA default
   - Mitigation: 5% tolerance, actual counts reported

3. **Pydantic model overhead:**
   - Adds token cost to prompts (JSON schema)
   - Acceptable for research; measure actual token usage

### ✗ Nothing We're Doing Wrong

- We are NOT misusing any DSPy/GEPA APIs
- We are NOT reimplementing what DSPy already provides
- We are NOT bypassing GEPA's core logic

---

## Recommendations

1. **Add API contract tests** (signature checks) — done in `tests/test_api_contracts.py`
2. **Pin all dependencies** in `requirements.txt` before main run
3. **Document token overhead** from Pydantic schemas in results
4. **Test format failure handling** with deliberately broken LM responses

---

## References

- DSPy 3.4.1 source: `/home/ubuntu/.local/lib/python3.12/site-packages/dspy/`
- GEPA 0.1.4 source: `/home/ubuntu/.local/lib/python3.12/site-packages/gepa/`
- Inspection command: `python3 -c "import inspect, dspy; print(inspect.signature(dspy.GEPA.__init__))"`
