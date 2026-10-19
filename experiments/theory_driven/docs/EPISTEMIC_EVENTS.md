# Epistemic Event Traces - Implementation Guide

**Status**: Complete (2026-10-19)  
**LOC**: ~1,000 (875 LOC Python + tests + notebook)  
**Tests**: 12 tests in `test_event_emission.py`

## Overview

Epistemic events provide structured, typed traces of optimization steps that enable:
- Understanding *why* each instruction change was proposed
- Tracking lineage from root to final candidate
- Distinguishing progressive vs degenerating revisions (Lakatos)
- Computing cost breakdowns by decision type
- Cross-arm comparison on the same examples
- Theory-driven feedback loops

## Architecture

```
events/
├── schema.py      # Pydantic models (350 LOC)
├── emitter.py     # EventEmitter for step emission (250 LOC)
├── storage.py     # JSONL persistence (150 LOC)
├── analysis.py    # EventAnalyzer for insights (250 LOC)
└── __init__.py    # Public API
```

## Schema (Pydantic Models)

### Core Event Types

**`EpistemicEvent`** - Top-level event wrapper
- `event_id`, `event_type` (initialization, optimization_step, finalization)
- `experiment_name`, `benchmark_name`, `run_id`
- `timestamp`
- `step: OptimizationStep` (if optimization_step)
- `metadata: Dict` (for init/finalize)

**`OptimizationStep`** - One optimization step
- `step_id`, `timestamp`, `arm_name`, `iteration`
- `conjecture: Conjecture` - Proposed change
- `refutations: List[Refutation]` (optional, for Popperian arms)
- `clause_credences: List[ClauseCredence]` (optional, for Bayesian arms)
- `test_selection: TestSelection` - Which test was run
- `test_outcome: TestOutcome` - Test results
- `decision: AcceptRejectDecision` - Accept/reject decision
- `lakatos_label: LakatosLabel` - Progressive vs degenerating
- `lakatos_rationale: str`
- `calls_consumed: int`, `cost_consumed: float`
- `call_trace_ids: List[str]` - Link to raw LM traces
- `mlflow_run_id`, `mlflow_span_id` (optional)

### Supporting Types

**`Conjecture`** - Proposed instruction change
- `candidate_id`, `parent_id` (lineage)
- `module_name` - Which module is being updated
- `old_instruction`, `new_instruction`
- `instruction_diff` - Unified diff
- `generated_by` - Arm name
- `generation_reason` - Why this change

**`Refutation`** (Popperian arms)
- `claim` - Falsifiable claim
- `counterexample_ids` - Example indices
- `severity: float` (0-1)
- `proposed_fix`
- `example_details` (optional)

**`ClauseCredence`** (Bayesian arms)
- `clause_id`, `clause_text`, `clause_kind`
- `credence_before`, `credence_after` (0-1)
- `evidence_count`
- `credence_method` (jev, verbalized, base_rate)

**`TestSelection`**
- `test_type` (full_validation, minibatch, clause_test, survival_check)
- `target_clause_id` (optional)
- `example_ids`, `example_count`
- `selection_rationale` - Why this test
- `expected_information_gain` (optional)
- `test_cost_units`

**`TestOutcome`**
- `passed: bool`, `score: float`
- `example_scores: List[float]`
- `feedback` (optional)
- `survival_check: bool`, `survived` (optional)

**`AcceptRejectDecision`**
- `accepted: bool`
- `decision_reason`
- `minibatch_score`, `validation_score` (optional)
- `survival_check_passed` (optional)
- `budget_exhausted: bool`

**`LakatosLabel`** (Enum)
- `PROGRESSIVE` - Predicts/fixes new cases beyond counterexamples
- `DEGENERATING` - Ad hoc patch fixing only the counterexample
- `NEUTRAL` - Neither clearly progressive nor degenerating
- `UNKNOWN` - Not yet evaluated

## Usage

### 1. Create EventEmitter

```python
from events import EventEmitter

emitter = EventEmitter(
    experiment_name="theory_driven",
    benchmark_name="hover",
    run_id="run_001",
    arm_name="popperian"
)
```

### 2. Emit Initialization Event

```python
emitter.emit_initialization({
    "budget": 50,
    "seed": 42,
    "model": "qwen3.5-9b"
})
```

### 3. Emit Optimization Step

```python
from events import (
    Conjecture, TestSelection, TestOutcome,
    AcceptRejectDecision, Refutation
)

# Create step components
conjecture = Conjecture(
    candidate_id="candidate_001",
    parent_id="root",
    module_name="qa",
    old_instruction="Answer the question.",
    new_instruction="Answer the question using evidence from the context.",
    generated_by="popperian",
    generation_reason="Failed on examples without explicit reasoning"
)

# Compute diff
diff = emitter.compute_instruction_diff(
    conjecture.old_instruction,
    conjecture.new_instruction
)
conjecture.instruction_diff = diff

# Test selection
test_selection = TestSelection(
    test_type="minibatch",
    example_ids=[10, 15, 23],
    example_count=3,
    selection_rationale="Highest severity refutations",
    expected_information_gain=0.8,
    test_cost_units=3
)

# Test outcome
test_outcome = TestOutcome(
    passed=True,
    score=0.85,
    example_scores=[0.8, 0.9, 0.85]
)

# Decision
decision = AcceptRejectDecision(
    accepted=True,
    decision_reason="Score improved from 0.6 to 0.85",
    minibatch_score=0.85
)

# Optional: Refutations (for Popperian arms)
refutations = [
    Refutation(
        claim="Model fails when context lacks explicit answer",
        counterexample_ids=[10, 15],
        severity=0.9,
        proposed_fix="Add explicit reasoning step"
    )
]

# Emit step
event = emitter.emit_step(
    conjecture=conjecture,
    test_selection=test_selection,
    test_outcome=test_outcome,
    decision=decision,
    calls_consumed=5,
    cost_consumed=0.01,
    call_trace_ids=["trace_001", "trace_002"],
    refutations=refutations
)

# Lakatos label is automatically assessed
# Can also manually specify:
# lakatos_label=LakatosLabel.PROGRESSIVE,
# lakatos_rationale="Validated on full set"
```

### 4. Save Events

```python
from events import EventStorage
from pathlib import Path

storage = EventStorage(Path("results"))
events = emitter.get_events()

filepath = storage.save_events(
    events,
    run_id="run_001",
    arm_name="popperian"
)

print(f"Saved to {filepath}")
# Saved to results/epistemic_events/events_run_001_popperian.jsonl
```

### 5. Load and Analyze Events

```python
from events import EventStorage, EventAnalyzer

storage = EventStorage(Path("results"))
events = storage.load_events(filepath)

analyzer = EventAnalyzer(events)

# Build lineage tree
tree = analyzer.build_lineage_tree()
print(tree)
# {'root': ['candidate_001', 'candidate_002'],
#  'candidate_001': ['candidate_003']}

# Count by Lakatos label
counts = analyzer.count_by_lakatos_label()
print(counts)
# {LakatosLabel.PROGRESSIVE: 8, LakatosLabel.DEGENERATING: 2}

# Progressive ratio
ratio = analyzer.get_progressive_ratio()
print(f"Progressive ratio: {ratio:.2%}")
# Progressive ratio: 80.00%

# Cost breakdown
breakdown = analyzer.compute_cost_breakdown()
print(breakdown)
# {'total_calls': 50, 'total_cost': 0.15,
#  'by_label': {LakatosLabel.PROGRESSIVE: {'calls': 40, 'cost': 0.12}}}

# Timeline
timeline = analyzer.extract_timeline()
for entry in timeline[:3]:
    print(entry)

# Compare candidates
comparison = analyzer.compare_candidates("candidate_001", "candidate_002")
print(comparison)
```

## Integration with Arms

### Example: Popperian Arm

```python
class PopperianArm(BaseArm):
    def __init__(self, ...):
        super().__init__(...)
        self.event_emitter = None
    
    def set_event_emitter(self, emitter: EventEmitter):
        """Set event emitter for this arm."""
        self.event_emitter = emitter
    
    def compile(self, ...):
        # ... optimization loop ...
        
        if self.event_emitter:
            # Create conjecture
            conjecture = Conjecture(
                candidate_id=f"candidate_{step:03d}",
                parent_id=parent_id,
                module_name=module.name,
                old_instruction=old_inst,
                new_instruction=new_inst,
                generated_by=self.name,
                generation_reason=f"Refutation: {refutation.claim}"
            )
            
            # ... test the candidate ...
            
            # Emit step
            self.event_emitter.emit_step(
                conjecture=conjecture,
                test_selection=test_selection,
                test_outcome=test_outcome,
                decision=decision,
                calls_consumed=calls_this_step,
                cost_consumed=cost_this_step,
                refutations=[refutation],
                call_trace_ids=trace_ids
            )
```

### Example: Bayesian/Popperian Arm

```python
class BayesianPopperianArm(BaseArm):
    def compile(self, ...):
        # ... Jev clause decomposition ...
        
        if self.event_emitter:
            # Build clause credences
            clause_credences = []
            for clause in clauses:
                cc = ClauseCredence(
                    clause_id=clause.id,
                    clause_text=clause.text,
                    clause_kind=clause.kind,
                    credence_before=clause.prior,
                    credence_after=clause.posterior,
                    evidence_count=len(evidence),
                    credence_method="jev"
                )
                clause_credences.append(cc)
            
            # ... test selection based on EIG ...
            
            test_selection = TestSelection(
                test_type="clause_test",
                target_clause_id=most_suspect_clause_id,
                example_ids=test_examples,
                example_count=len(test_examples),
                selection_rationale=f"Highest EIG: {eig:.2f} bits",
                expected_information_gain=eig,
                test_cost_units=len(test_examples)
            )
            
            # Emit step
            self.event_emitter.emit_step(
                conjecture=conjecture,
                test_selection=test_selection,
                test_outcome=test_outcome,
                decision=decision,
                calls_consumed=calls_this_step,
                cost_consumed=cost_this_step,
                clause_credences=clause_credences,
                call_trace_ids=trace_ids
            )
```

## Notebook Analysis

See `notebooks/replay_events.ipynb` for:
- **Timeline view**: Accepted vs rejected over time, Lakatos labels
- **Lineage tree**: Text-based tree from root to leaves
- **Lakatos analysis**: Distribution, progressive ratio, cost by label
- **Cross-arm comparison**: Compare multiple arms on same benchmark
- **Refutation patterns**: Severity distribution, high-severity claims
- **Clause credence evolution**: Track how credences change over time

Example visualizations:
- Cumulative accepted/rejected plot
- Lakatos labels over time (scatter, colored by label)
- Severity histogram for refutations
- Credence evolution line plots

## Testing

`tests/test_event_emission.py` includes 12 tests:
1. `test_event_emitter_basic` - Basic emitter functionality
2. `test_event_emitter_step` - Step event emission
3. `test_event_validation` - Pydantic validation
4. `test_lakatos_assessment` - Lakatos label assessment
5. `test_instruction_diff` - Diff computation
6. `test_event_storage_roundtrip` - Save/load JSONL
7. `test_event_analyzer` - Analyzer functionality

**Run tests**:
```bash
cd experiments/theory_driven
pytest tests/test_event_emission.py -v
```

All tests pass offline with mock data.

## Schema Validation

Pydantic automatically validates:
- Field types (str, int, float, bool, enum)
- Constraints (0 ≤ severity ≤ 1, 0 ≤ credence ≤ 1)
- Required fields
- Nested models

Example validation error:
```python
Refutation(
    claim="test",
    counterexample_ids=[1, 2],
    severity=1.5,  # Invalid: >1
    proposed_fix="fix"
)
# Raises: pydantic.ValidationError: severity must be ≤ 1
```

## Linkage to Raw Traces

Events link to raw LM call traces via `call_trace_ids`:

```python
# In ProvenanceLogger
trace_id = f"trace_{uuid.uuid4()}"
provenance_logger.log_call(
    trace_id=trace_id,
    prompt=prompt,
    completion=completion,
    # ... other fields ...
)

# In arm
call_trace_ids.append(trace_id)

# In event
emitter.emit_step(
    # ...
    call_trace_ids=call_trace_ids
)
```

This enables:
- Tracing decisions back to specific LM calls
- Cost reconciliation (sum of trace costs = step cost)
- Provenance auditing (which prompts led to which decisions)

## MLflow Integration (Schema Support)

Events include MLflow fields:
- `mlflow_run_id` - MLflow run ID
- `mlflow_span_id` - MLflow span ID

To integrate:

```python
import mlflow

with mlflow.start_run() as run:
    run_id = run.info.run_id
    
    with mlflow.start_span(name="optimization_step") as span:
        span_id = span.span_id
        
        # Emit event
        emitter.emit_step(
            # ...
            mlflow_run_id=run_id,
            mlflow_span_id=span_id
        )
        
        # Log to MLflow
        mlflow.log_dict(event.dict(), f"events/step_{step_id}.json")
```

**Note**: MLflow integration not tested in this implementation.

## Cross-Arm Comparison

To compare how different arms approach the same problem:

```python
# Load events for multiple arms
storage = EventStorage(Path("results"))

arm_a_events = storage.load_events("events_run_001_popperian.jsonl")
arm_b_events = storage.load_events("events_run_001_bayesian_popperian.jsonl")

analyzer_a = EventAnalyzer(arm_a_events)
analyzer_b = EventAnalyzer(arm_b_events)

# Compare
print(f"Arm A: {analyzer_a.get_progressive_ratio():.2%} progressive")
print(f"Arm B: {analyzer_b.get_progressive_ratio():.2%} progressive")

# Compare specific candidates
comparison = analyzer_a.compare_candidates("candidate_001", "candidate_002")
print(comparison)
```

Or use the notebook for visual cross-arm diff.

## Future Extensions

Potential additions:
1. **Event filtering**: Query events by Lakatos label, test type, cost range
2. **Diff visualization**: Visual diff of instruction changes
3. **Lineage graph**: GraphViz rendering of lineage tree
4. **Cost forecasting**: Predict future cost based on event patterns
5. **Theory-driven feedback**: Feed successful patterns back to proposer
6. **Event-driven metrics**: Metrics computed from events (not just outcomes)

## References

- Schema: `experiments/theory_driven/events/schema.py`
- Emitter: `experiments/theory_driven/events/emitter.py`
- Storage: `experiments/theory_driven/events/storage.py`
- Analyzer: `experiments/theory_driven/events/analysis.py`
- Tests: `experiments/theory_driven/tests/test_event_emission.py`
- Notebook: `experiments/theory_driven/notebooks/replay_events.ipynb`
- Status: `experiments/theory_driven/IMPLEMENTATION_STATUS.md`
