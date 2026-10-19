"""Pydantic schemas for epistemic events.

These schemas are reusable beyond this experiment and define the structure
of events emitted during optimization.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class LakatosLabel(str, Enum):
    """Lakatos label for revision progressiveness."""
    PROGRESSIVE = "progressive"  # Predicts/fixes new cases beyond counterexamples
    DEGENERATING = "degenerating"  # Ad hoc patch fixing only the counterexample
    NEUTRAL = "neutral"  # Neither clearly progressive nor degenerating
    UNKNOWN = "unknown"  # Not yet evaluated


class Refutation(BaseModel):
    """Typed refutation object (used in Popperian arms)."""
    claim: str = Field(..., description="Falsifiable claim about why instruction fails")
    counterexample_ids: List[int] = Field(..., description="Indices of failing examples")
    severity: float = Field(..., ge=0, le=1, description="Severity rating (0-1)")
    proposed_fix: str = Field(..., description="Proposed fix for this failure")
    
    # Optional metadata
    example_details: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Full example data for counterexamples"
    )


class ClauseCredence(BaseModel):
    """Per-clause blame credence (used in Bayesian/Popperian arms)."""
    clause_id: int = Field(..., description="Unique clause identifier")
    clause_text: str = Field(..., description="Text of the clause")
    clause_kind: Literal["constraint", "procedure", "format", "heuristic"] = Field(
        ...,
        description="Type of clause"
    )
    
    # Credence tracking
    credence_before: float = Field(..., ge=0, le=1, description="P(implicated) before test")
    credence_after: Optional[float] = Field(None, ge=0, le=1, description="P(implicated) after test")
    
    # Evidence
    evidence_count: int = Field(0, description="Number of observations")
    
    # Method
    credence_method: Literal["jev", "verbalized", "base_rate"] = Field(
        ...,
        description="How credence was computed"
    )


class Conjecture(BaseModel):
    """Proposed instruction change."""
    candidate_id: str = Field(..., description="Unique identifier for this candidate")
    parent_id: Optional[str] = Field(None, description="Parent candidate ID (lineage)")
    
    module_name: str = Field(..., description="Module/component being updated")
    old_instruction: str = Field(..., description="Current instruction")
    new_instruction: str = Field(..., description="Proposed instruction")
    
    # Diff representation
    instruction_diff: Optional[str] = Field(
        None,
        description="Text diff (unified format) of instruction change"
    )
    
    # Generation context
    generated_by: str = Field(..., description="Arm/proposer that generated this")
    generation_reason: Optional[str] = Field(
        None,
        description="Why this change was proposed"
    )


class TestSelection(BaseModel):
    """Record of which test was chosen and why."""
    test_type: Literal["full_validation", "minibatch", "clause_test", "survival_check"] = Field(
        ...,
        description="Type of test"
    )
    
    # For clause-level tests
    target_clause_id: Optional[int] = Field(None, description="Clause being tested")
    
    # Test examples
    example_ids: List[int] = Field(..., description="Indices of examples used")
    example_count: int = Field(..., description="Number of examples")
    
    # Selection rationale
    selection_rationale: str = Field(
        ...,
        description="Why this test was chosen (e.g., highest EIG, most severe)"
    )
    expected_information_gain: Optional[float] = Field(
        None,
        description="Expected information gain (bits)"
    )
    
    # Cost
    test_cost_units: int = Field(..., description="Cost of this test in units")


class TestOutcome(BaseModel):
    """Outcome of running a test."""
    passed: bool = Field(..., description="Whether test passed")
    score: float = Field(..., description="Test score (0-1)")
    
    # Per-example results
    example_scores: List[float] = Field(..., description="Score for each test example")
    
    # Feedback
    feedback: Optional[str] = Field(None, description="Feedback from metric")
    
    # For survival checks
    survival_check: bool = Field(False, description="Was this a survival check?")
    survived: Optional[bool] = Field(None, description="Did candidate survive?")


class AcceptRejectDecision(BaseModel):
    """Decision to accept or reject a candidate."""
    accepted: bool = Field(..., description="Whether candidate was accepted")
    
    # Reason
    decision_reason: str = Field(..., description="Why this decision was made")
    
    # Supporting data
    minibatch_score: Optional[float] = Field(None, description="Score on minibatch")
    validation_score: Optional[float] = Field(None, description="Score on validation set")
    
    # For Popperian arms
    survival_check_passed: Optional[bool] = Field(
        None,
        description="Whether survival check passed"
    )
    
    # Budget
    budget_exhausted: bool = Field(False, description="Was budget exhausted?")


class OptimizationStep(BaseModel):
    """Complete record of one optimization step."""
    
    # Identity
    step_id: str = Field(..., description="Unique step identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    arm_name: str = Field(..., description="Which arm produced this step")
    iteration: int = Field(..., description="Iteration number within optimization")
    
    # Conjecture
    conjecture: Conjecture = Field(..., description="Proposed change")
    
    # Evidence (varies by arm)
    refutations: Optional[List[Refutation]] = Field(
        None,
        description="Refutations (for Popperian arms)"
    )
    clause_credences: Optional[List[ClauseCredence]] = Field(
        None,
        description="Clause credences (for Bayesian/Popperian arms)"
    )
    
    # Test
    test_selection: TestSelection = Field(..., description="Which test was run")
    test_outcome: TestOutcome = Field(..., description="Test outcome")
    
    # Decision
    decision: AcceptRejectDecision = Field(..., description="Accept/reject decision")
    
    # Lakatos label
    lakatos_label: LakatosLabel = Field(
        LakatosLabel.UNKNOWN,
        description="Progressive vs degenerating"
    )
    lakatos_rationale: Optional[str] = Field(
        None,
        description="Why this label was assigned"
    )
    
    # Cost accounting
    calls_consumed: int = Field(..., description="LM calls consumed in this step")
    cost_consumed: float = Field(..., description="Cost consumed in this step")
    
    # Linkage to raw traces
    call_trace_ids: List[str] = Field(
        default_factory=list,
        description="IDs of raw LM call traces for this step"
    )
    
    # MLflow
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    mlflow_span_id: Optional[str] = Field(None, description="MLflow span ID")


class EpistemicEvent(BaseModel):
    """Top-level epistemic event.
    
    This is the main event type emitted during optimization. It wraps an
    OptimizationStep with additional metadata.
    """
    
    event_id: str = Field(..., description="Unique event identifier")
    event_type: Literal["optimization_step", "initialization", "finalization"] = Field(
        ...,
        description="Type of event"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Experiment context
    experiment_name: str = Field(..., description="Experiment name")
    benchmark_name: str = Field(..., description="Benchmark being optimized")
    run_id: str = Field(..., description="Unique run identifier")
    
    # The actual step (if optimization_step)
    step: Optional[OptimizationStep] = Field(None, description="Optimization step details")
    
    # For initialization/finalization events
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
