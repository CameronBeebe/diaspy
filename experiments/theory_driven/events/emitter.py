"""Event emitter for optimization steps."""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
import difflib

from .schema import (
    EpistemicEvent,
    OptimizationStep,
    Conjecture,
    TestSelection,
    TestOutcome,
    AcceptRejectDecision,
    LakatosLabel,
    Refutation,
    ClauseCredence,
)


class EventEmitter:
    """Emits structured epistemic events during optimization."""
    
    def __init__(
        self,
        experiment_name: str,
        benchmark_name: str,
        run_id: str,
        arm_name: str
    ):
        """Initialize event emitter.
        
        Args:
            experiment_name: Name of experiment
            benchmark_name: Name of benchmark
            run_id: Unique run identifier
            arm_name: Name of optimization arm
        """
        self.experiment_name = experiment_name
        self.benchmark_name = benchmark_name
        self.run_id = run_id
        self.arm_name = arm_name
        
        self.events: List[EpistemicEvent] = []
        self.step_counter = 0
    
    def emit_initialization(self, metadata: Dict[str, Any]) -> EpistemicEvent:
        """Emit initialization event.
        
        Args:
            metadata: Initialization metadata
            
        Returns:
            Initialization event
        """
        event = EpistemicEvent(
            event_id=str(uuid.uuid4()),
            event_type="initialization",
            experiment_name=self.experiment_name,
            benchmark_name=self.benchmark_name,
            run_id=self.run_id,
            metadata=metadata,
        )
        
        self.events.append(event)
        return event
    
    def emit_step(
        self,
        conjecture: Conjecture,
        test_selection: TestSelection,
        test_outcome: TestOutcome,
        decision: AcceptRejectDecision,
        calls_consumed: int,
        cost_consumed: float,
        call_trace_ids: Optional[List[str]] = None,
        refutations: Optional[List[Refutation]] = None,
        clause_credences: Optional[List[ClauseCredence]] = None,
        lakatos_label: LakatosLabel = LakatosLabel.UNKNOWN,
        lakatos_rationale: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_span_id: Optional[str] = None,
    ) -> EpistemicEvent:
        """Emit optimization step event.
        
        Args:
            conjecture: Proposed instruction change
            test_selection: Which test was run
            test_outcome: Test outcome
            decision: Accept/reject decision
            calls_consumed: LM calls consumed
            cost_consumed: Cost consumed
            call_trace_ids: IDs of raw LM traces
            refutations: Refutations (for Popperian arms)
            clause_credences: Clause credences (for Bayesian arms)
            lakatos_label: Progressive vs degenerating
            lakatos_rationale: Why this label
            mlflow_run_id: MLflow run ID
            mlflow_span_id: MLflow span ID
            
        Returns:
            Optimization step event
        """
        self.step_counter += 1
        
        step = OptimizationStep(
            step_id=f"{self.run_id}_step_{self.step_counter:04d}",
            arm_name=self.arm_name,
            iteration=self.step_counter,
            conjecture=conjecture,
            test_selection=test_selection,
            test_outcome=test_outcome,
            decision=decision,
            refutations=refutations,
            clause_credences=clause_credences,
            lakatos_label=lakatos_label,
            lakatos_rationale=lakatos_rationale,
            calls_consumed=calls_consumed,
            cost_consumed=cost_consumed,
            call_trace_ids=call_trace_ids or [],
            mlflow_run_id=mlflow_run_id,
            mlflow_span_id=mlflow_span_id,
        )
        
        event = EpistemicEvent(
            event_id=str(uuid.uuid4()),
            event_type="optimization_step",
            experiment_name=self.experiment_name,
            benchmark_name=self.benchmark_name,
            run_id=self.run_id,
            step=step,
        )
        
        self.events.append(event)
        return event
    
    def emit_finalization(self, metadata: Dict[str, Any]) -> EpistemicEvent:
        """Emit finalization event.
        
        Args:
            metadata: Finalization metadata
            
        Returns:
            Finalization event
        """
        event = EpistemicEvent(
            event_id=str(uuid.uuid4()),
            event_type="finalization",
            experiment_name=self.experiment_name,
            benchmark_name=self.benchmark_name,
            run_id=self.run_id,
            metadata=metadata,
        )
        
        self.events.append(event)
        return event
    
    def compute_instruction_diff(
        self,
        old_instruction: str,
        new_instruction: str
    ) -> str:
        """Compute unified diff of instruction change.
        
        Args:
            old_instruction: Old instruction
            new_instruction: New instruction
            
        Returns:
            Unified diff string
        """
        old_lines = old_instruction.splitlines(keepends=True)
        new_lines = new_instruction.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old_instruction",
            tofile="new_instruction",
            lineterm=""
        )
        
        return "".join(diff)
    
    def assess_lakatos_label(
        self,
        test_outcome: TestOutcome,
        test_selection: TestSelection,
        conjecture: Conjecture,
    ) -> tuple[LakatosLabel, str]:
        """Assess whether revision is progressive or degenerating.
        
        Progressive: Fixes/predicts new cases beyond the counterexamples
        Degenerating: Ad hoc patch fixing only the counterexample
        
        Args:
            test_outcome: Test outcome
            test_selection: Test selection
            conjecture: Conjecture
            
        Returns:
            Tuple of (label, rationale)
        """
        # Simple heuristic: if test includes examples beyond the minibatch
        # and they pass, it's progressive
        
        if test_selection.test_type == "full_validation":
            if test_outcome.passed or test_outcome.score > 0.5:
                return (
                    LakatosLabel.PROGRESSIVE,
                    "Validated on full set, fixes beyond counterexamples"
                )
            else:
                return (
                    LakatosLabel.DEGENERATING,
                    "Failed validation, may be overfitting to counterexamples"
                )
        
        elif test_selection.test_type == "survival_check":
            if test_outcome.survived:
                return (
                    LakatosLabel.PROGRESSIVE,
                    "Survived counterexample re-test, fixes persist"
                )
            else:
                return (
                    LakatosLabel.DEGENERATING,
                    "Failed survival check, ad hoc patch"
                )
        
        else:
            return (
                LakatosLabel.NEUTRAL,
                "Minibatch test, progressiveness unclear"
            )
    
    def get_events(self) -> List[EpistemicEvent]:
        """Get all events emitted so far.
        
        Returns:
            List of events
        """
        return self.events.copy()
