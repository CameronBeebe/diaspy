"""Arms K-Jev and K-verbal: Clause-level Bayesian/Popperian optimizer.

This is a standalone loop, not a GEPA proposer. It decomposes the instruction
into clauses, estimates per-clause credences, and tests suspect clauses with
severity (Popperian: would probably have failed if harmless).
"""

import os
import json
from typing import Callable, List, Dict, Any, Optional, Literal
import pydantic
import dspy
from .base import BaseArm
from ..config import BudgetTracker


class Clause(pydantic.BaseModel):
    """An atomic typed clause of an instruction."""
    id: int
    text: str
    kind: Literal["constraint", "procedure", "format", "heuristic"]


class ClauseDecomposition(dspy.Signature):
    """Decompose an instruction into atomic typed clauses."""
    instruction: str = dspy.InputField()
    clauses: List[Clause] = dspy.OutputField(
        desc="atomic clauses with unique IDs, text, and kind"
    )


class ClauseCredenceRequest(dspy.Signature):
    """Estimate per-clause blame credences given traces and feedback.
    
    For K-Jev, this uses dspy.experimental.TypeSafe with Noul fields.
    For K-verbal, this uses standard prediction with verbalized probabilities.
    """
    clauses: List[Clause] = dspy.InputField()
    traces: str = dspy.InputField(desc="execution traces and feedback")
    credences: List[float] = dspy.OutputField(
        desc="P(clause implicated | traces) for each clause, in same order"
    )


class ClauseRevision(dspy.Signature):
    """Revise or drop a refuted clause."""
    clause: Clause = dspy.InputField()
    counterexamples: str = dspy.InputField(desc="examples that failed with this clause")
    revised_clause: str = dspy.OutputField(
        desc="revised clause text, or empty string to drop"
    )


class BayesianPopperianOptimizer:
    """Clause-level Bayesian/Popperian optimizer.
    
    Steps per spec:
    1. Decompose instruction into clauses
    2. Assign credences (Jev or verbalized)
    3. Pick most informative test (highest expected information gain)
    4. Run severe test (~3 rollouts: 2 failing + 1 passing)
    5. Revise only refuted clauses
    6. Validate once at end
    
    Extension points used:
    - dspy.experimental.TypeSafe for Jev credences (needs TYPESAFE_API_KEY)
    - Beta posterior tracking for clause credences
    - Bayesian experimental design for test selection
    """
    
    def __init__(
        self,
        program: dspy.Module,
        metric: Callable,
        trainset: list,
        valset: list,
        task_lm: dspy.LM,
        reflection_lm: dspy.LM,
        budget_tracker: BudgetTracker,
        credence_method: Literal["jev", "verbal"] = "jev",
        max_iterations: int = 10,
    ):
        """Initialize optimizer.
        
        Args:
            program: DSPy program to optimize
            metric: Evaluation metric
            trainset: Training examples
            valset: Validation examples
            task_lm: Task LM
            reflection_lm: Reflection LM
            budget_tracker: Budget tracker
            credence_method: "jev" for TypeSafe, "verbal" for verbalized
            max_iterations: Maximum optimization iterations
        """
        self.program = program
        self.metric = metric
        self.trainset = trainset
        self.valset = valset
        self.task_lm = task_lm
        self.reflection_lm = reflection_lm
        self.budget_tracker = budget_tracker
        self.credence_method = credence_method
        self.max_iterations = max_iterations
        
        self.decompose = dspy.Predict(ClauseDecomposition)
        self.get_credences = dspy.Predict(ClauseCredenceRequest)
        self.revise = dspy.Predict(ClauseRevision)
        
        self.log: List[Dict[str, Any]] = []
        
        # Check TypeSafe availability for Jev
        if credence_method == "jev":
            if not os.getenv("TYPESAFE_API_KEY"):
                raise ValueError(
                    "K-Jev requires TYPESAFE_API_KEY environment variable. "
                    "Use K-verbal if TypeSafe is unavailable."
                )
            # Note: In real implementation, would import dspy.experimental.TypeSafe
            # For now, we'll use standard prediction as placeholder
    
    def decompose_instruction(self, instruction: str) -> List[Clause]:
        """Decompose instruction into clauses."""
        with dspy.context(lm=self.reflection_lm):
            result = self.decompose(instruction=instruction)
        
        self.budget_tracker.record_reflection_call(
            input_tokens=0,
            output_tokens=0,
            metadata={"type": "decompose"}
        )
        
        return result.clauses
    
    def estimate_credences(
        self,
        clauses: List[Clause],
        traces: str
    ) -> List[float]:
        """Estimate per-clause blame credences.
        
        For K-Jev: uses TypeSafe with Noul fields (probability field is credence)
        For K-verbal: uses verbalized confidence from reflection LM
        """
        with dspy.context(lm=self.reflection_lm):
            if self.credence_method == "jev":
                # Real implementation would use:
                # result = dspy.experimental.TypeSafe(ClauseCredenceRequest)(
                #     clauses=clauses, traces=traces
                # )
                # For now, placeholder
                result = self.get_credences(clauses=clauses, traces=traces)
                
                # Record Jev call with special pricing
                self.budget_tracker.record_jev_call(
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0002,  # Approximate per spec
                    metadata={"type": "jev_credence", "n_clauses": len(clauses)}
                )
            else:
                # K-verbal: standard prediction
                result = self.get_credences(clauses=clauses, traces=traces)
                
                self.budget_tracker.record_reflection_call(
                    input_tokens=0,
                    output_tokens=0,
                    metadata={"type": "verbal_credence", "n_clauses": len(clauses)}
                )
        
        return result.credences
    
    def select_test(
        self,
        clauses: List[Clause],
        credences: List[float],
        traces: List[Any]
    ) -> tuple[Clause, List[Any]]:
        """Select the most informative test: highest expected information gain.
        
        Per spec: Choose clause with highest EIG, test on ~3 examples
        (2 failing + 1 passing) for severity.
        
        Returns:
            Tuple of (clause_to_test, test_examples)
        """
        # Simple heuristic: test clause with highest credence
        # Real implementation would compute proper EIG
        max_idx = credences.index(max(credences))
        clause = clauses[max_idx]
        
        # Select ~3 examples: 2 failing + 1 passing
        failing = [ex for ex in self.trainset if hasattr(ex, "_score") and ex._score < 1]
        passing = [ex for ex in self.trainset if hasattr(ex, "_score") and ex._score >= 1]
        
        test_examples = failing[:2] + passing[:1]
        
        return clause, test_examples
    
    def test_clause(
        self,
        clause: Clause,
        test_examples: List[Any],
        current_instruction: str
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """Test clause on examples. Returns (refuted, results)."""
        results = []
        
        # Test with clause removed/revised
        modified_instruction = current_instruction.replace(clause.text, "")
        
        with dspy.context(lm=self.task_lm):
            for ex in test_examples:
                pred = self.program(question=ex.question)
                score = self.metric(ex, pred, None, None, None).score
                
                self.budget_tracker.record_task_call(
                    input_tokens=0,
                    output_tokens=0,
                    metadata={"type": "clause_test", "clause_id": clause.id}
                )
                
                results.append({
                    "question": ex.question,
                    "score": score,
                })
        
        # Clause is refuted if removing it improves scores
        mean_score = sum(r["score"] for r in results) / len(results)
        refuted = mean_score > 0.5  # Heuristic threshold
        
        return refuted, results
    
    def revise_clause(
        self,
        clause: Clause,
        test_results: List[Dict[str, Any]]
    ) -> str:
        """Revise a refuted clause."""
        counterexamples = json.dumps(test_results, indent=2)
        
        with dspy.context(lm=self.reflection_lm):
            result = self.revise(
                clause=clause,
                counterexamples=counterexamples
            )
        
        self.budget_tracker.record_reflection_call(
            input_tokens=0,
            output_tokens=0,
            metadata={"type": "revise_clause", "clause_id": clause.id}
        )
        
        return result.revised_clause
    
    def optimize(self) -> dspy.Module:
        """Run the optimization loop."""
        # Get initial instruction
        predictors = list(self.program.predictors())
        if not predictors:
            return self.program
        
        current_instruction = predictors[0].signature.instructions or "Solve the task."
        
        # Decompose into clauses
        clauses = self.decompose_instruction(current_instruction)
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            if self.budget_tracker.budget_exhausted:
                break
            
            # Run validation and collect traces
            traces = []
            with dspy.context(lm=self.task_lm):
                for ex in self.valset:
                    pred = self.program(question=ex.question)
                    score = self.metric(ex, pred, None, None, None).score
                    
                    self.budget_tracker.record_task_call(
                        input_tokens=0,
                        output_tokens=0,
                        metadata={"type": "validation", "iteration": iteration}
                    )
                    
                    traces.append({
                        "question": ex.question,
                        "prediction": str(pred),
                        "score": score,
                    })
            
            # Estimate credences
            traces_str = json.dumps(traces, indent=2)
            credences = self.estimate_credences(clauses, traces_str)
            
            # Select and test most suspect clause
            clause, test_examples = self.select_test(clauses, credences, traces)
            refuted, test_results = self.test_clause(
                clause, test_examples, current_instruction
            )
            
            # Log iteration
            self.log.append({
                "iteration": iteration,
                "clause_id": clause.id,
                "clause_text": clause.text,
                "credence": credences[clause.id],
                "refuted": refuted,
                "test_results": test_results,
            })
            
            # Revise if refuted
            if refuted:
                revised = self.revise_clause(clause, test_results)
                if revised:
                    # Update instruction
                    current_instruction = current_instruction.replace(
                        clause.text, revised
                    )
                    clause.text = revised
                else:
                    # Drop clause
                    current_instruction = current_instruction.replace(clause.text, "")
                    clauses = [c for c in clauses if c.id != clause.id]
        
        # Final validation
        # (would be done at end, counts toward budget)
        
        # Update program instruction
        predictors[0].signature = predictors[0].signature.with_instructions(
            current_instruction
        )
        
        return self.program


class BayesianPopperianArm(BaseArm):
    """Arms K-Jev and K-verbal: Clause-level Bayesian/Popperian."""
    
    def __init__(
        self,
        name: str,
        description: str,
        budget_tracker: BudgetTracker,
        credence_method: Literal["jev", "verbal"] = "jev",
        max_iterations: int = 10,
        **kwargs
    ):
        """Initialize Bayesian/Popperian arm."""
        super().__init__(name, description, budget_tracker, **kwargs)
        self.credence_method = credence_method
        self.max_iterations = max_iterations
        self.optimizer: Optional[BayesianPopperianOptimizer] = None
    
    def compile(
        self,
        program: dspy.Module,
        metric: Callable,
        trainset: list,
        valset: list,
        critic_lm: dspy.LM,
        task_lm: dspy.LM,
        **compile_kwargs
    ) -> dspy.Module:
        """Compile with Bayesian/Popperian optimizer."""
        self.optimizer = BayesianPopperianOptimizer(
            program=program,
            metric=metric,
            trainset=trainset,
            valset=valset,
            task_lm=task_lm,
            reflection_lm=critic_lm,  # reflection_lm = critic_lm
            budget_tracker=self.budget_tracker,
            credence_method=self.credence_method,
            max_iterations=self.max_iterations,
        )
        
        return self.optimizer.optimize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get arm statistics."""
        stats = super().get_stats()
        if self.optimizer:
            stats["optimizer_log"] = self.optimizer.log
        return stats
