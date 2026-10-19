"""Arm B: Typed Popperian proposer for GEPA.

Based on popperian_gepa_skeleton.py, adapted for real LMs.
"""

import json
from typing import Callable, List, Dict, Any, Optional
import pydantic
import dspy
from .base import BaseArm
from ..config import BudgetTracker


class Refutation(pydantic.BaseModel):
    """One falsifiable claim about why the current instruction fails."""
    claim: str  # "instruction fails because ..."
    counterexample_ids: List[int]  # indices into the reflective minibatch
    severity: float = pydantic.Field(ge=0, le=1)
    proposed_fix: str


class Critique(dspy.Signature):
    """Conjecture-and-refutation: state each way the instruction fails as a
    falsifiable claim, cite the example ids that break it, and rate severity.
    """
    current_instruction: str = dspy.InputField()
    examples_with_feedback: str = dspy.InputField(
        desc="numbered examples: inputs, outputs, feedback"
    )
    refutations: List[Refutation] = dspy.OutputField()


class Rewrite(dspy.Signature):
    """Rewrite the instruction so that it survives every refutation. Address each one."""
    current_instruction: str = dspy.InputField()
    refutations: List[Refutation] = dspy.InputField()
    new_instruction: str = dspy.OutputField()


class PopperianProposer:
    """GEPA ProposalFn: critic -> typed refutations -> rewrite -> mechanical survival check.
    
    Extension points used (per robustness audit requirements):
    - GEPA instruction_proposer: Custom proposal function (documented in GEPA API)
    - Mechanical survival check: Re-runs counterexamples with new instruction
    - Refutation archive: Regression tests from high-severity past refutations
    """
    
    def __init__(
        self,
        program: dspy.Module,
        metric: Callable,
        gold_by_question: Dict[str, Any],
        critic_lm: dspy.LM,
        task_lm: dspy.LM,
        budget_tracker: BudgetTracker,
        max_attempts: int = 2,
        severity_tau: float = 0.5,
        archive_cap: int = 6,
    ):
        """Initialize Popperian proposer.
        
        Args:
            program: DSPy program being optimized
            metric: Evaluation metric
            gold_by_question: Mapping from question to gold example
            critic_lm: LM for critique and rewrite
            task_lm: LM for task execution
            budget_tracker: Budget tracker for recording calls
            max_attempts: Maximum rewrite attempts per iteration
            severity_tau: Minimum severity for archive inclusion
            archive_cap: Maximum archive size
        """
        self.program = program
        self.metric = metric
        self.gold = gold_by_question
        self.critic_lm = critic_lm
        self.task_lm = task_lm
        self.budget_tracker = budget_tracker
        self.max_attempts = max_attempts
        self.tau = severity_tau
        self.archive_cap = archive_cap
        
        self.critique = dspy.Predict(Critique)
        self.rewrite = dspy.Predict(Rewrite)
        self.log: List[Dict[str, Any]] = []
        
        # Refutation archive: counterexamples become regression tests
        self.archive: List[str] = []
    
    def survives(
        self,
        predictor_name: str,
        instruction: str,
        refutations: List[Refutation],
        batch: list
    ) -> tuple[bool, List[Any]]:
        """Re-run each counterexample under the new instruction; all must now score 1.
        
        This is the mechanical survival check that distinguishes arm B from arm A.
        
        Args:
            predictor_name: Name of the predictor to update
            instruction: New instruction to test
            refutations: Refutations that led to this instruction
            batch: Current minibatch
            
        Returns:
            Tuple of (survived, tested_questions)
        """
        # Get counterexample questions
        ids = sorted({
            i for r in refutations
            if r.severity >= self.tau
            for i in r.counterexample_ids
            if i < len(batch)
        })
        
        questions = [batch[i]["Inputs"]["question"] for i in ids]
        
        # Add archived regression tests
        questions += [q for q in self.archive if q not in questions]
        
        # Test each question with new instruction
        prog = self.program.deepcopy()
        predictors = dict(prog.named_predictors())
        if predictor_name in predictors:
            predictors[predictor_name].signature = \
                predictors[predictor_name].signature.with_instructions(instruction)
        
        with dspy.context(lm=self.task_lm):
            for q in questions:
                gold = self.gold[q]
                pred = prog(question=q)
                
                # Record task LM call in budget
                # Note: In real runs, we'd extract token counts from LM response
                self.budget_tracker.record_task_call(
                    input_tokens=0,  # Would extract from response
                    output_tokens=0,
                    metadata={"type": "survival_check", "question": q}
                )
                
                if self.metric(gold, pred, None, None, None).score < 1:
                    return False, questions
        
        # Survived: archive this round's counterexamples
        for q in questions[:len(ids)]:
            if q not in self.archive:
                self.archive = (self.archive + [q])[-self.archive_cap:]
        
        return True, questions
    
    def __call__(
        self,
        candidate: Dict[str, str],
        reflective_dataset: Dict[str, list],
        components_to_update: List[str]
    ) -> Dict[str, str]:
        """GEPA proposal function interface.
        
        Args:
            candidate: Current instruction dict {component_name: instruction}
            reflective_dataset: Minibatch dict {component_name: examples}
            components_to_update: List of component names to update
            
        Returns:
            Updated instruction dict
        """
        out = {}
        
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            
            batch = list(reflective_dataset[name])
            shown = "\n".join(
                f"[{i}] {json.dumps(ex, default=str)}"
                for i, ex in enumerate(batch)
            )
            
            # Critique with reflection LM
            with dspy.context(lm=self.critic_lm):
                critique_result = self.critique(
                    current_instruction=candidate[name],
                    examples_with_feedback=shown
                )
                refutations = critique_result.refutations
            
            # Record critique call
            self.budget_tracker.record_reflection_call(
                input_tokens=0,  # Would extract from response
                output_tokens=0,
                metadata={"type": "critique", "component": name}
            )
            
            # Rewrite attempts
            new = candidate[name]
            for attempt in range(1, self.max_attempts + 1):
                with dspy.context(lm=self.critic_lm):
                    rewrite_result = self.rewrite(
                        current_instruction=candidate[name],
                        refutations=refutations
                    )
                    new = rewrite_result.new_instruction
                
                # Record rewrite call
                self.budget_tracker.record_reflection_call(
                    input_tokens=0,
                    output_tokens=0,
                    metadata={"type": "rewrite", "component": name, "attempt": attempt}
                )
                
                # Survival check
                ok, ids = self.survives(name, new, refutations, batch)
                
                self.log.append({
                    "attempt": attempt,
                    "n_refutations": len(refutations),
                    "checked_ids": ids,
                    "survived": ok,
                    "new_instruction": new,
                })
                
                if ok:
                    break
            
            # Return new instruction if it survived, otherwise return old instruction
            # This rejects failing rewrites immediately instead of letting GEPA test them
            if ok:
                out[name] = new
            else:
                # Keep parent instruction - reject the failing rewrite
                out[name] = self.program.predictors()[name].signature.instructions
        
        return out


class PopperianArm(BaseArm):
    """Arm B: GEPA + typed Popperian proposer."""
    
    def __init__(
        self,
        name: str,
        description: str,
        budget_tracker: BudgetTracker,
        max_metric_calls: int = 200,
        reflection_minibatch_size: int = 3,
        use_merge: bool = False,
        seed: int = 0,
        max_attempts: int = 2,
        severity_tau: float = 0.5,
        archive_cap: int = 6,
        **kwargs
    ):
        """Initialize Popperian arm."""
        super().__init__(name, description, budget_tracker, **kwargs)
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.use_merge = use_merge
        self.seed = seed
        self.max_attempts = max_attempts
        self.severity_tau = severity_tau
        self.archive_cap = archive_cap
        self.proposer: Optional[PopperianProposer] = None
    
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
        """Compile with Popperian proposer.
        
        Args:
            program: Program to optimize
            metric: Evaluation metric
            trainset: Training examples
            valset: Validation examples
            critic_lm: LM for critique/rewrite
            task_lm: LM for task execution
            **compile_kwargs: Additional args
            
        Returns:
            Optimized program
        """
        # Build gold lookup
        gold_by_question = {ex.question: ex for ex in trainset + valset}
        
        # Create proposer
        self.proposer = PopperianProposer(
            program=program,
            metric=metric,
            gold_by_question=gold_by_question,
            critic_lm=critic_lm,
            task_lm=task_lm,
            budget_tracker=self.budget_tracker,
            max_attempts=self.max_attempts,
            severity_tau=self.severity_tau,
            archive_cap=self.archive_cap,
        )
        
        # Create GEPA with custom proposer
        gepa = dspy.GEPA(
            metric=metric,
            instruction_proposer=self.proposer,
            max_metric_calls=self.max_metric_calls,
            reflection_minibatch_size=self.reflection_minibatch_size,
            use_merge=self.use_merge,
            seed=self.seed,
            track_stats=True,
            gepa_kwargs={
                "stop_callbacks": [self.budget_tracker],
                **self.kwargs
            }
        )
        
        # Compile
        optimized = gepa.compile(
            program,
            trainset=trainset,
            valset=valset,
        )
        
        return optimized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get arm statistics including proposer log."""
        stats = super().get_stats()
        if self.proposer:
            stats["proposer_log"] = self.proposer.log
            stats["archive_size"] = len(self.proposer.archive)
        return stats
