"""Arm C: GEPA + untyped prose critic."""

import dspy
from typing import Callable, List, Dict, Any
from .base import BaseArm
from ..config import BudgetTracker


class ProseCritique(dspy.Signature):
    """Untyped prose critique of instruction failures."""
    current_instruction: str = dspy.InputField()
    examples_with_feedback: str = dspy.InputField(
        desc="examples with their inputs, outputs, and feedback"
    )
    critique: str = dspy.OutputField(
        desc="natural language critique explaining why the instruction fails"
    )


class ProseRewrite(dspy.Signature):
    """Rewrite instruction based on prose critique."""
    current_instruction: str = dspy.InputField()
    critique: str = dspy.InputField()
    new_instruction: str = dspy.OutputField()


class ProseCriticProposer:
    """GEPA ProposalFn with untyped prose critique.
    
    Unlike the Popperian proposer, this uses plain text critique without
    typed Refutation objects and without mechanical survival checks.
    """
    
    def __init__(
        self,
        critic_lm: dspy.LM,
        budget_tracker: BudgetTracker,
    ):
        """Initialize prose critic proposer.
        
        Args:
            critic_lm: LM for critique and rewrite
            budget_tracker: Budget tracker
        """
        self.critic_lm = critic_lm
        self.budget_tracker = budget_tracker
        self.critique = dspy.Predict(ProseCritique)
        self.rewrite = dspy.Predict(ProseRewrite)
        self.log: List[Dict[str, Any]] = []
    
    def __call__(
        self,
        candidate: Dict[str, str],
        reflective_dataset: Dict[str, list],
        components_to_update: List[str]
    ) -> Dict[str, str]:
        """GEPA proposal function interface.
        
        Args:
            candidate: Current instruction dict
            reflective_dataset: Minibatch dict
            components_to_update: Components to update
            
        Returns:
            Updated instruction dict
        """
        out = {}
        
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            
            batch = list(reflective_dataset[name])
            import json
            shown = "\n".join(
                f"Example {i+1}:\n{json.dumps(ex, default=str, indent=2)}"
                for i, ex in enumerate(batch)
            )
            
            # Critique
            with dspy.context(lm=self.critic_lm):
                critique_result = self.critique(
                    current_instruction=candidate[name],
                    examples_with_feedback=shown
                )
            
            self.budget_tracker.record_reflection_call(
                input_tokens=0,
                output_tokens=0,
                metadata={"type": "prose_critique", "component": name}
            )
            
            # Rewrite
            with dspy.context(lm=self.critic_lm):
                rewrite_result = self.rewrite(
                    current_instruction=candidate[name],
                    critique=critique_result.critique
                )
            
            self.budget_tracker.record_reflection_call(
                input_tokens=0,
                output_tokens=0,
                metadata={"type": "prose_rewrite", "component": name}
            )
            
            self.log.append({
                "component": name,
                "critique": critique_result.critique,
                "old_instruction": candidate[name],
                "new_instruction": rewrite_result.new_instruction,
            })
            
            out[name] = rewrite_result.new_instruction
        
        return out


class ProseCriticArm(BaseArm):
    """Arm C: GEPA + untyped prose critic."""
    
    def __init__(
        self,
        name: str,
        description: str,
        budget_tracker: BudgetTracker,
        max_metric_calls: int = 200,
        reflection_minibatch_size: int = 3,
        use_merge: bool = False,
        seed: int = 0,
        **kwargs
    ):
        """Initialize prose critic arm."""
        super().__init__(name, description, budget_tracker, **kwargs)
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.use_merge = use_merge
        self.seed = seed
        self.proposer = None
    
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
        """Compile with prose critic proposer."""
        self.proposer = ProseCriticProposer(
            critic_lm=critic_lm,
            budget_tracker=self.budget_tracker,
        )
        
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
        
        optimized = gepa.compile(
            program,
            trainset=trainset,
            valset=valset,
        )
        
        return optimized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get arm statistics."""
        stats = super().get_stats()
        if self.proposer:
            stats["proposer_log"] = self.proposer.log
        return stats
