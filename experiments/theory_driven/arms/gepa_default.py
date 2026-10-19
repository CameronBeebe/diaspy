"""Arm A: GEPA with default InstructionProposer."""

import dspy
from typing import Callable
from experiments.theory_driven.arms.base import BaseArm
from experiments.theory_driven.arms.gepa_trace import GEPAProposalTrace
from experiments.theory_driven.config.budget_tracker import BudgetTracker, UnifiedBudget


class GEPADefaultArm(BaseArm):
    """GEPA with default InstructionProposer (baseline)."""
    
    def __init__(
        self,
        name: str,
        description: str,
        budget_tracker: BudgetTracker | UnifiedBudget,
        max_metric_calls: int = 200,
        reflection_minibatch_size: int = 3,
        use_merge: bool = False,
        seed: int = 0,
        **kwargs
    ):
        """Initialize GEPA default arm.
        
        Args:
            name: Arm name
            description: Description
            budget_tracker: Budget tracker
            max_metric_calls: Maximum metric calls (will be limited by budget tracker)
            reflection_minibatch_size: Size of reflection minibatches
            use_merge: Whether to use merge in GEPA
            seed: Random seed
            **kwargs: Additional GEPA parameters
        """
        super().__init__(name, description, budget_tracker, **kwargs)
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.use_merge = use_merge
        self.seed = seed
    
    def compile(
        self,
        program: dspy.Module,
        metric: Callable,
        trainset: list,
        valset: list,
        task_lm=None,
        reflection_lm=None,
        event_sink: Callable | None = None,
        **compile_kwargs
    ) -> dspy.Module:
        """Compile with GEPA default.
        
        Per spec: GEPA can overshoot by one iteration when using stop_callbacks.
        The budget tracker is passed as a stop callback to halt when budget exhausted.
        
        Args:
            program: DSPy program to optimize
            metric: Metric function
            trainset: Training examples
            valset: Validation examples
            task_lm: Task language model (optional, uses dspy.settings.lm if not provided)
            reflection_lm: Reflection language model (required by GEPA)
            **compile_kwargs: Additional compile arguments
        """
        # GEPA requires a reflection_lm
        if reflection_lm is None:
            reflection_lm = dspy.settings.lm

        self.proposer = GEPAProposalTrace(event_sink)
        callbacks = [*(self.kwargs.get("callbacks") or []), self.proposer]
        
        # Create GEPA optimizer with budget tracker as stop callback
        # Note: max_metric_calls should equal budget, not hard-coded 200
        gepa = dspy.GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            instruction_proposer=self.proposer,
            max_metric_calls=self.max_metric_calls,
            reflection_minibatch_size=self.reflection_minibatch_size,
            use_merge=self.use_merge,
            seed=self.seed,  # Pass seed to GEPA
            track_stats=True,
            num_threads=1,
            gepa_kwargs={**self.kwargs, "stop_callbacks": [self.budget_tracker],
                         "callbacks": callbacks},
        )
        
        # Compile
        try:
            with dspy.context(lm=task_lm if task_lm is not None else dspy.settings.lm):
                optimized = gepa.compile(
                    program,
                    trainset=trainset,
                    valset=valset,
                )
        finally:
            self.proposer.flush()
        
        return optimized
