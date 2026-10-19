"""Arm A: GEPA with default InstructionProposer."""

import dspy
from typing import Callable
from .base import BaseArm
from ..config import BudgetTracker


class GEPADefaultArm(BaseArm):
    """GEPA with default InstructionProposer (baseline)."""
    
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
        **compile_kwargs
    ) -> dspy.Module:
        """Compile with GEPA default.
        
        Per spec: GEPA can overshoot by one iteration when using stop_callbacks.
        The budget tracker is passed as a stop callback to halt when budget exhausted.
        """
        # Create GEPA optimizer with budget tracker as stop callback
        gepa = dspy.GEPA(
            metric=metric,
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
