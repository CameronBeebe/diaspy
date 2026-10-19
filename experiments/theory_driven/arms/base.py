"""Base class for optimizer arms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import dspy
from ..config import BudgetTracker


class BaseArm(ABC):
    """Base class for all optimizer arms."""
    
    def __init__(
        self,
        name: str,
        description: str,
        budget_tracker: BudgetTracker,
        **kwargs
    ):
        """Initialize arm.
        
        Args:
            name: Arm name (e.g., "A", "B", "K-Jev")
            description: Human-readable description
            budget_tracker: Budget tracker instance
            **kwargs: Arm-specific parameters
        """
        self.name = name
        self.description = description
        self.budget_tracker = budget_tracker
        self.kwargs = kwargs
    
    @abstractmethod
    def compile(
        self,
        program: dspy.Module,
        metric: Callable,
        trainset: list,
        valset: list,
        **compile_kwargs
    ) -> dspy.Module:
        """Compile/optimize the program.
        
        Args:
            program: DSPy program to optimize
            metric: Evaluation metric
            trainset: Training examples
            valset: Validation examples
            **compile_kwargs: Additional compilation arguments
            
        Returns:
            Optimized program
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get arm-specific statistics."""
        return {
            "name": self.name,
            "description": self.description,
            "budget_stats": self.budget_tracker.get_stats(),
        }
