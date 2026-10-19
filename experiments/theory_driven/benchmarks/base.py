"""Base benchmark class."""

from abc import ABC, abstractmethod
from typing import List, Callable, Any, Optional
import dspy


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, name: str):
        """Initialize benchmark.
        
        Args:
            name: Benchmark name
        """
        self.name = name
    
    @abstractmethod
    def load_data(
        self,
        train_size: int = 20,
        val_size: int = 20,
        test_size: Optional[int] = None,
    ) -> tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Load train, val, test splits.
        
        Args:
            train_size: Number of training examples
            val_size: Number of validation examples
            test_size: Number of test examples (None = full test split)
            
        Returns:
            Tuple of (trainset, valset, testset)
        """
        pass
    
    @abstractmethod
    def get_program(self) -> dspy.Module:
        """Get the task program."""
        pass
    
    @abstractmethod
    def get_metric(self) -> Callable:
        """Get the evaluation metric."""
        pass
    
    @abstractmethod
    def setup(self) -> tuple[bool, str]:
        """Run benchmark setup (e.g., check retrieval index).
        
        Returns:
            Tuple of (success, message)
        """
        pass
