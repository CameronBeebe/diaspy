"""Simple QA benchmark with deterministic exact-match metric.

This is a minimal real benchmark for smoke testing without requiring
complex retrieval infrastructure like BM25S.
"""

import dspy
from typing import List, Callable
from .base import BaseBenchmark


# Real QA examples with unambiguous answers
SIMPLE_QA_DATA = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What color is the sky on a clear day?", "answer": "blue"},
    {"question": "How many legs does a spider have?", "answer": "8"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "How many days are in a week?", "answer": "7"},
    {"question": "What is the opposite of hot?", "answer": "cold"},
    {"question": "How many months are in a year?", "answer": "12"},
    {"question": "What shape has three sides?", "answer": "triangle"},
    {"question": "What is frozen water called?", "answer": "ice"},
    {"question": "How many wheels does a bicycle have?", "answer": "2"},
    {"question": "What is the first month of the year?", "answer": "January"},
    {"question": "How many hours are in a day?", "answer": "24"},
    {"question": "What do bees make?", "answer": "honey"},
    {"question": "What is the opposite of up?", "answer": "down"},
    {"question": "How many sides does a square have?", "answer": "4"},
    {"question": "What color are most trees' leaves?", "answer": "green"},
    {"question": "How many cents are in a dollar?", "answer": "100"},
    {"question": "What is the opposite of day?", "answer": "night"},
    {"question": "How many seasons are in a year?", "answer": "4"},
    {"question": "What do we call frozen rain?", "answer": "snow"},
]


class SimpleQA(dspy.Signature):
    """Answer simple questions."""
    question = dspy.InputField()
    answer = dspy.OutputField()


class SimpleQABenchmark(BaseBenchmark):
    """Simple QA benchmark for smoke testing."""
    
    def __init__(self, test_size: int = 20):
        """Initialize simple QA benchmark.
        
        Args:
            test_size: Number of test examples
        """
        super().__init__(
            name="simple_qa",
            description="Simple QA with exact-match metric",
            test_size=test_size
        )
    
    def load_data(self, seed: int = 42) -> tuple:
        """Load simple QA data.
        
        Args:
            seed: Random seed for splitting
            
        Returns:
            (train, val, test) tuple of example lists
        """
        import random
        random.seed(seed)
        
        # Convert to dspy.Example
        examples = [
            dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question")
            for item in SIMPLE_QA_DATA
        ]
        
        # Shuffle
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Split: 50% train, 25% val, 25% test
        n = len(shuffled)
        train_end = n // 2
        val_end = train_end + n // 4
        
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]
        
        # Limit test size
        if len(test) > self.test_size:
            test = test[:self.test_size]
        
        return train, val, test
    
    def get_program(self) -> dspy.Module:
        """Get simple QA program.
        
        Returns:
            Simple Predict program
        """
        return dspy.Predict(SimpleQA)
    
    def get_metric(self) -> Callable:
        """Get exact-match metric.
        
        Returns:
            Metric function that returns 0 or 1
        """
        def exact_match(example, pred, trace=None):
            """Exact match metric (case-insensitive).
            
            Args:
                example: Gold example with 'answer' field
                pred: Prediction object with 'answer' field
                trace: Optional trace
                
            Returns:
                1 if exact match, 0 otherwise
            """
            if not hasattr(pred, 'answer') or pred.answer is None:
                return 0
            
            gold = str(example.answer).strip().lower()
            pred_text = str(pred.answer).strip().lower()
            
            return 1 if gold == pred_text else 0
        
        return exact_match
    
    def setup(self, experiment_dir) -> None:
        """Setup (nothing needed for simple QA)."""
        pass
