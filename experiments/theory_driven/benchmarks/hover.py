"""HoVer benchmark: deterministic recall metric with local BM25S retrieval.

Per spec:
- Primary benchmark
- Deterministic recall metric (no judge needed)
- BM25S over 5.2M Wikipedia-2017 abstracts (500MB)
- Free setup step checks if index fits in memory (fallback to PUPA if not)
"""

import os
import random
from typing import List, Callable, Optional
from pathlib import Path
import dspy
from .base import BaseBenchmark


class HoVerBenchmark(BaseBenchmark):
    """HoVer benchmark with deterministic recall metric."""
    
    def __init__(self):
        """Initialize HoVer benchmark."""
        super().__init__("hover")
        self.retriever = None
    
    def setup(self) -> tuple[bool, str]:
        """Check if BM25S retrieval index fits in memory.
        
        Returns:
            Tuple of (success, message)
        """
        # Check available memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            
            # Per spec: need ~500MB for index, check if we have ~3GB free
            if available_gb < 3.0:
                return False, (
                    f"Insufficient memory: {available_gb:.1f}GB available, "
                    "need ~3GB for BM25S index. Fallback to PUPA."
                )
        except ImportError:
            # psutil not available, assume OK
            pass
        
        # Check if we can import BM25S
        try:
            import bm25s
            return True, "BM25S available"
        except ImportError:
            return False, "BM25S not installed. Fallback to PUPA."
    
    def load_data(
        self,
        train_size: int = 20,
        val_size: int = 20,
        test_size: Optional[int] = None,
    ) -> tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Load HoVer data.
        
        Per spec: reproduce baseline at GEPA paper splits (150/300/300).
        For experiments, use train_size=20, val_size=20.
        
        Args:
            train_size: Training set size
            val_size: Validation set size
            test_size: Test set size (None = full, or subset for pilot)
            
        Returns:
            Tuple of (trainset, valset, testset)
        """
        # Placeholder: In real implementation, would load from dataset
        # For now, create dummy data structure
        
        # Dummy question-answer pairs
        dummy_data = [
            {
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "gold_passages": [f"doc_{i}_1", f"doc_{i}_2"],
            }
            for i in range(1000)  # Enough for any split
        ]
        
        random.seed(42)
        random.shuffle(dummy_data)
        
        # Create splits
        train = dummy_data[:train_size]
        val = dummy_data[train_size:train_size + val_size]
        
        if test_size is None:
            test = dummy_data[train_size + val_size:]
        else:
            test = dummy_data[train_size + val_size:train_size + val_size + test_size]
        
        # Convert to DSPy examples
        trainset = [
            dspy.Example(
                question=ex["question"],
                answer=ex["answer"],
                gold_passages=ex["gold_passages"]
            ).with_inputs("question")
            for ex in train
        ]
        
        valset = [
            dspy.Example(
                question=ex["question"],
                answer=ex["answer"],
                gold_passages=ex["gold_passages"]
            ).with_inputs("question")
            for ex in val
        ]
        
        testset = [
            dspy.Example(
                question=ex["question"],
                answer=ex["answer"],
                gold_passages=ex["gold_passages"]
            ).with_inputs("question")
            for ex in test
        ]
        
        return trainset, valset, testset
    
    def get_program(self) -> dspy.Module:
        """Get HoVer program (simple QA with retrieval)."""
        
        class HoVerQA(dspy.Module):
            def __init__(self, retriever=None):
                super().__init__()
                self.retriever = retriever
                self.qa = dspy.Predict("question -> answer")
            
            def forward(self, question):
                # In real implementation, would retrieve passages
                # For now, just answer directly
                return self.qa(question=question)
        
        return HoVerQA(self.retriever)
    
    def get_metric(self) -> Callable:
        """Get HoVer metric: deterministic recall of gold passages.
        
        Per spec: feedback names gold documents found and missing.
        """
        def hover_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            # Extract retrieved passages (in real implementation)
            retrieved = getattr(pred, "retrieved_passages", [])
            gold_passages = getattr(gold, "gold_passages", [])
            
            # Calculate recall
            found = [p for p in gold_passages if p in retrieved]
            missing = [p for p in gold_passages if p not in retrieved]
            
            recall = len(found) / len(gold_passages) if gold_passages else 0.0
            
            feedback = {
                "recall": recall,
                "found": found,
                "missing": missing,
            }
            
            return dspy.Prediction(
                score=recall,
                feedback=str(feedback)
            )
        
        return hover_metric
