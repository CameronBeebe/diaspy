"""PUPA benchmark: secondary benchmark with LLM judge.

Per spec:
- Secondary benchmark
- Free data (Columbia-NLP/PUPA)
- LLM judge (3 calls per rollout, makes it noisier and costlier)
- Judge is same model as critic
"""

import random
from typing import List, Callable, Optional
import dspy
from .base import BaseBenchmark


class PUPABenchmark(BaseBenchmark):
    """PUPA benchmark with LLM judge."""
    
    def __init__(self):
        """Initialize PUPA benchmark."""
        super().__init__("pupa")
    
    def setup(self) -> tuple[bool, str]:
        """PUPA setup is trivial (just load data)."""
        return True, "PUPA data is free and readily available"
    
    def load_data(
        self,
        train_size: int = 20,
        val_size: int = 20,
        test_size: Optional[int] = None,
    ) -> tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Load PUPA data.
        
        Per spec: GEPA paper used 111/111/221 splits.
        For experiments, use train_size=20, val_size=20.
        
        Args:
            train_size: Training set size
            val_size: Validation set size
            test_size: Test set size (None = full)
            
        Returns:
            Tuple of (trainset, valset, testset)
        """
        # Placeholder: In real implementation, would load from Columbia-NLP/PUPA
        # For now, create dummy data
        
        dummy_data = [
            {
                "question": f"PUPA Question {i}",
                "answer": f"PUPA Answer {i}",
                "context": f"PUPA Context {i}",
            }
            for i in range(500)
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
                context=ex["context"]
            ).with_inputs("question", "context")
            for ex in train
        ]
        
        valset = [
            dspy.Example(
                question=ex["question"],
                answer=ex["answer"],
                context=ex["context"]
            ).with_inputs("question", "context")
            for ex in val
        ]
        
        testset = [
            dspy.Example(
                question=ex["question"],
                answer=ex["answer"],
                context=ex["context"]
            ).with_inputs("question", "context")
            for ex in test
        ]
        
        return trainset, valset, testset
    
    def get_program(self) -> dspy.Module:
        """Get PUPA program."""
        
        class PUPAQA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.Predict("context, question -> answer")
            
            def forward(self, question, context):
                return self.qa(context=context, question=question)
        
        return PUPAQA()
    
    def get_metric(self, judge_lm: Optional[dspy.LM] = None) -> Callable:
        """Get PUPA metric with LLM judge.
        
        Args:
            judge_lm: LM to use as judge (same as critic/reflection LM)
            
        Returns:
            Metric function
        """
        
        class JudgePUPA(dspy.Signature):
            """Judge the quality of an answer to a PUPA question."""
            question: str = dspy.InputField()
            context: str = dspy.InputField()
            gold_answer: str = dspy.InputField()
            pred_answer: str = dspy.InputField()
            score: float = dspy.OutputField(desc="score from 0 to 1")
            feedback: str = dspy.OutputField(desc="explanation of score")
        
        judge = dspy.Predict(JudgePUPA) if judge_lm else None
        
        def pupa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            if judge is None:
                # Fallback: simple string match
                score = 1.0 if gold.answer.strip() == pred.answer.strip() else 0.0
                feedback = "exact match" if score > 0 else "no match"
            else:
                # Use LLM judge (3 calls per rollout per spec)
                # Note: In real implementation, would use judge_lm context
                with dspy.context(lm=judge_lm) if judge_lm else nullcontext():
                    result = judge(
                        question=gold.question,
                        context=gold.context,
                        gold_answer=gold.answer,
                        pred_answer=pred.answer
                    )
                    score = result.score
                    feedback = result.feedback
            
            return dspy.Prediction(score=score, feedback=feedback)
        
        return pupa_metric
