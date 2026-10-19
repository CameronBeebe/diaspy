"""IFBench task setup and evaluation.

Minimal implementation for testing. Real IFBench data/verifiers would go here.
"""
import dspy
from typing import Any, List, Optional
import dataclasses
import re


@dataclasses.dataclass
class IFBenchExample:
    """One IFBench example with constraints."""
    question: str
    answer: str
    constraints: List[str]
    constraint_kwargs: dict


class IFBenchProgram(dspy.Module):
    """Simple single-stage program for testing."""
    
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")
    
    def forward(self, question: str) -> dspy.Prediction:
        ans = self.answer(question=question)
        return dspy.Prediction(answer=ans.answer)


def ifbench_metric(gold: Any, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    """Check if answer matches gold and constraints satisfied.
    
    For testing, simplified to exact match. Real version would use verifiers.
    """
    # Use refined_answer if present, else answer
    pred_ans = getattr(pred, "refined_answer", getattr(pred, "answer", ""))
    gold_ans = getattr(gold, "answer", "")
    
    # Exact match for testing
    match = pred_ans.strip().lower() == gold_ans.strip().lower()
    
    # Placeholder constraint checking
    feedback = "correct" if match else f"wrong: expected '{gold_ans}', got '{pred_ans}'"
    
    return dspy.Prediction(score=float(match), feedback=feedback)


# ========== Dummy data for offline testing ==========

def create_dummy_ifbench_data(n: int = 10, seed: int = 42) -> List[dspy.Example]:
    """Generate fake IFBench-like data for testing."""
    import random
    random.seed(seed)
    
    questions = [
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("How many legs does a spider have?", "8"),
        ("What color is the sky?", "blue"),
        ("What is 10-3?", "7"),
        ("Who wrote Hamlet?", "Shakespeare"),
        ("What is 5*5?", "25"),
        ("What is the boiling point of water?", "100"),
        ("How many days in a week?", "7"),
        ("What is the square root of 16?", "4"),
    ]
    
    sampled = random.sample(questions, min(n, len(questions)))
    
    examples = []
    for q, a in sampled:
        ex = dspy.Example(
            question=q,
            answer=a,
            constraints=["be concise", "include units if applicable"],
            constraint_kwargs={}
        ).with_inputs("question")
        examples.append(ex)
    
    return examples


def create_perturbed_testset(testset: List[dspy.Example]) -> List[dspy.Example]:
    """Create deterministically perturbed test set.
    
    As per design: constraint moved, whitespace/case noise, etc.
    For testing, simple perturbations.
    """
    perturbed = []
    for ex in testset:
        # Add whitespace and case variation
        q_perturbed = ex.question.upper() + "  "
        
        ex_perturbed = dspy.Example(
            question=q_perturbed,
            answer=ex.answer,
            constraints=ex.constraints,
            constraint_kwargs=ex.constraint_kwargs
        ).with_inputs("question")
        
        perturbed.append(ex_perturbed)
    
    return perturbed
