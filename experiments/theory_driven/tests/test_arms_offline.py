"""Offline tests for arms using DummyLM.

Tests each arm's plumbing without making real API calls.
"""

import pytest
import dspy
from dspy.utils import DummyLM

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BudgetTracker
from arms import GEPADefaultArm, ProseCriticArm, PopperianArm


class SimpleDummyLM(DummyLM):
    """Simple DummyLM that returns predictable outputs."""
    
    def __init__(self, name="dummy"):
        super().__init__({})
        self.name = name
        self.call_count = 0
    
    def __call__(self, prompt=None, **kwargs):
        self.call_count += 1
        # Return a simple prediction
        return ["Test answer"]


@pytest.fixture
def dummy_lm():
    """Create a dummy LM for testing."""
    return SimpleDummyLM()


@pytest.fixture
def simple_data():
    """Create simple test data."""
    train = [
        dspy.Example(question="Q1", answer="A1").with_inputs("question"),
        dspy.Example(question="Q2", answer="A2").with_inputs("question"),
    ]
    val = [
        dspy.Example(question="Q3", answer="A3").with_inputs("question"),
        dspy.Example(question="Q4", answer="A4").with_inputs("question"),
    ]
    return train, val


@pytest.fixture
def simple_program():
    """Create simple DSPy program."""
    return dspy.Predict("question -> answer")


@pytest.fixture
def simple_metric():
    """Create simple metric."""
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Simple exact match
        match = pred.answer.strip() == gold.answer.strip()
        return dspy.Prediction(score=1.0 if match else 0.0, feedback="test")
    return metric


def test_gepa_default_arm_smoke(dummy_lm, simple_data, simple_program, simple_metric):
    """Smoke test for GEPA default arm with DummyLM."""
    train, val = simple_data
    
    # Create budget tracker
    tracker = BudgetTracker(max_units=20)
    
    # Create arm
    arm = GEPADefaultArm(
        name="A",
        description="Test GEPA",
        budget_tracker=tracker,
        max_metric_calls=10,
    )
    
    # Configure DSPy with dummy LM
    dspy.settings.configure(lm=dummy_lm)
    
    # This should not crash (actual optimization requires real GEPA)
    # For now, just test that arm can be created
    assert arm.name == "A"
    assert arm.budget_tracker == tracker


def test_popperian_arm_smoke(dummy_lm, simple_data, simple_program, simple_metric):
    """Smoke test for Popperian arm with DummyLM."""
    train, val = simple_data
    
    tracker = BudgetTracker(max_units=20)
    
    arm = PopperianArm(
        name="B",
        description="Test Popperian",
        budget_tracker=tracker,
    )
    
    assert arm.name == "B"
    assert arm.budget_tracker == tracker


def test_prose_critic_arm_smoke(dummy_lm, simple_data, simple_program, simple_metric):
    """Smoke test for prose critic arm with DummyLM."""
    train, val = simple_data
    
    tracker = BudgetTracker(max_units=20)
    
    arm = ProseCriticArm(
        name="C",
        description="Test prose critic",
        budget_tracker=tracker,
    )
    
    assert arm.name == "C"
    assert arm.budget_tracker == tracker


def test_budget_tracker():
    """Test budget tracker functionality."""
    tracker = BudgetTracker(max_units=10)
    
    assert tracker.total_units == 0
    assert not tracker.budget_exhausted
    
    # Record some calls
    tracker.record_task_call(input_tokens=100, output_tokens=50)
    assert tracker.total_units == 1
    
    tracker.record_reflection_call(input_tokens=200, output_tokens=100)
    assert tracker.total_units == 2
    
    # Record until exhausted
    for _ in range(8):
        tracker.record_task_call()
    
    assert tracker.total_units == 10
    assert tracker.budget_exhausted
    
    # Test stop callback interface
    assert tracker(None) == True  # Should stop


def test_budget_tracker_cost_accounting():
    """Test budget tracker cost calculation."""
    tracker = BudgetTracker(
        max_units=10,
        cost_per_million_input=0.117,
        cost_per_million_output=0.455,
    )
    
    # Record a call with known token counts
    tracker.record_task_call(input_tokens=1000, output_tokens=500)
    
    # Check cost calculation
    expected_cost = (1000 * 0.117 / 1_000_000) + (500 * 0.455 / 1_000_000)
    assert abs(tracker.total_cost - expected_cost) < 1e-6
    
    stats = tracker.get_stats()
    assert stats["total_units"] == 1
    assert stats["task_lm_calls"] == 1
    assert abs(stats["total_cost"] - expected_cost) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
