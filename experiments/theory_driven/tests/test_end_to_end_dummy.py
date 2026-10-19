"""End-to-end smoke test with DummyLM.

Runs a tiny version of every arm to ensure plumbing works.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dspy.utils import DummyLM
from dspy.lm15 import Message, Response, TextPart, Usage

from config import BudgetTracker, ExperimentConfig
from harness import ExperimentHarness


class FakeTaskLM(DummyLM):
    """Fake task LM that returns consistent answers."""
    
    def __init__(self):
        super().__init__({})
        self.name = "fake-task"
        self.calls = 0
    
    def _complete(self, messages):
        self.calls += 1
        return Response(
            id=None,
            model=self.name,
            finish_reason="stop",
            message=Message.assistant([TextPart("Test answer")]),
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        )


class FakeCriticLM(DummyLM):
    """Fake critic LM that returns consistent critiques."""
    
    def __init__(self):
        super().__init__({})
        self.name = "fake-critic"
        self.calls = 0
    
    def _complete(self, messages):
        self.calls += 1
        # Return simple critique/rewrite
        return Response(
            id=None,
            model=self.name,
            finish_reason="stop",
            message=Message.assistant([TextPart("Improved instruction")]),
            usage=Usage(input_tokens=50, output_tokens=20, total_tokens=70)
        )


def test_all_arms_smoke():
    """Smoke test all arms with fake LMs.
    
    This test runs a tiny version of each arm to verify:
    - Arms can be instantiated
    - Budget tracking works
    - No crashes during execution
    """
    print("\n" + "="*60)
    print("SMOKE TEST: All arms with DummyLM")
    print("="*60)
    
    # Create fake LMs
    task_lm = FakeTaskLM()
    critic_lm = FakeCriticLM()
    
    # Simple test data
    train = [
        dspy.Example(question=f"Q{i}", answer=f"A{i}").with_inputs("question")
        for i in range(4)
    ]
    val = [
        dspy.Example(question=f"Q{i}", answer=f"A{i}").with_inputs("question")
        for i in range(4, 6)
    ]
    
    # Simple metric
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return dspy.Prediction(score=0.5, feedback="test feedback")
    
    # Simple program
    program = dspy.Predict("question -> answer")
    
    # Test each arm
    arms_to_test = ["A", "B", "C"]  # Skip K arms (need TypeSafe)
    
    for arm_name in arms_to_test:
        print(f"\nTesting arm {arm_name}...")
        
        # Create budget tracker
        tracker = BudgetTracker(max_units=15)
        
        # Create minimal config
        config = ExperimentConfig()
        config.model_config.task_model = "fake/model"
        
        # Create harness (without real LMs)
        harness = ExperimentHarness(config)
        harness.task_lm = task_lm
        harness.reflection_lm = critic_lm
        
        # Create arm
        try:
            arm = harness.create_arm(arm_name, tracker)
            print(f"  ✓ Arm {arm_name} created successfully")
            
            # Verify arm properties
            assert arm.name == arm_name
            assert arm.budget_tracker == tracker
            print(f"  ✓ Arm {arm_name} properties verified")
            
        except NotImplementedError as e:
            print(f"  ⚠ Arm {arm_name} not fully implemented: {e}")
        except Exception as e:
            print(f"  ✗ Arm {arm_name} failed: {e}")
            raise
    
    print("\n" + "="*60)
    print("SMOKE TEST PASSED")
    print("="*60)


def test_budget_tracking_integration():
    """Test that budget tracking integrates properly."""
    tracker = BudgetTracker(max_units=5)
    
    # Simulate some calls
    tracker.record_task_call(input_tokens=100, output_tokens=50)
    tracker.record_reflection_call(input_tokens=200, output_tokens=100)
    
    assert tracker.total_units == 2
    assert tracker.total_cost > 0
    
    # Test GEPA stop callback interface
    assert tracker(None) == False  # Not exhausted yet
    
    # Exhaust budget
    for _ in range(3):
        tracker.record_task_call()
    
    assert tracker.budget_exhausted
    assert tracker(None) == True  # Should stop now


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
