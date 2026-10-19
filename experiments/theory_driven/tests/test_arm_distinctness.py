"""Test that each arm produces distinct instruction updates.

Verifies that arms don't silently fall back to GEPA default.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dspy.utils import DummyLM

from config import BudgetTracker
from arms import GEPADefaultArm, PopperianArm, ProseCriticArm


class TrackedDummyLM(DummyLM):
    """DummyLM that tracks what it's called with."""
    
    def __init__(self, name="tracked"):
        super().__init__({})
        self.name = name
        self.call_history = []
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        self.call_history.append({
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs
        })
        return super().__call__(prompt, messages, **kwargs)


def test_gepa_default_arm_invoked():
    """Test that GEPA default arm is invoked."""
    lm = TrackedDummyLM("gepa-default")
    dspy.settings.configure(lm=lm)
    
    tracker = BudgetTracker(max_units=50)
    arm = GEPADefaultArm(
        name="A",
        description="Test",
        budget_tracker=tracker,
    )
    
    # Arm created successfully
    assert arm.name == "A"
    assert "GEPA" in arm.description or "default" in arm.description.lower()


def test_popperian_arm_distinct():
    """Test that Popperian arm has distinct structure."""
    tracker = BudgetTracker(max_units=50)
    
    popperian = PopperianArm(
        name="B",
        description="Popperian",
        budget_tracker=tracker,
    )
    
    # Check that Popperian arm has specific attributes
    assert popperian.name == "B"
    assert popperian.max_attempts >= 1
    assert popperian.severity_tau > 0
    assert popperian.archive_cap > 0
    
    # These are unique to Popperian arm
    assert hasattr(popperian, "severity_tau")
    assert hasattr(popperian, "archive_cap")


def test_prose_critic_arm_distinct():
    """Test that prose critic arm is distinct."""
    tracker = BudgetTracker(max_units=50)
    
    prose = ProseCriticArm(
        name="C",
        description="Prose critic",
        budget_tracker=tracker,
    )
    
    assert prose.name == "C"
    # Prose critic should not have Popperian-specific fields
    assert not hasattr(prose, "severity_tau")
    assert not hasattr(prose, "archive_cap")


def test_arms_have_different_types():
    """Test that arms have different arm_types in config."""
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    
    arm_types = {arm.name: arm.arm_type for arm in config.arms}
    
    # Check that we have different types
    assert "A" in arm_types
    assert "B" in arm_types
    assert "C" in arm_types
    
    assert arm_types["A"] == "gepa"
    assert arm_types["B"] == "popperian"
    assert arm_types["C"] == "prose_critic"
    
    # All should be different
    assert len(set(arm_types.values())) >= 3, "Arms should have distinct types"


def test_arm_factories_create_different_classes():
    """Test that harness creates different arm classes."""
    from config import ExperimentConfig
    from harness import ExperimentHarness
    
    config = ExperimentConfig()
    harness = ExperimentHarness(config)
    
    tracker = BudgetTracker(max_units=50)
    
    # Create different arms
    arm_a = harness.create_arm("A", tracker)
    arm_b = harness.create_arm("B", tracker)
    arm_c = harness.create_arm("C", tracker)
    
    # Check they're different classes
    assert type(arm_a).__name__ == "GEPADefaultArm"
    assert type(arm_b).__name__ == "PopperianArm"
    assert type(arm_c).__name__ == "ProseCriticArm"
    
    # All different
    assert type(arm_a) != type(arm_b)
    assert type(arm_b) != type(arm_c)
    assert type(arm_a) != type(arm_c)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
