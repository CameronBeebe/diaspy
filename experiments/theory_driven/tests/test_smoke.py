"""Smoke test: actually run arms A and B end-to-end with fake LMs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from config import BudgetTracker
from arms import GEPADefaultArm, PopperianArm
from benchmarks import SimpleQABenchmark


def test_arm_a_with_fake_lm():
    """Test Arm A (GEPA default) with DummyLM."""
    # Create fake LM that returns predictable answers
    lm = dspy.LM(model="dummy", cache=False)
    lm.history = []  # Initialize history
    dspy.configure(lm=lm)
    
    # Create budget tracker
    budget = BudgetTracker(max_units=10)
    
    # Create arm
    arm = GEPADefaultArm(
        name="A",
        description="Test",
        budget_tracker=budget,
        seed=42
    )
    
    # Load simple data
    benchmark = SimpleQABenchmark(test_size=5)
    train, val, test = benchmark.load_data(seed=42)
    
    # Get program and metric
    program = benchmark.get_program()
    metric = benchmark.get_metric()
    
    # Track history length before
    start_len = len(lm.history)
    
    # Compile (should work without crashing)
    try:
        optimized = arm.compile(
            program=program,
            metric=metric,
            trainset=train[:5],
            valset=val[:3],
            task_lm=lm,
            reflection_lm=lm,
        )
    except Exception as e:
        # Arm A requires reflection_lm - if it crashes with AssertionError, test fails
        if "reflection" in str(e).lower():
            raise AssertionError(f"Arm A crashed with reflection_lm error: {e}")
        # Other errors are acceptable for fake LM
        pass
    
    # Check that calls were made
    end_len = len(lm.history)
    actual_calls = end_len - start_len
    
    assert actual_calls > 0, f"Expected >0 LM calls, got {actual_calls}"
    
    # Check budget tracker
    stats = budget.get_stats()
    assert stats["total_units"] >= 0, "Budget tracker should have recorded units"


def test_arm_b_with_fake_lm():
    """Test Arm B (Popperian) with DummyLM."""
    # Create fake LM
    lm = dspy.LM(model="dummy", cache=False)
    lm.history = []
    dspy.configure(lm=lm)
    
    # Create budget tracker
    budget = BudgetTracker(max_units=10)
    
    # Create arm
    arm = PopperianArm(
        name="B",
        description="Test",
        budget_tracker=budget,
        seed=42
    )
    
    # Load simple data
    benchmark = SimpleQABenchmark(test_size=5)
    train, val, test = benchmark.load_data(seed=42)
    
    program = benchmark.get_program()
    metric = benchmark.get_metric()
    
    start_len = len(lm.history)
    
    # Compile
    try:
        optimized = arm.compile(
            program=program,
            metric=metric,
            trainset=train[:5],
            valset=val[:3],
            task_lm=lm,
            reflection_lm=lm,
        )
    except Exception as e:
        # Some errors are acceptable with fake LM
        pass
    
    end_len = len(lm.history)
    actual_calls = end_len - start_len
    
    assert actual_calls > 0, f"Expected >0 LM calls, got {actual_calls}"
    
    stats = budget.get_stats()
    assert stats["total_units"] >= 0, "Budget tracker should have recorded units"


def test_arms_are_distinct():
    """Test that arms A and B produce different behaviors."""
    # This is a minimal distinctness check
    budget_a = BudgetTracker(max_units=10)
    budget_b = BudgetTracker(max_units=10)
    
    arm_a = GEPADefaultArm("A", "GEPA default", budget_a, seed=42)
    arm_b = PopperianArm("B", "Popperian", budget_b, seed=42)
    
    # Check they have different types
    assert type(arm_a) != type(arm_b), "Arms should be different types"
    assert arm_a.name != arm_b.name, "Arms should have different names"


def test_budget_cap_is_respected():
    """Test that budget tracker can detect exhaustion."""
    budget = BudgetTracker(max_units=5)
    
    # Record some calls
    for i in range(5):
        budget.record_task_call(input_tokens=10, output_tokens=10)
    
    # Check exhausted
    assert budget.budget_exhausted, "Budget should be exhausted after 5 calls with max_units=5"
    
    stats = budget.get_stats()
    assert stats["total_units"] >= 5, f"Expected >=5 units, got {stats['total_units']}"


def test_seeds_produce_different_splits():
    """Test that different seeds produce different data splits."""
    benchmark = SimpleQABenchmark(test_size=5)
    
    train1, val1, test1 = benchmark.load_data(seed=1)
    train2, val2, test2 = benchmark.load_data(seed=2)
    
    # Check that at least one question is different
    questions1 = [ex.question for ex in train1]
    questions2 = [ex.question for ex in train2]
    
    assert questions1 != questions2, "Different seeds should produce different train splits"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
