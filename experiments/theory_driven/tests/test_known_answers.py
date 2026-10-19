"""Known-answer tests for experiment infrastructure.

Tests that fundamental components work correctly:
- Metric scores correct > incorrect
- Budget tracker counts all call types
- Cap aborts correctly
"""

import pytest

import dspy
from experiments.theory_driven.config import BudgetTracker, BudgetEnforcer, CostLimitExceeded


def test_metric_scores_correctness():
    """Test that metric scores correct answer higher than incorrect."""
    # Simple example
    gold = dspy.Example(answer="42")
    
    correct_pred = dspy.Prediction(answer="42")
    incorrect_pred = dspy.Prediction(answer="wrong")
    
    # Simple exact-match metric
    def metric(gold, pred, *args):
        score = 1.0 if gold.answer == pred.answer else 0.0
        return dspy.Prediction(score=score, feedback="test")
    
    correct_score = metric(gold, correct_pred).score
    incorrect_score = metric(gold, incorrect_pred).score
    
    assert correct_score > incorrect_score, "Metric should score correct > incorrect"
    assert correct_score == 1.0, "Correct should score 1.0"
    assert incorrect_score == 0.0, "Incorrect should score 0.0"


def test_budget_tracker_counts_all_call_types():
    """Test that budget tracker counts task, reflection, and Jev calls."""
    tracker = BudgetTracker(max_units=100)
    
    # Record different call types
    tracker.record_task_call(input_tokens=100, output_tokens=50)
    tracker.record_reflection_call(input_tokens=200, output_tokens=100)
    tracker.record_jev_call(input_tokens=150, output_tokens=0, cost=0.0002)
    
    # Check counts
    assert tracker.total_units == 3, "Should count 3 units total"
    
    stats = tracker.get_stats()
    assert stats["task_lm_calls"] == 1
    assert stats["reflection_lm_calls"] == 1
    assert stats["jev_calls"] == 1
    assert stats["total_units"] == 3


def test_budget_tracker_cost_accounting():
    """Test cost calculation with provider-reported costs."""
    tracker = BudgetTracker(
        max_units=10,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
    )
    
    # Test with provider-reported cost (preferred)
    tracker.record_task_call(
        input_tokens=1000,
        output_tokens=500,
        provider_cost=0.001234
    )
    
    assert abs(tracker.total_cost - 0.001234) < 1e-6, "Should use provider cost"
    
    # Test with fallback calculation
    tracker2 = BudgetTracker(
        max_units=10,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
    )
    
    tracker2.record_task_call(input_tokens=1000, output_tokens=500)
    
    expected = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
    assert abs(tracker2.total_cost - expected) < 1e-6, "Fallback calculation should work"


def test_budget_cap_aborts():
    """Test that budget cap correctly triggers abort."""
    tracker = BudgetTracker(max_units=5)
    enforcer = BudgetEnforcer(max_cost_dollars=0.001, tracker=tracker)
    
    # Below cap - should not abort
    tracker.record_task_call(input_tokens=10, output_tokens=5)
    enforcer.check()  # Should not raise
    
    assert not tracker.budget_exhausted
    
    # Exhaust budget
    for _ in range(4):
        tracker.record_task_call()
    
    assert tracker.budget_exhausted
    
    # Stopper should return True (stop)
    assert tracker(None)


def test_cost_limit_enforcer():
    """Test that cost limit enforcer raises exception."""
    tracker = BudgetTracker(max_units=100, cost_per_million_input=1.0, cost_per_million_output=1.0)
    enforcer = BudgetEnforcer(max_cost_dollars=0.01, tracker=tracker)
    
    # Add calls that exceed cost limit
    for _ in range(20):
        tracker.record_task_call(input_tokens=1000, output_tokens=1000)
    
    # Should raise CostLimitExceeded
    with pytest.raises(CostLimitExceeded):
        enforcer.check()


def test_budget_call_log():
    """Test that call log records all calls."""
    tracker = BudgetTracker(max_units=10)
    
    tracker.record_task_call(input_tokens=100, output_tokens=50, metadata={"step": 1})
    tracker.record_reflection_call(input_tokens=200, output_tokens=100, metadata={"step": 2})
    
    log = tracker.get_call_log()
    
    assert len(log) == 2
    assert log[0]["type"] == "task_lm"
    assert log[0]["metadata"]["step"] == 1
    assert log[1]["type"] == "reflection_lm"
    assert log[1]["metadata"]["step"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
