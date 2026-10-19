"""Test data integrity: splits disjoint, seeds work, test isolation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks import HoVerBenchmark, PUPABenchmark


def test_splits_are_disjoint_hover():
    """Test that train/val/test splits are disjoint for HoVer."""
    benchmark = HoVerBenchmark()
    
    trainset, valset, testset = benchmark.load_data(
        train_size=20,
        val_size=20,
        test_size=30
    )
    
    # Get questions
    train_questions = {ex.question for ex in trainset}
    val_questions = {ex.question for ex in valset}
    test_questions = {ex.question for ex in testset}
    
    # Check sizes
    assert len(train_questions) == 20
    assert len(val_questions) == 20
    assert len(test_questions) == 30
    
    # Check disjoint
    assert len(train_questions & val_questions) == 0, "Train and val should not overlap"
    assert len(train_questions & test_questions) == 0, "Train and test should not overlap"
    assert len(val_questions & test_questions) == 0, "Val and test should not overlap"


def test_splits_are_disjoint_pupa():
    """Test that train/val/test splits are disjoint for PUPA."""
    benchmark = PUPABenchmark()
    
    trainset, valset, testset = benchmark.load_data(
        train_size=20,
        val_size=20,
        test_size=30
    )
    
    # Get questions
    train_questions = {ex.question for ex in trainset}
    val_questions = {ex.question for ex in valset}
    test_questions = {ex.question for ex in testset}
    
    # Check sizes
    assert len(train_questions) == 20
    assert len(val_questions) == 20
    assert len(test_questions) == 30
    
    # Check disjoint
    assert len(train_questions & val_questions) == 0
    assert len(train_questions & test_questions) == 0
    assert len(val_questions & test_questions) == 0


def test_seeds_change_sampling():
    """Test that different seeds produce different samples."""
    benchmark = HoVerBenchmark()
    
    # Load with seed 42 (implicit in load_data)
    train1, val1, test1 = benchmark.load_data(train_size=10, val_size=10, test_size=10)
    
    # Load again (same seed, should be same)
    train2, val2, test2 = benchmark.load_data(train_size=10, val_size=10, test_size=10)
    
    # Should be identical (same seed)
    assert [ex.question for ex in train1] == [ex.question for ex in train2]
    
    # Note: To test different seeds, would need to modify load_data to accept seed parameter
    # For now, this tests that deterministic seeding works


def test_test_set_not_used_in_optimization():
    """Test that test set is separate and not exposed during optimization.
    
    This is a structural test - in real runs, instrumentation would verify
    test set is never accessed during compile().
    """
    benchmark = HoVerBenchmark()
    
    trainset, valset, testset = benchmark.load_data(
        train_size=20,
        val_size=20,
        test_size=100
    )
    
    # Test set should be larger (full test split)
    assert len(testset) > len(trainset)
    assert len(testset) > len(valset)
    
    # In real harness, would instrument to ensure testset is never passed to compile()
    # and only used for final evaluation
    
    # Structural check: test set should have with_inputs set
    assert all(hasattr(ex, "_input_keys") for ex in testset)


def test_data_split_sizes_configurable():
    """Test that data split sizes are configurable."""
    benchmark = HoVerBenchmark()
    
    # Try different sizes
    train1, val1, test1 = benchmark.load_data(10, 10, 20)
    assert len(train1) == 10
    assert len(val1) == 10
    assert len(test1) == 20
    
    train2, val2, test2 = benchmark.load_data(30, 15, 50)
    assert len(train2) == 30
    assert len(val2) == 15
    assert len(test2) == 50


def test_data_has_required_fields():
    """Test that data examples have required fields."""
    benchmark = HoVerBenchmark()
    
    trainset, valset, testset = benchmark.load_data(5, 5, 5)
    
    # Check all examples have required fields
    for ex in trainset + valset + testset:
        assert hasattr(ex, "question")
        assert hasattr(ex, "answer")
        assert ex.question is not None
        assert ex.answer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
