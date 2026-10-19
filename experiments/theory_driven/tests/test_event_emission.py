"""Test that every arm emits well-formed epistemic events."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from events import (
    EventEmitter,
    Conjecture,
    TestSelection,
    TestOutcome,
    AcceptRejectDecision,
    LakatosLabel,
)


def test_event_emitter_basic():
    """Test basic event emitter functionality."""
    emitter = EventEmitter(
        experiment_name="test_exp",
        benchmark_name="test_bench",
        run_id="test_run_001",
        arm_name="test_arm"
    )
    
    # Emit initialization
    init_event = emitter.emit_initialization({"test": "data"})
    
    assert init_event.event_type == "initialization"
    assert init_event.experiment_name == "test_exp"
    assert init_event.benchmark_name == "test_bench"
    assert init_event.run_id == "test_run_001"
    assert init_event.metadata == {"test": "data"}


def test_event_emitter_step():
    """Test step event emission."""
    emitter = EventEmitter(
        experiment_name="test_exp",
        benchmark_name="test_bench",
        run_id="test_run_001",
        arm_name="test_arm"
    )
    
    # Create step components
    conjecture = Conjecture(
        candidate_id="candidate_001",
        parent_id="root",
        module_name="test_module",
        old_instruction="Old instruction",
        new_instruction="New instruction",
        generated_by="test_arm",
    )
    
    test_selection = TestSelection(
        test_type="minibatch",
        example_ids=[0, 1, 2],
        example_count=3,
        selection_rationale="Random minibatch",
        test_cost_units=3,
    )
    
    test_outcome = TestOutcome(
        passed=True,
        score=0.8,
        example_scores=[0.7, 0.9, 0.8],
    )
    
    decision = AcceptRejectDecision(
        accepted=True,
        decision_reason="Score improved",
        minibatch_score=0.8,
    )
    
    # Emit step
    step_event = emitter.emit_step(
        conjecture=conjecture,
        test_selection=test_selection,
        test_outcome=test_outcome,
        decision=decision,
        calls_consumed=5,
        cost_consumed=0.01,
    )
    
    assert step_event.event_type == "optimization_step"
    assert step_event.step is not None
    assert step_event.step.arm_name == "test_arm"
    assert step_event.step.iteration == 1
    assert step_event.step.conjecture.candidate_id == "candidate_001"
    assert step_event.step.decision.accepted == True


def test_event_validation():
    """Test that pydantic validation works."""
    # Valid conjecture
    conjecture = Conjecture(
        candidate_id="c001",
        module_name="mod",
        old_instruction="old",
        new_instruction="new",
        generated_by="arm_a",
    )
    
    assert conjecture.candidate_id == "c001"
    
    # Invalid severity (should fail validation)
    from events.schema import Refutation
    
    with pytest.raises(Exception):  # Pydantic ValidationError
        Refutation(
            claim="test",
            counterexample_ids=[1, 2],
            severity=1.5,  # Invalid: >1
            proposed_fix="fix"
        )


def test_lakatos_assessment():
    """Test Lakatos label assessment."""
    emitter = EventEmitter(
        experiment_name="test",
        benchmark_name="test",
        run_id="test",
        arm_name="test"
    )
    
    conjecture = Conjecture(
        candidate_id="c001",
        module_name="mod",
        old_instruction="old",
        new_instruction="new",
        generated_by="test",
    )
    
    # Progressive case: full validation passed
    test_selection = TestSelection(
        test_type="full_validation",
        example_ids=list(range(20)),
        example_count=20,
        selection_rationale="Full validation",
        test_cost_units=20,
    )
    
    test_outcome = TestOutcome(
        passed=True,
        score=0.9,
        example_scores=[0.9] * 20,
    )
    
    label, rationale = emitter.assess_lakatos_label(
        test_outcome, test_selection, conjecture
    )
    
    assert label == LakatosLabel.PROGRESSIVE
    assert "beyond counterexamples" in rationale.lower()
    
    # Degenerating case: failed validation
    test_outcome_failed = TestOutcome(
        passed=False,
        score=0.3,
        example_scores=[0.3] * 20,
    )
    
    label, rationale = emitter.assess_lakatos_label(
        test_outcome_failed, test_selection, conjecture
    )
    
    assert label == LakatosLabel.DEGENERATING
    assert "overfitting" in rationale.lower() or "ad hoc" in rationale.lower()


def test_instruction_diff():
    """Test instruction diff computation."""
    emitter = EventEmitter(
        experiment_name="test",
        benchmark_name="test",
        run_id="test",
        arm_name="test"
    )
    
    old = "Line 1\nLine 2\nLine 3"
    new = "Line 1\nLine 2 modified\nLine 3"
    
    diff = emitter.compute_instruction_diff(old, new)
    
    assert "@@" in diff  # Unified diff header
    assert "-Line 2" in diff
    assert "+Line 2 modified" in diff


def test_event_storage_roundtrip():
    """Test saving and loading events."""
    from events import EventStorage
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EventStorage(Path(tmpdir))
        
        # Create emitter and emit events
        emitter = EventEmitter(
            experiment_name="test",
            benchmark_name="test",
            run_id="test_run",
            arm_name="test_arm"
        )
        
        emitter.emit_initialization({"start": True})
        
        # Save
        events = emitter.get_events()
        filepath = storage.save_events(events, "test_run", "test_arm")
        
        assert filepath.exists()
        
        # Load
        loaded_events = storage.load_events(filepath)
        
        assert len(loaded_events) == len(events)
        assert loaded_events[0].event_type == "initialization"


def test_event_analyzer():
    """Test event analysis."""
    from events import EventAnalyzer
    
    # Create some test events
    emitter = EventEmitter(
        experiment_name="test",
        benchmark_name="test",
        run_id="test",
        arm_name="test"
    )
    
    # Emit a few steps with lineage
    for i in range(3):
        parent_id = f"candidate_{i:03d}" if i > 0 else None
        
        conjecture = Conjecture(
            candidate_id=f"candidate_{i+1:03d}",
            parent_id=parent_id,
            module_name="mod",
            old_instruction=f"old_{i}",
            new_instruction=f"new_{i}",
            generated_by="test",
        )
        
        test_selection = TestSelection(
            test_type="minibatch",
            example_ids=[0],
            example_count=1,
            selection_rationale="test",
            test_cost_units=1,
        )
        
        test_outcome = TestOutcome(
            passed=True,
            score=0.8,
            example_scores=[0.8],
        )
        
        decision = AcceptRejectDecision(
            accepted=True,
            decision_reason="test",
        )
        
        emitter.emit_step(
            conjecture=conjecture,
            test_selection=test_selection,
            test_outcome=test_outcome,
            decision=decision,
            calls_consumed=1,
            cost_consumed=0.001,
            lakatos_label=LakatosLabel.PROGRESSIVE,
        )
    
    events = emitter.get_events()
    analyzer = EventAnalyzer(events)
    
    # Test lineage tree
    tree = analyzer.build_lineage_tree()
    assert "candidate_001" in tree  # Root should have child
    
    # Test Lakatos counting
    counts = analyzer.count_by_lakatos_label()
    assert counts[LakatosLabel.PROGRESSIVE] == 3
    
    # Test progressive ratio
    ratio = analyzer.get_progressive_ratio()
    assert ratio == 1.0  # All progressive
    
    # Test timeline
    timeline = analyzer.extract_timeline()
    assert len(timeline) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
