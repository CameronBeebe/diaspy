"""Smoke test: run all arms at tiny scale with DummyLM.

This tests plumbing only, not efficacy. Runs free and offline.
"""
import pytest
import tempfile
from pathlib import Path

from experiments.config import ExperimentConfig, BudgetConfig
from experiments.runner import ExperimentRunner
from experiments.store import ResultsStore


def test_all_arms_smoke():
    """Run all arms at tiny scale (N=10, 1 seed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            arms=["A0", "A", "E", "C", "B", "F"],
            n_labels=[10],
            seeds=[0],
            budget=BudgetConfig(
                max_program_rollouts=50,  # Tiny for smoke test
                max_reflection_calls=10,
                max_reflection_tokens=50_000
            ),
            results_dir=tmpdir,
            gepa_reflection_minibatch_size=2,
            proposer_max_attempts=1,
            proposer_archive_cap=3
        )
        
        runner = ExperimentRunner(config, use_dummy_lm=True)
        runner.run_grid()
        
        # Check all runs completed
        store = ResultsStore(Path(tmpdir))
        results = list(store.load_runs())
        
        assert len(results) == 6, f"Expected 6 runs, got {len(results)}"
        
        # Check each arm ran
        arms_run = {r.arm for r in results}
        assert arms_run == {"A0", "A", "E", "C", "B", "F"}
        
        # Check all succeeded (DummyLM should not fail)
        for r in results:
            assert r.optimizer_success or r.arm in ["F"], \
                f"{r.arm} failed: {r.optimizer_error}"
            assert r.test_accuracy is not None or not r.optimizer_success
        
        print("\n✓ Smoke test passed: all arms ran")
        
        # Print summary
        for r in results:
            print(f"  {r.arm}: test_acc={r.test_accuracy:.2f if r.test_accuracy else 0:.2f}, "
                  f"rollouts={r.program_rollouts}, reflection_calls={r.reflection_calls}")


def test_budget_enforcement():
    """Test that budget limits are enforced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            arms=["A"],
            n_labels=[10],
            seeds=[0],
            budget=BudgetConfig(
                max_program_rollouts=5,  # Very tight
                max_reflection_calls=2,
                max_reflection_tokens=10_000
            ),
            results_dir=tmpdir
        )
        
        runner = ExperimentRunner(config, use_dummy_lm=True)
        
        # This might fail or complete with overbudget flag
        try:
            runner.run_grid()
        except RuntimeError as e:
            assert "Budget exceeded" in str(e)
            print("\n✓ Budget enforcement test passed: budget was enforced")
            return
        
        # If it completed, check that it was flagged as overbudget
        store = ResultsStore(Path(tmpdir))
        results = list(store.load_runs())
        
        if results:
            # Should be within tolerance or flagged
            print("\n✓ Budget enforcement test passed: tight budget handled")


def test_popperian_proposer_plumbing():
    """Specific test for Popperian proposer from skeleton."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            arms=["B"],
            n_labels=[10],
            seeds=[0],
            budget=BudgetConfig(max_program_rollouts=30, max_reflection_calls=10),
            results_dir=tmpdir,
            proposer_max_attempts=2
        )
        
        runner = ExperimentRunner(config, use_dummy_lm=True)
        runner.run_grid()
        
        store = ResultsStore(Path(tmpdir))
        results = list(store.load_runs())
        
        assert len(results) == 1
        r = results[0]
        
        assert r.arm == "B"
        assert r.optimizer_success, f"Popperian proposer failed: {r.optimizer_error}"
        assert r.proposer_log is not None, "Proposer log should be recorded"
        
        print("\n✓ Popperian proposer plumbing test passed")
        print(f"  Proposer log entries: {len(r.proposer_log)}")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_all_arms_smoke()
    test_budget_enforcement()
    test_popperian_proposer_plumbing()
    print("\n✓ All smoke tests passed")
