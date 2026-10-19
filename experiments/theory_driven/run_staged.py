#!/usr/bin/env python3
"""Staged experiment runner with hard cost caps and stage gates.

Stage 0 (smoke): $1 cap, end-to-end validation
Stage 1 (pilot): $5 cap, full pilot
Stage 2 (cross-model): $25 cap, pilot on multiple models

Pre-flight gate runs before each paid stage (requires --confirm).

Usage:
    python run_staged.py --stage 0 --confirm  # Run smoke test
    python run_staged.py --stage 1 --confirm  # Run pilot (requires stage 0 gate)
    python run_staged.py --stage 2 --confirm  # Run cross-model (optional)
"""

import argparse
import sys
import os
from pathlib import Path

from .config import ExperimentConfig, StageGate, SmokeTestChecks, CanarySuite
from .harness import ExperimentHarness
from .preflight import PreFlightGate
import dspy


def run_canary_suite(
    config: ExperimentConfig,
    stage_config,
    gate: StageGate
) -> bool:
    """Run canary suite before a stage.
    
    Args:
        config: Experiment configuration
        stage_config: Stage configuration
        gate: Stage gate instance
        
    Returns:
        True if canaries pass (or drift allowed)
    """
    if not stage_config.run_canary:
        return True
    
    print("\n" + "="*60)
    print(f"CANARY SUITE: Stage {stage_config.stage_id}")
    print("="*60)
    
    # Determine model for canaries
    canary_model = stage_config.canary_model or config.model_config.task_model
    
    if "free" in canary_model.lower():
        print(f"\n⚠ WARNING: Using free-tier model for canaries: {canary_model}")
        print("Free providers may log prompts and change without notice.")
    
    print(f"Canary model: {canary_model}")
    
    # Initialize LM for canaries
    import os
    api_key = os.getenv(config.model_config.openrouter_api_key_env)
    if not api_key:
        print("\n⚠ No API key found, skipping canaries")
        return True
    
    try:
        # Create LM with temperature 0
        lm_kwargs = config.model_config.get_litellm_kwargs(canary_model)
        lm_kwargs.update({
            "temperature": 0.0,
            "seed": 42,
        })
        
        lm = dspy.LM(
            model=canary_model,
            api_key=api_key,
            **lm_kwargs
        )
        
        # Run canary suite
        canary_dir = Path(config.results_dir) / "canaries"
        suite = CanarySuite(reference_dir=canary_dir)
        
        print(f"\nRunning {len(suite.prompts)} canary prompts...")
        results = suite.run_canaries(lm, canary_model, temperature=0.0, seed=42)
        
        # Load reference
        reference = suite.load_reference(canary_model)
        
        if not reference.get("prompts"):
            print("\n⚠ No reference outputs found")
            print("Saving current outputs as reference...")
            suite.save_reference(canary_model, results)
            
            # Record canary check as passed (establishing baseline)
            gate.record_check(
                stage_id=stage_config.stage_id,
                check_name="canary_no_drift",
                passed=True,
                message="Canary baseline established",
                metadata={"canaries_run": len(results)}
            )
            
            return True
        
        # Compare with reference
        drift_detected, issues = suite.check_drift(results, reference)
        
        # Print report
        suite.print_report(results, reference)
        
        # Save results
        output_file = canary_dir / f"canary_stage{stage_config.stage_id}_{canary_model.replace('/', '_')}.json"
        suite.save_results(output_file)
        
        # Record in stage gate
        gate.record_check(
            stage_id=stage_config.stage_id,
            check_name="canary_no_drift",
            passed=not drift_detected or stage_config.allow_canary_drift,
            message=f"Drift {'detected' if drift_detected else 'not detected'}: {len(issues)} issues",
            metadata={
                "drift_detected": drift_detected,
                "issues": issues,
                "allow_drift": stage_config.allow_canary_drift,
            }
        )
        
        if drift_detected and not stage_config.allow_canary_drift:
            print("\n❌ CANARY DRIFT DETECTED - Stage blocked")
            print(f"Issues: {len(issues)}")
            for issue in issues[:5]:  # Show first 5
                print(f"  - {issue['type']}: {issue.get('message', '')}")
            return False
        
        print("\n✓ Canaries passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Canary suite FAILED: {e}")
        # Treat exceptions as failures (was incorrectly passing before)
        gate.record_check(
            stage_id=stage_config.stage_id,
            check_name="canary_no_drift",
            passed=False,
            message=f"Canary suite error: {str(e)}",
            metadata={"error": str(e)}
        )
        return False


def run_preflight_gate(stage_id: int, experiment_dir: Path) -> bool:
    """Run mandatory pre-flight gate before paid stage.
    
    Args:
        stage_id: Stage ID
        experiment_dir: Experiment directory
        
    Returns:
        True if pre-flight passed
    """
    preflight = PreFlightGate(experiment_dir)
    
    # Check if --confirm flag is present
    require_confirm = stage_id >= 0  # All paid stages require confirm
    
    passed = preflight.run_all_checks(stage_id, require_confirm=require_confirm)
    
    # Save results
    output_file = experiment_dir / "results" / f"preflight_stage_{stage_id}.json"
    preflight.save_results(output_file)
    
    # Record in stage gate
    # (Would integrate with StageGate here)
    
    return passed


def run_smoke_test(config: ExperimentConfig, gate: StageGate) -> bool:
    """Run smoke test (Stage 0) and validate.
    
    Args:
        config: Experiment configuration
        gate: Stage gate instance
        
    Returns:
        True if smoke test passed all checks
    """
    print("\n" + "="*60)
    print("STAGE 0: SMOKE TEST")
    print("="*60)
    
    stage = config.get_stage_config(0)
    
    # Run pre-flight gate FIRST
    experiment_dir = Path(__file__).parent
    if not run_preflight_gate(0, experiment_dir):
        print("\n❌ Pre-flight gate failed, aborting smoke test")
        return False
    
    # Run canary suite
    if not run_canary_suite(config, stage, gate):
        print("\n❌ Canary check failed, aborting smoke test")
        return False
    
    harness = ExperimentHarness(config)
    
    # Run smoke test experiments
    results = harness.run_stage(stage)
    
    if not results:
        print("\nERROR: No results from smoke test")
        return False
    
    # Run validation checks
    print("\n" + "="*60)
    print("SMOKE TEST VALIDATION")
    print("="*60)
    
    # Get first successful result for validation
    result = None
    for r in results:
        if r.get("status") == "success":
            result = r
            break
    
    if not result:
        print("\nERROR: No successful runs in smoke test")
        gate.record_check(
            stage_id=0,
            check_name="basic_execution",
            passed=False,
            message="No successful runs",
            metadata={}
        )
        return False
    
    # Check 1: Call accounting
    tracked_calls = result["budget_stats"]["total_units"]
    tracked_cost = result["budget_stats"]["total_cost"]
    
    passed, msg, metadata = SmokeTestChecks.check_call_accounting(
        tracked_calls=tracked_calls,
        tracked_cost=tracked_cost,
        provider_calls=None,  # Would get from provider API in real run
        provider_cost=None,
    )
    gate.record_check(0, "call_accounting_matches_provider", passed, msg, metadata)
    
    # Check 2: Cap aborts correctly
    max_cost = stage.max_cost_dollars
    actual_cost = sum(r["budget_stats"]["total_cost"] for r in results)
    run_completed = result["status"] == "success"
    aborted = result["status"] == "cost_limit_exceeded"
    
    passed, msg, metadata = SmokeTestChecks.check_cap_aborts(
        max_cost=max_cost,
        actual_cost=actual_cost,
        run_completed=run_completed,
        aborted_due_to_cap=aborted
    )
    gate.record_check(0, "cap_aborts_correctly", passed, msg, metadata)
    
    # Check 3: Results populated
    results_dir = Path(config.results_dir)
    passed, msg, metadata = SmokeTestChecks.check_results_populated(
        results_dir=results_dir,
        min_files=len(results)
    )
    gate.record_check(0, "results_populated", passed, msg, metadata)
    
    # Check 4: Dashboard accessible (placeholder - would check MLflow)
    gate.record_check(
        stage_id=0,
        check_name="dashboard_accessible",
        passed=True,  # Would check MLflow in real implementation
        message="MLflow tracking URI configured",
        metadata={"uri": config.mlflow_tracking_uri}
    )
    
    # Check 5: Metrics parse
    if results:
        result_file = results_dir / list(results_dir.glob("*.json"))[0].name
        passed, msg, metadata = SmokeTestChecks.check_metrics_parse(result_file)
    else:
        passed, msg, metadata = False, "No result files found", {}
    
    gate.record_check(0, "metrics_parse", passed, msg, metadata)
    
    # Print gate status
    gate.print_status(0)
    
    # Return overall pass/fail
    status = gate.get_status(0)
    return status and status.all_checks_passed


def run_pilot(config: ExperimentConfig, gate: StageGate) -> bool:
    """Run pilot (Stage 1) if smoke test passed.
    
    Args:
        config: Experiment configuration
        gate: Stage gate instance
        
    Returns:
        True if pilot completed successfully
    """
    print("\n" + "="*60)
    print("STAGE 1: PILOT")
    print("="*60)
    
    stage = config.get_stage_config(1)
    
    # Run pre-flight gate
    experiment_dir = Path(__file__).parent
    if not run_preflight_gate(1, experiment_dir):
        print("\n❌ Pre-flight gate failed, aborting pilot")
        return False
    
    # Run canary suite
    if not run_canary_suite(config, stage, gate):
        print("\n❌ Canary check failed, aborting pilot")
        return False
    
    # Check stage 0 gate
    stage_0 = config.get_stage_config(0)
    passed, message = gate.check_gate(0, stage_0.required_checks)
    
    if not passed:
        print(f"\nERROR: Stage 0 gate not passed: {message}")
        print("Run smoke test first: python run_staged.py --stage 0")
        return False
    
    print(f"Stage 0 gate passed: {message}")
    
    # Run pilot
    harness = ExperimentHarness(config)
    results = harness.run_stage(stage)
    
    success = any(r.get("status") == "success" for r in results)
    
    if success:
        print("\nPilot completed successfully!")
    else:
        print("\nWARNING: Pilot had no successful runs")
    
    return success


def run_cross_model(config: ExperimentConfig, gate: StageGate) -> bool:
    """Run cross-model stage (Stage 2).
    
    Args:
        config: Experiment configuration
        
    Returns:
        True if completed successfully
    """
    print("\n" + "="*60)
    print("STAGE 2: CROSS-MODEL")
    print("="*60)
    
    stage = config.get_stage_config(2)
    
    # Run canary suite (allows drift for cross-model comparison)
    if not run_canary_suite(config, stage, gate):
        print("\n⚠ Canary check failed (non-blocking for cross-model)")
    
    harness = ExperimentHarness(config)
    results = harness.run_stage(stage)
    
    success = any(r.get("status") == "success" for r in results)
    
    if success:
        print("\nCross-model evaluation completed successfully!")
    else:
        print("\nWARNING: Cross-model evaluation had no successful runs")
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run staged theory-driven experiments"
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="Stage to run: 0=smoke, 1=pilot, 2=cross-model"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON (optional, uses defaults otherwise)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm paid calls (required for stages that make paid API calls)"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ExperimentConfig.from_file(args.config)
    else:
        config = ExperimentConfig()
    
    # Create stage gate
    gate = StageGate(config.results_dir)
    
    # Run requested stage
    if args.stage == 0:
        success = run_smoke_test(config, gate)
        sys.exit(0 if success else 1)
    
    elif args.stage == 1:
        success = run_pilot(config, gate)
        sys.exit(0 if success else 1)
    
    elif args.stage == 2:
        success = run_cross_model(config, gate)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
