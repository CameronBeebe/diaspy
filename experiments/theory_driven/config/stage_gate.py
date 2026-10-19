"""Stage gate mechanism for experiment execution.

Ensures smoke test passes required checks before pilot can run.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CheckResult:
    """Result of a single stage gate check."""
    check_name: str
    passed: bool
    message: str
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class StageGateStatus:
    """Status of a stage gate."""
    stage_id: int
    stage_name: str
    all_checks_passed: bool
    checks: List[CheckResult]
    gate_passed_at: Optional[str] = None


class StageGate:
    """Manages stage gates and check requirements."""
    
    def __init__(self, results_dir: str):
        """Initialize stage gate.
        
        Args:
            results_dir: Directory for storing gate status files
        """
        self.results_dir = Path(results_dir)
        self.gates_dir = self.results_dir / "stage_gates"
        self.gates_dir.mkdir(parents=True, exist_ok=True)
    
    def get_gate_file(self, stage_id: int) -> Path:
        """Get path to gate status file for a stage."""
        return self.gates_dir / f"stage_{stage_id}_gate.json"
    
    def record_check(
        self,
        stage_id: int,
        check_name: str,
        passed: bool,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record the result of a check.
        
        Args:
            stage_id: Stage ID
            check_name: Name of the check
            passed: Whether check passed
            message: Human-readable message
            metadata: Additional metadata
        """
        result = CheckResult(
            check_name=check_name,
            passed=passed,
            message=message,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )
        
        # Load existing status or create new
        gate_file = self.get_gate_file(stage_id)
        if gate_file.exists():
            with open(gate_file, 'r') as f:
                data = json.load(f)
            status = StageGateStatus(**data)
            
            # Update or add check
            existing = [c for c in status.checks if c["check_name"] != check_name]
            status.checks = existing + [asdict(result)]
        else:
            status = StageGateStatus(
                stage_id=stage_id,
                stage_name=f"stage_{stage_id}",
                all_checks_passed=False,
                checks=[asdict(result)],
            )
        
        # Update all_checks_passed status
        status.all_checks_passed = all(c["passed"] for c in status.checks)
        if status.all_checks_passed:
            status.gate_passed_at = datetime.utcnow().isoformat()
        
        # Save
        with open(gate_file, 'w') as f:
            json.dump(asdict(status), f, indent=2)
    
    def check_gate(self, stage_id: int, required_checks: List[str]) -> tuple[bool, str]:
        """Check if a stage gate has passed.
        
        Args:
            stage_id: Stage ID to check
            required_checks: List of required check names
            
        Returns:
            Tuple of (passed, message)
        """
        gate_file = self.get_gate_file(stage_id)
        
        if not gate_file.exists():
            return False, f"Stage {stage_id} gate file not found. Run smoke test first."
        
        with open(gate_file, 'r') as f:
            data = json.load(f)
        status = StageGateStatus(**data)
        
        # Check all required checks are present and passed
        check_names = {c["check_name"] for c in status.checks}
        missing = set(required_checks) - check_names
        if missing:
            return False, f"Missing required checks: {', '.join(sorted(missing))}"
        
        failed = [c["check_name"] for c in status.checks 
                  if c["check_name"] in required_checks and not c["passed"]]
        if failed:
            return False, f"Failed checks: {', '.join(sorted(failed))}"
        
        return True, "All checks passed"
    
    def get_status(self, stage_id: int) -> Optional[StageGateStatus]:
        """Get current status of a stage gate."""
        gate_file = self.get_gate_file(stage_id)
        if not gate_file.exists():
            return None
        
        with open(gate_file, 'r') as f:
            data = json.load(f)
        return StageGateStatus(**data)
    
    def print_status(self, stage_id: int) -> None:
        """Print stage gate status in human-readable format."""
        status = self.get_status(stage_id)
        if not status:
            print(f"Stage {stage_id}: No gate status file found")
            return
        
        print(f"\n{'='*60}")
        print(f"Stage {stage_id} ({status.stage_name}) Gate Status")
        print(f"{'='*60}")
        print(f"Overall: {'✓ PASSED' if status.all_checks_passed else '✗ FAILED'}")
        if status.gate_passed_at:
            print(f"Passed at: {status.gate_passed_at}")
        print(f"\nChecks:")
        
        for check in status.checks:
            symbol = "✓" if check["passed"] else "✗"
            print(f"  {symbol} {check['check_name']}")
            print(f"    {check['message']}")
            if check.get("metadata"):
                for key, value in check["metadata"].items():
                    print(f"    {key}: {value}")
        print(f"{'='*60}\n")


# Smoke test check implementations
class SmokeTestChecks:
    """Implementations of smoke test validation checks."""
    
    @staticmethod
    def check_call_accounting(
        tracked_calls: int,
        tracked_cost: float,
        provider_calls: Optional[int] = None,
        provider_cost: Optional[float] = None,
        tolerance: float = 0.1
    ) -> tuple[bool, str, Dict[str, Any]]:
        """Check if call accounting matches provider reports.
        
        Args:
            tracked_calls: Calls tracked by budget tracker
            tracked_cost: Cost tracked by budget tracker
            provider_calls: Calls reported by provider (if available)
            provider_cost: Cost reported by provider (if available)
            tolerance: Acceptable relative difference (10% default)
            
        Returns:
            Tuple of (passed, message, metadata)
        """
        metadata = {
            "tracked_calls": tracked_calls,
            "tracked_cost": tracked_cost,
            "provider_calls": provider_calls,
            "provider_cost": provider_cost,
        }
        
        # If provider data not available, pass with note
        if provider_calls is None and provider_cost is None:
            return True, "Provider data not available; tracking internally only", metadata
        
        # Check calls if available
        if provider_calls is not None:
            call_diff = abs(tracked_calls - provider_calls) / max(tracked_calls, 1)
            if call_diff > tolerance:
                return False, f"Call count mismatch: {tracked_calls} vs {provider_calls} ({call_diff:.1%})", metadata
        
        # Check cost if available
        if provider_cost is not None:
            cost_diff = abs(tracked_cost - provider_cost) / max(tracked_cost, 0.01)
            if cost_diff > tolerance:
                return False, f"Cost mismatch: ${tracked_cost:.4f} vs ${provider_cost:.4f} ({cost_diff:.1%})", metadata
        
        return True, f"Accounting matches: {tracked_calls} calls, ${tracked_cost:.4f}", metadata
    
    @staticmethod
    def check_cap_aborts(
        max_cost: float,
        actual_cost: float,
        run_completed: bool,
        aborted_due_to_cap: bool
    ) -> tuple[bool, str, Dict[str, Any]]:
        """Check if cost cap correctly aborted run or allowed completion.
        
        Args:
            max_cost: Maximum allowed cost
            actual_cost: Actual cost incurred
            run_completed: Whether run completed normally
            aborted_due_to_cap: Whether run aborted due to cap
            
        Returns:
            Tuple of (passed, message, metadata)
        """
        metadata = {
            "max_cost": max_cost,
            "actual_cost": actual_cost,
            "run_completed": run_completed,
            "aborted_due_to_cap": aborted_due_to_cap,
        }
        
        # If cost exceeded and run aborted, that's correct
        if actual_cost >= max_cost and aborted_due_to_cap:
            return True, f"Cap correctly aborted at ${actual_cost:.4f} >= ${max_cost:.4f}", metadata
        
        # If cost under cap and run completed, that's correct
        if actual_cost < max_cost and run_completed:
            return True, f"Run completed under cap: ${actual_cost:.4f} < ${max_cost:.4f}", metadata
        
        # Otherwise something is wrong
        if actual_cost >= max_cost and not aborted_due_to_cap:
            return False, f"Cap not enforced: ${actual_cost:.4f} >= ${max_cost:.4f} but run did not abort", metadata
        
        return False, f"Unexpected state: cost ${actual_cost:.4f}, cap ${max_cost:.4f}, completed={run_completed}, aborted={aborted_due_to_cap}", metadata
    
    @staticmethod
    def check_results_populated(results_dir: Path, min_files: int = 1) -> tuple[bool, str, Dict[str, Any]]:
        """Check if results directory is populated.
        
        Args:
            results_dir: Results directory path
            min_files: Minimum number of result files expected
            
        Returns:
            Tuple of (passed, message, metadata)
        """
        if not results_dir.exists():
            return False, f"Results directory does not exist: {results_dir}", {}
        
        result_files = list(results_dir.glob("**/*.json"))
        metadata = {
            "results_dir": str(results_dir),
            "num_files": len(result_files),
            "files": [str(f.relative_to(results_dir)) for f in result_files[:10]],  # First 10
        }
        
        if len(result_files) < min_files:
            return False, f"Insufficient result files: {len(result_files)} < {min_files}", metadata
        
        return True, f"Results populated: {len(result_files)} files", metadata
    
    @staticmethod
    def check_metrics_parse(results_file: Path) -> tuple[bool, str, Dict[str, Any]]:
        """Check if metrics can be parsed from results.
        
        Args:
            results_file: Path to a results JSON file
            
        Returns:
            Tuple of (passed, message, metadata)
        """
        if not results_file.exists():
            return False, f"Results file not found: {results_file}", {}
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Check for expected keys
            required = ["benchmark", "arm", "budget", "seed"]
            missing = [k for k in required if k not in data]
            if missing:
                return False, f"Missing required keys: {', '.join(missing)}", {}
            
            # Check for metrics
            if "metrics" not in data:
                return False, "No metrics field in results", {}
            
            metadata = {
                "benchmark": data.get("benchmark"),
                "arm": data.get("arm"),
                "metrics_keys": list(data.get("metrics", {}).keys()),
            }
            
            return True, "Metrics parsed successfully", metadata
            
        except json.JSONDecodeError as e:
            return False, f"JSON parse error: {e}", {}
        except Exception as e:
            return False, f"Error parsing metrics: {e}", {}
