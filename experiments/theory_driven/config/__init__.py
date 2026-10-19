"""Configuration management for theory-driven experiments."""

from experiments.theory_driven.config.experiment_config import (
    ExperimentConfig,
    ArmConfig,
    BenchmarkConfig,
    ModelConfig,
    StageConfig,
)
from experiments.theory_driven.config.budget_tracker import BudgetTracker, BudgetEnforcer, CostLimitExceeded
from experiments.theory_driven.config.stage_gate import StageGate, StageGateStatus, CheckResult, SmokeTestChecks
from experiments.theory_driven.config.provenance import ProvenanceLogger, LMCallTrace, fetch_openrouter_generation_metadata, reconcile_costs
from experiments.theory_driven.config.canary import CanarySuite, CanaryPrompt, CanaryResult

__all__ = [
    "ExperimentConfig",
    "ArmConfig",
    "BenchmarkConfig",
    "ModelConfig",
    "StageConfig",
    "BudgetTracker",
    "BudgetEnforcer",
    "CostLimitExceeded",
    "StageGate",
    "StageGateStatus",
    "CheckResult",
    "SmokeTestChecks",
    "ProvenanceLogger",
    "LMCallTrace",
    "fetch_openrouter_generation_metadata",
    "reconcile_costs",
    "CanarySuite",
    "CanaryPrompt",
    "CanaryResult",
]
