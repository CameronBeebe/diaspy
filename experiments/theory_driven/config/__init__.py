"""Configuration management for theory-driven experiments."""

from .experiment_config import (
    ExperimentConfig,
    ArmConfig,
    BenchmarkConfig,
    ModelConfig,
    StageConfig,
)
from .budget_tracker import BudgetTracker, BudgetEnforcer, CostLimitExceeded
from .stage_gate import StageGate, StageGateStatus, CheckResult, SmokeTestChecks
from .provenance import ProvenanceLogger, LMCallTrace, fetch_openrouter_generation_metadata, reconcile_costs
from .canary import CanarySuite, CanaryPrompt, CanaryResult

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
