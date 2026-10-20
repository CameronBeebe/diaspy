"""Versioned resume fingerprints contain scientific settings, never secrets."""
from dataclasses import asdict
from importlib import metadata
import hashlib
import json
import math
from pathlib import Path
from typing import Any, TypeGuard

from experiments.theory_driven.config import ExperimentConfig, StageConfig
from experiments.theory_driven.config.variants import parse_arm_variant


import subprocess


def _get_git_version() -> str:
    """Return current git SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5
        )
        return result.stdout.strip()[:12]  # Short SHA
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


_CODE_VERSION = _get_git_version()


def fingerprint(config: ExperimentConfig, stage: StageConfig, benchmark: str,
                variant: str, budget: int, seed: int) -> str:
    base, policy = parse_arm_variant(variant)
    arm = next(a for a in config.arms + config.full_grid_arms if a.name == base)
    def safe(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()
                    if k.lower() not in {"api_key", "access_token", "api_token", "secret", "password", "authorization"}}
        if hasattr(value, "model"):
            return {"lm_model": value.model, "kwargs": safe(value.kwargs)}
        raise ValueError("Resume requires serializable arm configuration")
    benchmark_config = next((b for b in config.benchmarks if b.name == benchmark), None)
    payload = {"version": 2, "code_version": _CODE_VERSION,
               "dspy": metadata.version("dspy"), "gepa": metadata.version("gepa"),
               "model": safe(asdict(config.model_config)), "benchmark": benchmark,
               "benchmark_options": safe(asdict(benchmark_config)) if benchmark_config else {},
               "train_size": config.train_size, "val_size": config.val_size, "test_size": stage.test_size,
               "arm": variant, "arm_type": arm.arm_type, "arm_options": safe(arm.kwargs),
               "policy": policy or stage.survival_policy, "minimum": stage.min_survival_questions,
               "budget": budget, "seed": seed, "stage_cap": stage.max_cost_dollars,
               "canary": {"model": stage.canary_model, "enabled": stage.run_canary,
                          "allow_drift": stage.allow_canary_drift}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def completed_result(directory: Path, benchmark: str, variant: str, budget: int,
                     seed: int, config_hash: str, test_size: int) -> dict | None:
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        if (row.get("benchmark"), row.get("arm"), row.get("budget"), row.get("seed"), row.get("config_hash")) != (
                benchmark, variant, budget, seed, config_hash):
            continue
        scores = row.get("test_scores")
        stats = row.get("budget_stats")
        cost = stats.get("total_cost") if isinstance(stats, dict) else None
        if (row.get("status") == "success" and _finite_number(row.get("test_mean"))
                and row.get("test_size") == test_size
                and isinstance(scores, list) and len(scores) == test_size
                and all(_finite_number(score) for score in scores)
                and _finite_number(cost) and cost >= 0):
            return {**row, "resumed_from": str(path.resolve())}
    return None


def any_result_cost(directory: Path, benchmark: str, variant: str, budget: int,
                    seed: int, config_hash: str) -> float:
    """Sum cost across all matching result files (including aborted runs), not just the latest."""
    total_cost = 0.0
    for path in directory.glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        if (row.get("benchmark"), row.get("arm"), row.get("budget"), row.get("seed"), row.get("config_hash")) != (
                benchmark, variant, budget, seed, config_hash):
            continue
        stats = row.get("budget_stats")
        cost = stats.get("total_cost") if isinstance(stats, dict) else None
        if _finite_number(cost) and cost >= 0:
            total_cost += cost
    return total_cost


def _finite_number(value: Any) -> TypeGuard[int | float]:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
