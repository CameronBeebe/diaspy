"""Statistical analysis: paired bootstrap and refutation criteria.

Implements R1-R5 from the pre-registered design with 95% paired bootstrap CIs.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import dataclasses
from .store import RunResult


@dataclasses.dataclass
class BootstrapResult:
    """Bootstrap comparison result."""
    arm_x: str
    arm_y: str
    n_labels: Optional[int]  # None for pooled
    
    # Paired differences
    mean_diff: float  # mean(X - Y) in percentage points
    ci_lower: float
    ci_upper: float
    
    # Individual arm means
    mean_x: float
    mean_y: float
    
    # Sample details
    n_seeds: int
    n_test_items: int


def paired_bootstrap_ci(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Paired bootstrap for X - Y with CI.
    
    Args:
        x_scores: shape (n_seeds, n_items), each entry is 0 or 1
        y_scores: shape (n_seeds, n_items)
        n_bootstrap: number of bootstrap samples
        ci_level: confidence level
        seed: random seed
    
    Returns:
        (mean_diff, ci_lower, ci_upper) in percentage points
    """
    assert x_scores.shape == y_scores.shape
    rng = np.random.RandomState(seed)
    
    n_seeds, n_items = x_scores.shape
    
    # Observed difference
    mean_diff = (x_scores.mean() - y_scores.mean()) * 100
    
    # Bootstrap over seeds and items
    boot_diffs = []
    for _ in range(n_bootstrap):
        # Resample seeds with replacement
        seed_idx = rng.choice(n_seeds, size=n_seeds, replace=True)
        boot_x = x_scores[seed_idx, :]
        boot_y = y_scores[seed_idx, :]
        
        # Resample items with replacement
        item_idx = rng.choice(n_items, size=n_items, replace=True)
        boot_x = boot_x[:, item_idx]
        boot_y = boot_y[:, item_idx]
        
        boot_diff = (boot_x.mean() - boot_y.mean()) * 100
        boot_diffs.append(boot_diff)
    
    boot_diffs = np.array(boot_diffs)
    
    # Percentile CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    
    return mean_diff, ci_lower, ci_upper


def extract_paired_scores(
    results: List[RunResult],
    arm_x: str,
    arm_y: str,
    n_labels: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract paired per-item scores for two arms.
    
    Args:
        results: list of run results
        arm_x, arm_y: arms to compare
        n_labels: if specified, filter to this N; if None, pool all N
    
    Returns:
        (x_scores, y_scores) arrays of shape (n_seeds, n_items)
    """
    # Filter valid runs
    valid_x = [r for r in results 
               if r.arm == arm_x 
               and r.within_budget_tolerance
               and r.test_predictions is not None
               and (n_labels is None or r.n_labels == n_labels)]
    
    valid_y = [r for r in results
               if r.arm == arm_y
               and r.within_budget_tolerance
               and r.test_predictions is not None
               and (n_labels is None or r.n_labels == n_labels)]
    
    if not valid_x or not valid_y:
        raise ValueError(f"No valid runs for {arm_x} vs {arm_y} at N={n_labels}")
    
    # Group by seed
    x_by_seed = {}
    for r in valid_x:
        key = (r.seed, r.n_labels) if n_labels is None else r.seed
        x_by_seed[key] = [p["score"] for p in r.test_predictions]
    
    y_by_seed = {}
    for r in valid_y:
        key = (r.seed, r.n_labels) if n_labels is None else r.seed
        y_by_seed[key] = [p["score"] for p in r.test_predictions]
    
    # Ensure pairing
    common_keys = set(x_by_seed.keys()) & set(y_by_seed.keys())
    if not common_keys:
        raise ValueError(f"No paired seeds for {arm_x} vs {arm_y}")
    
    common_keys = sorted(common_keys)
    x_scores = np.array([x_by_seed[k] for k in common_keys])
    y_scores = np.array([y_by_seed[k] for k in common_keys])
    
    return x_scores, y_scores


def compare_arms(
    results: List[RunResult],
    arm_x: str,
    arm_y: str,
    n_labels: Optional[int] = None,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> BootstrapResult:
    """Compare two arms with paired bootstrap.
    
    Returns X - Y difference with CI.
    """
    x_scores, y_scores = extract_paired_scores(results, arm_x, arm_y, n_labels)
    
    mean_diff, ci_lower, ci_upper = paired_bootstrap_ci(
        x_scores, y_scores, n_bootstrap=n_bootstrap, seed=seed
    )
    
    return BootstrapResult(
        arm_x=arm_x,
        arm_y=arm_y,
        n_labels=n_labels,
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_x=x_scores.mean() * 100,
        mean_y=y_scores.mean() * 100,
        n_seeds=x_scores.shape[0],
        n_test_items=x_scores.shape[1]
    )


def check_refutation_criteria(results: List[RunResult]) -> Dict[str, dict]:
    """Check pre-registered refutation criteria R1-R5.
    
    Returns dict with results and interpretations for each criterion.
    """
    # R1: Pooled small-N B vs A
    # Refutes hypothesis if Δ_BA < +2.0 OR lower CI ≤ 0
    try:
        r1_pooled = compare_arms(results, "B", "A", n_labels=None)
        # Filter to small N only
        small_n_results = [r for r in results if r.n_labels in [10, 25]]
        r1 = compare_arms(small_n_results, "B", "A", n_labels=None)
        
        r1_refutes = r1.mean_diff < 2.0 or r1.ci_lower <= 0
        r1_result = {
            "criterion": "R1: Hypothesis (B > A at small N)",
            "mean_diff": r1.mean_diff,
            "ci_lower": r1.ci_lower,
            "ci_upper": r1.ci_upper,
            "threshold": 2.0,
            "refuted": r1_refutes,
            "interpretation": (
                "Hypothesis REFUTED: B does not sufficiently beat A at small N"
                if r1_refutes else
                "Hypothesis NOT refuted: B beats A by ≥2 points with positive lower CI"
            )
        }
    except ValueError as e:
        r1_result = {"error": str(e)}
    
    # R2: Low-data specificity
    # Refutes "low-data specific" if Δ_BA(N=100) ≥ Δ_BA(N=10)
    try:
        r2_n10 = compare_arms(results, "B", "A", n_labels=10)
        r2_n100 = compare_arms(results, "B", "A", n_labels=100)
        
        r2_refutes = r2_n100.mean_diff >= r2_n10.mean_diff
        r2_result = {
            "criterion": "R2: Low-data specificity",
            "diff_n10": r2_n10.mean_diff,
            "diff_n100": r2_n100.mean_diff,
            "refuted": r2_refutes,
            "interpretation": (
                "Low-data specificity REFUTED: gain persists or grows at N=100"
                if r2_refutes else
                "Low-data specificity NOT refuted: gain is larger at small N"
            )
        }
    except ValueError as e:
        r2_result = {"error": str(e)}
    
    # R3: Typing matters
    # Refutes "typing irrelevant" if pooled small-N Δ_BC ≤ 0
    try:
        small_n_results = [r for r in results if r.n_labels in [10, 25]]
        r3 = compare_arms(small_n_results, "B", "C", n_labels=None)
        
        r3_refutes = r3.mean_diff <= 0
        r3_result = {
            "criterion": "R3: Typing irrelevant",
            "mean_diff": r3.mean_diff,
            "ci_lower": r3.ci_lower,
            "ci_upper": r3.ci_upper,
            "refuted": r3_refutes,
            "interpretation": (
                "Typing IRRELEVANT (refuted): typed B ≤ untyped C"
                if r3_refutes else
                "Typing NOT irrelevant: typed B beats untyped C"
            )
        }
    except ValueError as e:
        r3_result = {"error": str(e)}
    
    # R4: Filter vs epistemology
    # Refutes "epistemology matters" if CI of Δ_BE includes 0
    try:
        small_n_results = [r for r in results if r.n_labels in [10, 25]]
        r4 = compare_arms(small_n_results, "B", "E", n_labels=None)
        
        r4_refutes = r4.ci_lower <= 0 <= r4.ci_upper
        r4_result = {
            "criterion": "R4: Filter vs epistemology",
            "mean_diff": r4.mean_diff,
            "ci_lower": r4.ci_lower,
            "ci_upper": r4.ci_upper,
            "refuted": r4_refutes,
            "interpretation": (
                "Epistemology irrelevant (refuted): B ≈ E (filter alone suffices)"
                if r4_refutes else
                "Epistemology matters: B significantly differs from filter-only E"
            )
        }
    except ValueError as e:
        r4_result = {"error": str(e)}
    
    # R5: Incumbent sufficiency
    # Refutes "B needed" if pooled small-N Δ_BF ≤ 0
    try:
        small_n_results = [r for r in results if r.n_labels in [10, 25]]
        r5 = compare_arms(small_n_results, "B", "F", n_labels=None)
        
        r5_refutes = r5.mean_diff <= 0
        r5_result = {
            "criterion": "R5: Incumbent (SIMBA) suffices",
            "mean_diff": r5.mean_diff,
            "ci_lower": r5.ci_lower,
            "ci_upper": r5.ci_upper,
            "refuted": r5_refutes,
            "interpretation": (
                "SIMBA sufficient (refuted): B ≤ SIMBA"
                if r5_refutes else
                "B needed: B beats SIMBA"
            )
        }
    except ValueError as e:
        r5_result = {"error": str(e)}
    
    return {
        "R1": r1_result,
        "R2": r2_result,
        "R3": r3_result,
        "R4": r4_result,
        "R5": r5_result
    }


def summarize_results(results: List[RunResult]) -> dict:
    """Generate summary statistics across all runs."""
    valid_runs = [r for r in results if r.within_budget_tolerance and r.test_accuracy is not None]
    
    if not valid_runs:
        return {"error": "No valid runs"}
    
    # Group by arm and N
    by_arm_n = {}
    for r in valid_runs:
        key = (r.arm, r.n_labels)
        if key not in by_arm_n:
            by_arm_n[key] = []
        by_arm_n[key].append(r.test_accuracy * 100)
    
    summary = {}
    for (arm, n), accs in by_arm_n.items():
        summary[f"{arm}_N{n}"] = {
            "arm": arm,
            "n_labels": n,
            "mean_accuracy": np.mean(accs),
            "std_accuracy": np.std(accs),
            "n_seeds": len(accs)
        }
    
    return summary
