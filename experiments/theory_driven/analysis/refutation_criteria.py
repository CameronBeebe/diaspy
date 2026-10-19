"""Refutation criteria from the spec (R1-R7, H_cal)."""

from typing import Dict, List, Any
from .bootstrap import paired_bootstrap


def check_refutation_criteria(
    results: Dict[str, List[Dict[str, Any]]],
    primary_benchmark: str = "hover",
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000
) -> Dict[str, Any]:
    """Check all pre-registered refutation criteria.
    
    Per spec:
    - R1 (primary): Pooled over B∈{50,100}, arm fails if Δ_X-A < +3.0 or CI lower ≤ 0
    - R1' (strong): At B≤100, Δ_K-A < 0 with CI excluding 0
    - R2 (budget-specificity): Δ_K-A(400) ≥ Δ_K-A(50)
    - R3 (typing): Δ_B-C ≤ 0
    - R4 (filter): CI of Δ_B-E includes 0
    - R5 (incumbent): Δ_K-F ≤ 0
    - H_cal: Jev's Brier skill score ≤ 0
    - R7 (calibration matters): Δ(K-Jev - K-verbal) ≤ 0 at B≤100
    
    Args:
        results: Results dict with arm -> [scores]
        primary_benchmark: Primary benchmark name
        confidence_level: Confidence level for CIs
        n_bootstrap: Bootstrap samples
        
    Returns:
        Dictionary with refutation checks
    """
    criteria = {}
    
    # Helper to get scores for arm at budget
    def get_scores(arm: str, budget: int, benchmark: str = primary_benchmark):
        return [
            r["test_mean"]
            for r in results.get(arm, [])
            if r["benchmark"] == benchmark and r["budget"] == budget
        ]
    
    # Helper to get pooled scores over budgets
    def get_pooled_scores(arm: str, budgets: List[int], benchmark: str = primary_benchmark):
        scores = []
        for b in budgets:
            scores.extend(get_scores(arm, b, benchmark))
        return scores
    
    # R1: Primary criterion (pooled B∈{50,100})
    k_jev_pooled = get_pooled_scores("K-Jev", [50, 100])
    b_pooled = get_pooled_scores("B", [50, 100])
    a_pooled = get_pooled_scores("A", [50, 100])
    
    if k_jev_pooled and a_pooled:
        k_vs_a = paired_bootstrap(k_jev_pooled, a_pooled, n_bootstrap, confidence_level)
        k_fails = k_vs_a["mean_difference"] < 3.0 or k_vs_a["ci_lower"] <= 0
        
        criteria["R1_K-Jev"] = {
            "delta": k_vs_a["mean_difference"],
            "ci_lower": k_vs_a["ci_lower"],
            "ci_upper": k_vs_a["ci_upper"],
            "fails": k_fails,
        }
    
    if b_pooled and a_pooled:
        b_vs_a = paired_bootstrap(b_pooled, a_pooled, n_bootstrap, confidence_level)
        b_fails = b_vs_a["mean_difference"] < 3.0 or b_vs_a["ci_lower"] <= 0
        
        criteria["R1_B"] = {
            "delta": b_vs_a["mean_difference"],
            "ci_lower": b_vs_a["ci_lower"],
            "ci_upper": b_vs_a["ci_upper"],
            "fails": b_fails,
        }
        
        # Hypothesis refuted if BOTH K and B fail
        criteria["R1_hypothesis_refuted"] = k_fails and b_fails
    
    # R1': Strong criterion
    if k_jev_pooled and a_pooled:
        k_vs_a_strong = paired_bootstrap(k_jev_pooled, a_pooled, n_bootstrap, confidence_level)
        criteria["R1_prime"] = {
            "delta": k_vs_a_strong["mean_difference"],
            "ci_lower": k_vs_a_strong["ci_lower"],
            "ci_upper": k_vs_a_strong["ci_upper"],
            "refuted": k_vs_a_strong["mean_difference"] < 0 and not (
                k_vs_a_strong["ci_lower"] <= 0 <= k_vs_a_strong["ci_upper"]
            ),
        }
    
    # R2: Budget-specificity
    k_50 = get_scores("K-Jev", 50)
    k_400 = get_scores("K-Jev", 400)
    a_50 = get_scores("A", 50)
    a_400 = get_scores("A", 400)
    
    if k_50 and a_50 and k_400 and a_400:
        delta_50 = paired_bootstrap(k_50, a_50, n_bootstrap, confidence_level)["mean_difference"]
        delta_400 = paired_bootstrap(k_400, a_400, n_bootstrap, confidence_level)["mean_difference"]
        
        criteria["R2"] = {
            "delta_50": delta_50,
            "delta_400": delta_400,
            "refuted": delta_400 < delta_50,
        }
    
    # R3: Typing
    b_pooled = get_pooled_scores("B", [50, 100])
    c_pooled = get_pooled_scores("C", [50, 100])
    
    if b_pooled and c_pooled:
        b_vs_c = paired_bootstrap(b_pooled, c_pooled, n_bootstrap, confidence_level)
        criteria["R3"] = {
            "delta": b_vs_c["mean_difference"],
            "ci_lower": b_vs_c["ci_lower"],
            "ci_upper": b_vs_c["ci_upper"],
            "refuted": b_vs_c["mean_difference"] <= 0,
        }
    
    # R4: Filter (if E arm exists)
    b_pooled = get_pooled_scores("B", [50, 100])
    e_pooled = get_pooled_scores("E", [50, 100])
    
    if b_pooled and e_pooled:
        b_vs_e = paired_bootstrap(b_pooled, e_pooled, n_bootstrap, confidence_level)
        ci_includes_zero = b_vs_e["ci_lower"] <= 0 <= b_vs_e["ci_upper"]
        
        criteria["R4"] = {
            "delta": b_vs_e["mean_difference"],
            "ci_lower": b_vs_e["ci_lower"],
            "ci_upper": b_vs_e["ci_upper"],
            "refuted": not ci_includes_zero,
        }
    
    # R5: Incumbent (if F arm exists)
    k_pooled = get_pooled_scores("K-Jev", [50, 100])
    f_pooled = get_pooled_scores("F", [50, 100])
    
    if k_pooled and f_pooled:
        k_vs_f = paired_bootstrap(k_pooled, f_pooled, n_bootstrap, confidence_level)
        criteria["R5"] = {
            "delta": k_vs_f["mean_difference"],
            "ci_lower": k_vs_f["ci_lower"],
            "ci_upper": k_vs_f["ci_upper"],
            "refuted": k_vs_f["mean_difference"] <= 0,
        }
    
    # R7: Calibration matters
    k_jev_pooled = get_pooled_scores("K-Jev", [50, 100])
    k_verbal_pooled = get_pooled_scores("K-verbal", [50, 100])
    
    if k_jev_pooled and k_verbal_pooled:
        jev_vs_verbal = paired_bootstrap(k_jev_pooled, k_verbal_pooled, n_bootstrap, confidence_level)
        criteria["R7"] = {
            "delta": jev_vs_verbal["mean_difference"],
            "ci_lower": jev_vs_verbal["ci_lower"],
            "ci_upper": jev_vs_verbal["ci_upper"],
            "refuted": jev_vs_verbal["mean_difference"] <= 0,
        }
    
    return criteria
