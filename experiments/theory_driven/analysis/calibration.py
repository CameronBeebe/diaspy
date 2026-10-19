"""Calibration metrics: Brier score and ECE for clause credences."""

import numpy as np
from typing import List, Tuple


def compute_brier_score(
    predictions: List[float],
    outcomes: List[int]
) -> float:
    """Compute Brier score.
    
    Args:
        predictions: Predicted probabilities (0 to 1)
        outcomes: Actual outcomes (0 or 1)
        
    Returns:
        Brier score (lower is better, 0 is perfect)
    """
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    return float(np.mean((predictions - outcomes) ** 2))


def compute_brier_skill_score(
    predictions: List[float],
    outcomes: List[int],
    baseline_rate: float = None
) -> float:
    """Compute Brier skill score against base rate.
    
    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        baseline_rate: Base rate (if None, use mean of outcomes)
        
    Returns:
        Brier skill score (positive means better than baseline)
    """
    brier = compute_brier_score(predictions, outcomes)
    
    if baseline_rate is None:
        baseline_rate = np.mean(outcomes)
    
    baseline_brier = compute_brier_score(
        [baseline_rate] * len(outcomes),
        outcomes
    )
    
    return (baseline_brier - brier) / baseline_brier


def compute_ece(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Args:
        predictions: Predicted probabilities (0 to 1)
        outcomes: Actual outcomes (0 or 1)
        n_bins: Number of probability bins
        
    Returns:
        ECE (0 to 1, lower is better)
    """
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins[:-1]) - 1
    
    ece = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_conf = np.mean(predictions[mask])
            bin_acc = np.mean(outcomes[mask])
            bin_size = np.sum(mask)
            
            ece += (bin_size / len(predictions)) * abs(bin_conf - bin_acc)
    
    return float(ece)


def analyze_calibration(
    clause_credences: List[float],
    clause_outcomes: List[int],
    n_bins: int = 10
) -> dict:
    """Analyze calibration of clause credences.
    
    Per spec: Jev credences should predict ablation outcomes.
    
    Args:
        clause_credences: Predicted credences for clauses
        clause_outcomes: Whether clause was actually implicated (1) or not (0)
        n_bins: Number of bins for ECE
        
    Returns:
        Dictionary with calibration metrics
    """
    brier = compute_brier_score(clause_credences, clause_outcomes)
    skill = compute_brier_skill_score(clause_credences, clause_outcomes)
    ece = compute_ece(clause_credences, clause_outcomes, n_bins)
    
    return {
        "brier_score": brier,
        "brier_skill_score": skill,
        "ece": ece,
        "n_clauses": len(clause_credences),
        "base_rate": np.mean(clause_outcomes),
    }
