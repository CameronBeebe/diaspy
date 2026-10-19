"""Paired bootstrap for statistical inference."""

import numpy as np
from typing import List, Tuple


def paired_bootstrap(
    scores_a: List[float],
    scores_b: List[float],
    n_samples: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> dict:
    """Paired bootstrap for comparing two arms.
    
    Args:
        scores_a: Scores for arm A
        scores_b: Scores for arm B
        n_samples: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95)
        random_seed: Random seed
        
    Returns:
        Dictionary with mean difference, CI, and p-value
    """
    rng = np.random.RandomState(random_seed)
    
    scores_a_array = np.array(scores_a)
    scores_b_array = np.array(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length for paired test")
    
    n = len(scores_a)
    differences = scores_a_array - scores_b_array
    observed_mean_diff = np.mean(differences)
    
    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_samples):
        indices = rng.choice(n, size=n, replace=True)
        boot_diff = np.mean(differences[indices])
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    
    # P-value (two-tailed)
    p_value = np.mean(bootstrap_diffs <= 0) if observed_mean_diff > 0 else np.mean(bootstrap_diffs >= 0)
    p_value = 2 * min(p_value, 1 - p_value)
    
    return {
        "mean_difference": observed_mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
        "p_value": p_value,
        "n_samples": n_samples,
        "significant": 0 not in (lower, upper) or (lower > 0 or upper < 0),
    }


def compute_confidence_interval(
    values: List[float],
    confidence_level: float = 0.95,
    n_samples: int = 10000,
    random_seed: int = 42
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean.
    
    Args:
        values: List of values
        confidence_level: Confidence level
        n_samples: Number of bootstrap samples
        random_seed: Random seed
        
    Returns:
        Tuple of (lower, upper) CI bounds
    """
    rng = np.random.RandomState(random_seed)
    values_array = np.array(values)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_samples):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means.append(np.mean(values_array[indices]))
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper
