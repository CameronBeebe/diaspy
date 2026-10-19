"""Analysis module for experiment results.

Implements:
- Paired bootstrap for refutation criteria (R1-R7)
- Brier/ECE calibration checks
- Progressive vs degenerating revision logging (Lakatos)
"""

from .bootstrap import paired_bootstrap, compute_confidence_interval
from .refutation_criteria import check_refutation_criteria
from .calibration import compute_brier_score, compute_ece

__all__ = [
    "paired_bootstrap",
    "compute_confidence_interval",
    "check_refutation_criteria",
    "compute_brier_score",
    "compute_ece",
]
