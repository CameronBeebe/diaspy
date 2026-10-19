"""Optimizer arm implementations.

Arms:
- A: GEPA default InstructionProposer
- C: GEPA + untyped prose critic
- B: GEPA + typed Popperian proposer (Refutation objects)
- K-Jev: Clause-level Bayesian/Popperian with Jev credences
- K-verbal: Clause-level with verbalized credences
- E: GEPA + check only (no critic)
- F: dspy.SIMBA
"""

from .base import BaseArm
from .gepa_default import GEPADefaultArm
from .prose_critic import ProseCriticArm
from .popperian import PopperianArm
from .bayesian_popperian import BayesianPopperianArm

__all__ = [
    "BaseArm",
    "GEPADefaultArm",
    "ProseCriticArm",
    "PopperianArm",
    "BayesianPopperianArm",
]
