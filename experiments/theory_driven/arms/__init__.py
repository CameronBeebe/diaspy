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

from experiments.theory_driven.arms.base import BaseArm
from experiments.theory_driven.arms.gepa_default import GEPADefaultArm
from experiments.theory_driven.arms.prose_critic import ProseCriticArm
from experiments.theory_driven.arms.popperian import PopperianArm
from experiments.theory_driven.arms.millian import MillianArm
from experiments.theory_driven.arms.bayesian_popperian import BayesianPopperianArm

__all__ = [
    "BaseArm",
    "GEPADefaultArm",
    "ProseCriticArm",
    "PopperianArm",
    "MillianArm",
    "BayesianPopperianArm",
]
