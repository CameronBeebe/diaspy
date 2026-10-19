"""Epistemic event traces for optimization steps.

Provides structured, typed events that record the reasoning behind each
optimization decision, enabling analysis and theory-driven feedback loops.
"""

from experiments.theory_driven.events.schema import (
    EpistemicEvent,
    Conjecture,
    ClauseCredence,
    TestSelection,
    TestOutcome,
    AcceptRejectDecision,
    LakatosLabel,
    Refutation,
    OptimizationStep,
)
from experiments.theory_driven.events.emitter import EventEmitter
from experiments.theory_driven.events.storage import EventStorage
from experiments.theory_driven.events.analysis import EventAnalyzer

__all__ = [
    "EpistemicEvent",
    "Conjecture",
    "ClauseCredence",
    "TestSelection",
    "TestOutcome",
    "AcceptRejectDecision",
    "LakatosLabel",
    "Refutation",
    "OptimizationStep",
    "EventEmitter",
    "EventStorage",
    "EventAnalyzer",
]
