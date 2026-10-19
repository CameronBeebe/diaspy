"""Epistemic event traces for optimization steps.

Provides structured, typed events that record the reasoning behind each
optimization decision, enabling analysis and theory-driven feedback loops.
"""

from .schema import (
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
from .emitter import EventEmitter
from .storage import EventStorage
from .analysis import EventAnalyzer

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
