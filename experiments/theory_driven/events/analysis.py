"""Analysis tools for epistemic events."""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .schema import EpistemicEvent, LakatosLabel


class EventAnalyzer:
    """Analyzes epistemic events to extract insights."""
    
    def __init__(self, events: List[EpistemicEvent]):
        """Initialize analyzer.
        
        Args:
            events: List of events to analyze
        """
        self.events = events
    
    def build_lineage_tree(self) -> Dict[str, List[str]]:
        """Build lineage tree from candidate IDs.
        
        Returns:
            Dict mapping parent_id -> list of child_ids
        """
        tree = defaultdict(list)
        
        for event in self.events:
            if event.event_type == "optimization_step" and event.step:
                conjecture = event.step.conjecture
                parent_id = conjecture.parent_id or "root"
                candidate_id = conjecture.candidate_id
                
                tree[parent_id].append(candidate_id)
        
        return dict(tree)
    
    def get_lineage_path(self, candidate_id: str) -> List[str]:
        """Get full lineage path from root to candidate.
        
        Args:
            candidate_id: Candidate ID
            
        Returns:
            List of candidate IDs from root to this candidate
        """
        # Build parent mapping
        parent_map = {}
        for event in self.events:
            if event.event_type == "optimization_step" and event.step:
                conjecture = event.step.conjecture
                if conjecture.parent_id:
                    parent_map[conjecture.candidate_id] = conjecture.parent_id
        
        # Trace back to root
        path = [candidate_id]
        current = candidate_id
        
        while current in parent_map:
            current = parent_map[current]
            path.insert(0, current)
        
        return path
    
    def count_by_lakatos_label(self) -> Dict[LakatosLabel, int]:
        """Count steps by Lakatos label.
        
        Returns:
            Dict mapping label -> count
        """
        counts = defaultdict(int)
        
        for event in self.events:
            if event.event_type == "optimization_step" and event.step:
                label = event.step.lakatos_label
                counts[label] += 1
        
        return dict(counts)
    
    def get_progressive_ratio(self) -> float:
        """Get ratio of progressive to total labeled steps.
        
        Returns:
            Ratio (0-1) or -1 if no labeled steps
        """
        counts = self.count_by_lakatos_label()
        
        progressive = counts.get(LakatosLabel.PROGRESSIVE, 0)
        degenerating = counts.get(LakatosLabel.DEGENERATING, 0)
        total_labeled = progressive + degenerating
        
        if total_labeled == 0:
            return -1.0
        
        return progressive / total_labeled
    
    def compute_cost_breakdown(self) -> Dict[str, Any]:
        """Compute cost breakdown across steps.
        
        Returns:
            Dict with cost statistics
        """
        total_calls = 0
        total_cost = 0.0
        
        costs_by_label = defaultdict(lambda: {"calls": 0, "cost": 0.0})
        
        for event in self.events:
            if event.event_type == "optimization_step" and event.step:
                step = event.step
                
                total_calls += step.calls_consumed
                total_cost += step.cost_consumed
                
                label = step.lakatos_label
                costs_by_label[label]["calls"] += step.calls_consumed
                costs_by_label[label]["cost"] += step.cost_consumed
        
        return {
            "total_calls": total_calls,
            "total_cost": total_cost,
            "by_label": dict(costs_by_label),
        }
    
    def extract_timeline(self) -> List[Dict[str, Any]]:
        """Extract timeline of events for visualization.
        
        Returns:
            List of timeline entries
        """
        timeline = []
        
        for event in self.events:
            entry = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
            }
            
            if event.event_type == "optimization_step" and event.step:
                step = event.step
                entry.update({
                    "step_id": step.step_id,
                    "iteration": step.iteration,
                    "candidate_id": step.conjecture.candidate_id,
                    "parent_id": step.conjecture.parent_id,
                    "accepted": step.decision.accepted,
                    "lakatos_label": step.lakatos_label.value,
                    "calls": step.calls_consumed,
                    "cost": step.cost_consumed,
                })
            
            timeline.append(entry)
        
        return timeline
    
    def compare_candidates(
        self,
        candidate_id_a: str,
        candidate_id_b: str
    ) -> Dict[str, Any]:
        """Compare two candidates.
        
        Args:
            candidate_id_a: First candidate ID
            candidate_id_b: Second candidate ID
            
        Returns:
            Comparison data
        """
        # Find events for each candidate
        event_a = None
        event_b = None
        
        for event in self.events:
            if event.event_type == "optimization_step" and event.step:
                cid = event.step.conjecture.candidate_id
                if cid == candidate_id_a:
                    event_a = event
                if cid == candidate_id_b:
                    event_b = event
        
        if not event_a or not event_b:
            return {"error": "One or both candidates not found"}
        
        step_a = event_a.step
        step_b = event_b.step
        
        return {
            "candidate_a": {
                "id": candidate_id_a,
                "instruction": step_a.conjecture.new_instruction,
                "score": step_a.test_outcome.score,
                "accepted": step_a.decision.accepted,
                "lakatos": step_a.lakatos_label.value,
                "calls": step_a.calls_consumed,
            },
            "candidate_b": {
                "id": candidate_id_b,
                "instruction": step_b.conjecture.new_instruction,
                "score": step_b.test_outcome.score,
                "accepted": step_b.decision.accepted,
                "lakatos": step_b.lakatos_label.value,
                "calls": step_b.calls_consumed,
            },
        }
    
    def get_refutation_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns from refutations (for Popperian arms).
        
        Returns:
            List of refutation patterns
        """
        patterns = []
        
        for event in self.events:
            if event.event_type == "optimization_step" and event.step:
                if event.step.refutations:
                    for ref in event.step.refutations:
                        patterns.append({
                            "step_id": event.step.step_id,
                            "claim": ref.claim,
                            "severity": ref.severity,
                            "n_counterexamples": len(ref.counterexample_ids),
                            "fix": ref.proposed_fix,
                        })
        
        return patterns
    
    def get_clause_credence_evolution(self) -> Dict[int, List[float]]:
        """Track how clause credences evolve over time.
        
        Returns:
            Dict mapping clause_id -> list of credences over time
        """
        evolution = defaultdict(list)
        
        for event in self.events:
            if event.event_type == "optimization_step" and event.step:
                if event.step.clause_credences:
                    for cc in event.step.clause_credences:
                        evolution[cc.clause_id].append(cc.credence_after or cc.credence_before)
        
        return dict(evolution)
