"""Unified budget tracking for experiments.

Tracks both call counts and token-weighted costs. One unit = one rollout
or one reflection/critic call. Per spec, GEPA can overshoot by one iteration.
"""

from typing import List, Optional, Dict, Any
import threading


class BudgetTracker:
    """Thread-safe budget tracker for unified call accounting."""
    
    def __init__(
        self,
        max_units: int,
        cost_per_million_input: float = 0.117,
        cost_per_million_output: float = 0.455,
    ):
        """Initialize budget tracker.
        
        Args:
            max_units: Maximum number of units (calls) allowed
            cost_per_million_input: Cost per million input tokens
            cost_per_million_output: Cost per million output tokens
        """
        self.max_units = max_units
        self.cost_per_million_input = cost_per_million_input
        self.cost_per_million_output = cost_per_million_output
        
        self._lock = threading.RLock()  # RLock to allow re-entrant access
        self._task_lm_calls = 0
        self._reflection_lm_calls = 0
        self._jev_calls = 0
        
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        
        self._call_log: List[Dict[str, Any]] = []
    
    def record_task_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider_cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a task LM call.
        
        Args:
            input_tokens: Input tokens (for fallback calculation)
            output_tokens: Output tokens (for fallback calculation)
            provider_cost: Provider-reported cost (preferred if available)
            metadata: Additional metadata
        """
        with self._lock:
            self._task_lm_calls += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            
            # Prefer provider-reported cost (e.g., from OpenRouter)
            if provider_cost is not None:
                cost = provider_cost
            else:
                # Fallback: estimate from tokens
                cost = (
                    input_tokens * self.cost_per_million_input / 1_000_000 +
                    output_tokens * self.cost_per_million_output / 1_000_000
                )
            
            self._total_cost += cost
            self._call_log.append({
                "type": "task_lm",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "provider_cost": provider_cost is not None,
                "metadata": metadata or {},
            })
    
    def record_reflection_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider_cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a reflection/critic LM call.
        
        Args:
            input_tokens: Input tokens (for fallback calculation)
            output_tokens: Output tokens (for fallback calculation)
            provider_cost: Provider-reported cost (preferred if available)
            metadata: Additional metadata
        """
        with self._lock:
            self._reflection_lm_calls += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            
            # Prefer provider-reported cost
            if provider_cost is not None:
                cost = provider_cost
            else:
                # Fallback: estimate from tokens
                cost = (
                    input_tokens * self.cost_per_million_input / 1_000_000 +
                    output_tokens * self.cost_per_million_output / 1_000_000
                )
            
            self._total_cost += cost
            self._call_log.append({
                "type": "reflection_lm",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "provider_cost": provider_cost is not None,
                "metadata": metadata or {},
            })
    
    def record_jev_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a Jev (TypeSafe) call.
        
        Jev has different pricing: $0.042 per million input, output free.
        """
        with self._lock:
            self._jev_calls += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost += cost
            self._call_log.append({
                "type": "jev",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "metadata": metadata or {},
            })
    
    @property
    def total_units(self) -> int:
        """Total units used (task calls + reflection calls + jev calls)."""
        with self._lock:
            return self._task_lm_calls + self._reflection_lm_calls + self._jev_calls
    
    @property
    def total_cost(self) -> float:
        """Total cost in dollars."""
        with self._lock:
            return self._total_cost
    
    @property
    def budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.total_units >= self.max_units
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current budget statistics."""
        with self._lock:
            return {
                "max_units": self.max_units,
                "total_units": self.total_units,
                "task_lm_calls": self._task_lm_calls,
                "reflection_lm_calls": self._reflection_lm_calls,
                "jev_calls": self._jev_calls,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_cost": self._total_cost,
                "budget_exhausted": self.budget_exhausted,
            }
    
    def get_call_log(self) -> List[Dict[str, Any]]:
        """Get the full call log."""
        with self._lock:
            return self._call_log.copy()
    
    def __call__(self, gepa_state: Any) -> bool:
        """GEPA stop callback: stop when budget is exhausted.
        
        Note: GEPA can overshoot by one iteration as documented in spec.
        """
        return self.budget_exhausted


class CostLimitExceeded(Exception):
    """Raised when cost limit is exceeded."""
    pass


class BudgetEnforcer:
    """Enforces hard cost limits on experiments."""
    
    def __init__(self, max_cost_dollars: float, tracker: BudgetTracker):
        """Initialize budget enforcer.
        
        Args:
            max_cost_dollars: Maximum allowed cost in dollars
            tracker: Budget tracker to monitor
        """
        self.max_cost_dollars = max_cost_dollars
        self.tracker = tracker
    
    def check(self) -> None:
        """Check if cost limit is exceeded and raise exception if so."""
        if self.tracker.total_cost > self.max_cost_dollars:
            raise CostLimitExceeded(
                f"Cost limit exceeded: ${self.tracker.total_cost:.2f} > "
                f"${self.max_cost_dollars:.2f}"
            )
