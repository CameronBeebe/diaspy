"""Experiment configuration and budget accounting."""
import dataclasses
from typing import Optional, Literal
import dspy


@dataclasses.dataclass
class BudgetConfig:
    """Per-run budget constraints."""
    max_program_rollouts: int = 1000
    max_reflection_calls: int = 60
    max_reflection_tokens: int = 300_000
    tolerance: float = 0.05  # discard runs >5% off budget


@dataclasses.dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Arms to run
    arms: list[str] = dataclasses.field(default_factory=lambda: ["A0", "A", "E", "C", "B", "F"])
    
    # Data regime
    n_labels: list[int] = dataclasses.field(default_factory=lambda: [10, 25, 50, 100])
    seeds: list[int] = dataclasses.field(default_factory=lambda: list(range(5)))
    
    # Budget
    budget: BudgetConfig = dataclasses.field(default_factory=BudgetConfig)
    
    # Model config (for real runs, not DummyLM tests)
    task_model: Optional[str] = None  # e.g., "openrouter/qwen/qwen3-8b"
    reflection_model: Optional[str] = None
    
    # Task-specific
    task: Literal["ifbench"] = "ifbench"
    
    # Output
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    
    # Optimization hyperparams
    gepa_reflection_minibatch_size: int = 3
    gepa_use_merge: bool = False
    proposer_max_attempts: int = 2
    proposer_tau: float = 0.5
    proposer_archive_cap: int = 6
    
    # SIMBA-specific
    simba_max_demos: int = 0  # prompt-only by design
    simba_bsize: Optional[int] = None  # defaults to min(32, |train|)


class BudgetTracker:
    """Tracks and enforces budget constraints with LM wrapper."""
    
    def __init__(self, config: BudgetConfig):
        self.config = config
        self.program_rollouts = 0
        self.reflection_calls = 0
        self.reflection_tokens = 0
        self.overbudget = False
    
    def count_program_call(self, n: int = 1):
        """Count task program rollouts."""
        self.program_rollouts += n
        if self.program_rollouts > self.config.max_program_rollouts:
            self.overbudget = True
    
    def count_reflection_call(self, input_tokens: int, output_tokens: int):
        """Count reflection/critic LM calls."""
        self.reflection_calls += 1
        self.reflection_tokens += input_tokens + output_tokens
        
        if self.reflection_calls > self.config.max_reflection_calls:
            self.overbudget = True
        if self.reflection_tokens > self.config.max_reflection_tokens:
            self.overbudget = True
    
    def within_tolerance(self) -> bool:
        """Check if actual usage is within tolerance of budget."""
        rollout_ratio = self.program_rollouts / self.config.max_program_rollouts
        call_ratio = self.reflection_calls / self.config.max_reflection_calls
        token_ratio = self.reflection_tokens / self.config.max_reflection_tokens
        
        max_ratio = max(rollout_ratio, call_ratio, token_ratio)
        return max_ratio <= (1 + self.config.tolerance)
    
    def report(self) -> dict:
        """Return usage statistics."""
        return {
            "program_rollouts": self.program_rollouts,
            "program_rollouts_budget": self.config.max_program_rollouts,
            "reflection_calls": self.reflection_calls,
            "reflection_calls_budget": self.config.max_reflection_calls,
            "reflection_tokens": self.reflection_tokens,
            "reflection_tokens_budget": self.config.max_reflection_tokens,
            "overbudget": self.overbudget,
            "within_tolerance": self.within_tolerance()
        }


class BudgetedLM(dspy.clients.lm.LM):
    """Wrapper that tracks calls and enforces budget.
    
    Inherits from dspy.clients.lm.LM to satisfy DSPy type checks.
    """
    
    def __init__(self, lm, tracker: BudgetTracker, is_reflection: bool = False):
        # Don't call super().__init__ as we're wrapping, not creating
        self.lm = lm
        self.tracker = tracker
        self.is_reflection = is_reflection
        self._history = []
        
        # Copy over essential LM attributes
        if hasattr(lm, 'model'):
            self.model = lm.model
        if hasattr(lm, 'provider'):
            self.provider = getattr(lm, 'provider', None)
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        """Forward call to underlying LM and track."""
        if self.tracker.overbudget:
            raise RuntimeError("Budget exceeded")
        
        response = self.lm(prompt=prompt, messages=messages, **kwargs)
        
        # Track token usage
        if hasattr(response, 'usage'):
            input_toks = getattr(response.usage, 'input_tokens', 0)
            output_toks = getattr(response.usage, 'output_tokens', 0)
        else:
            # Estimate for DummyLM
            input_toks = len(str(prompt or messages)) // 4
            output_toks = 100
        
        if self.is_reflection:
            self.tracker.count_reflection_call(input_toks, output_toks)
        else:
            self.tracker.count_program_call()
        
        self._history.append({
            "input_tokens": input_toks,
            "output_tokens": output_toks,
            "is_reflection": self.is_reflection
        })
        
        return response
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying LM."""
        if name in ('lm', 'tracker', 'is_reflection', '_history', 'model', 'provider'):
            return object.__getattribute__(self, name)
        return getattr(self.lm, name)
