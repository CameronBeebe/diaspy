"""Experiment configuration definitions."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class ModelConfig:
    """Configuration for LLM models.
    
    Primary model: Qwen3.5-9B via OpenRouter (thinking disabled)
    Note: Qwen3-8B no longer available pay-per-token. GEPA paper likely ran
    Qwen3-8B in thinking mode (temp 0.6/top-p 0.95/top-k 20), so baseline
    numbers are not directly comparable to our non-thinking runs.
    """
    # Primary task/reflection model
    task_model: str = "openrouter/qwen/qwen3.5-9b"
    reflection_model: str = "openrouter/qwen/qwen3.5-9b"
    
    # Reasoning control (OpenRouter's reasoning parameter)
    # For Qwen3.5: set to False to disable thinking/reasoning tokens
    enable_reasoning: bool = False
    
    # Alternative models
    baseline_check_model: str = "openrouter/openai/gpt-4.1-mini"  # For published baseline check
    strong_model: str = "xai/grok-4.3"  # Strong-model comparison
    strong_model_reasoning_effort: str = "none"  # For grok: 'none', 'low', 'medium', 'high'
    
    # Model provider configuration (via env vars)
    litellm_api_base: Optional[str] = None
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    xai_api_key_env: str = "XAI_API_KEY"
    
    # Legacy/fallback
    litellm_api_key_env: str = "OPENAI_API_KEY"
    
    # Alternative: local vLLM endpoint
    vllm_endpoint: Optional[str] = None  # e.g., "http://localhost:8000/v1"
    
    # Cost controls
    max_cost_dollars: float = 5.0  # Hard cap for pilot
    
    # Cost tracking: prefer provider-reported costs when available
    # Fallback estimates (approximate, for budget planning only)
    cost_per_million_input: float = 0.15  # Approximate for Qwen3.5-9B
    cost_per_million_output: float = 0.60
    
    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Load model config from environment variables."""
        return cls(
            task_model=os.getenv("TASK_MODEL", "openrouter/qwen/qwen3.5-9b"),
            reflection_model=os.getenv("REFLECTION_MODEL", "openrouter/qwen/qwen3.5-9b"),
            enable_reasoning=os.getenv("ENABLE_REASONING", "false").lower() == "true",
            baseline_check_model=os.getenv("BASELINE_CHECK_MODEL", "openrouter/openai/gpt-4.1-mini"),
            strong_model=os.getenv("STRONG_MODEL", "xai/grok-4.3"),
            strong_model_reasoning_effort=os.getenv("STRONG_MODEL_REASONING_EFFORT", "none"),
            litellm_api_base=os.getenv("LITELLM_API_BASE"),
            openrouter_api_key_env=os.getenv("OPENROUTER_API_KEY_ENV", "OPENROUTER_API_KEY"),
            xai_api_key_env=os.getenv("XAI_API_KEY_ENV", "XAI_API_KEY"),
            litellm_api_key_env=os.getenv("LITELLM_API_KEY_ENV", "OPENAI_API_KEY"),
            vllm_endpoint=os.getenv("VLLM_ENDPOINT"),
            max_cost_dollars=float(os.getenv("MAX_COST_DOLLARS", "5.0")),
        )
    
    def get_litellm_kwargs(
        self,
        model: str,
        pin_provider: bool = True,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get LiteLLM kwargs for a model, including reasoning control and provider pinning.
        
        Per OpenRouter docs: pass reasoning=False via extra_body for models
        that support reasoning/thinking tokens (Qwen3.5, etc).
        
        Per xAI docs: pass reasoning_effort='none' for Grok models.
        
        Provider pinning (OpenRouter): Set provider.order and provider.allow_fallbacks
        to ensure consistent routing and prevent silent fallbacks.
        
        Args:
            model: Model string (e.g., 'openrouter/qwen/qwen3.5-9b')
            pin_provider: Whether to pin provider routing (recommended for reproducibility)
            provider_name: Specific provider to pin (e.g., 'OpenAI', 'Together', 'Fireworks')
                          If None, lets OpenRouter choose but disables fallbacks
            
        Returns:
            Dictionary of kwargs for LiteLLM
        """
        kwargs = {}
        extra_body = {}
        
        # OpenRouter reasoning control
        if "openrouter/" in model and not self.enable_reasoning:
            # OpenRouter: pass reasoning parameter via extra_body
            # This disables thinking/reasoning tokens for models that support them
            extra_body["reasoning"] = False
        
        # OpenRouter provider pinning
        if "openrouter/" in model and pin_provider:
            provider_config: Dict[str, Any] = {
                "allow_fallbacks": False,  # Fail instead of falling back
                "require_parameters": True,  # Require our params to be respected
            }
            
            if provider_name:
                # Pin to specific provider
                provider_config["order"] = [provider_name]
            
            extra_body["provider"] = provider_config
        
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        # xAI Grok reasoning effort
        if "xai/grok" in model or "grok-" in model:
            kwargs["reasoning_effort"] = self.strong_model_reasoning_effort
        
        return kwargs


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""
    name: str
    train_size: int = 20
    val_size: int = 20
    test_size: Optional[int] = None  # Full test split if None
    
    # Benchmark-specific settings
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmConfig:
    """Configuration for an optimization arm."""
    name: str
    arm_type: str  # "gepa", "popperian", "bayesian_jev", "bayesian_verbal", "check_only", "simba"
    description: str
    enabled: bool = True
    
    # Arm-specific parameters
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageConfig:
    """Configuration for a single experiment stage."""
    stage_id: int
    name: str
    description: str
    max_cost_dollars: float
    budgets: List[int]
    n_seeds: int
    test_size: int
    benchmarks: List[str]
    arms: List[str]  # Arm names to include
    models: List[str]  # Model strings to test
    
    # Stage gate requirements
    requires_stage_gate: bool = False
    required_checks: List[str] = field(default_factory=list)
    
    # Canary suite for drift detection
    run_canary: bool = True
    canary_model: Optional[str] = None  # Model for canaries (can use free tier for smoke)
    allow_canary_drift: bool = False  # If False, stage blocks on drift detection


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    
    # Experiment metadata
    name: str = "theory-driven-low-budget"
    description: str = "Test critique-first optimizers at low budgets"
    
    # Budget settings
    budgets: List[int] = field(default_factory=lambda: [50, 100, 200])
    pilot_budgets: List[int] = field(default_factory=lambda: [50, 100])
    
    # Data settings
    train_size: int = 20
    val_size: int = 20
    test_size: int = 50
    n_seeds: int = 5
    pilot_seeds: int = 2
    pilot_test_size: int = 100  # Subset for pilot
    smoke_test_size: int = 50  # Subset for smoke test
    
    # Model configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Benchmarks
    benchmarks: List[BenchmarkConfig] = field(default_factory=lambda: [
        BenchmarkConfig(name="hotpotqa", test_size=50),
        BenchmarkConfig(name="hover", extra_kwargs={"retrieval": "bm25s"}),
        BenchmarkConfig(name="pupa"),
    ])
    
    # Arms (pilot subset by default)
    arms: List[ArmConfig] = field(default_factory=lambda: [
        ArmConfig(
            name="A",
            arm_type="gepa",
            description="GEPA default InstructionProposer",
        ),
        ArmConfig(
            name="C",
            arm_type="prose_critic",
            description="GEPA + untyped prose critic",
        ),
        ArmConfig(
            name="B",
            arm_type="popperian",
            description="GEPA + typed Popperian proposer with Refutation objects",
        ),
        ArmConfig(
            name="K-Jev",
            arm_type="bayesian_jev",
            description="Clause-level Bayesian/Popperian with Jev credences",
            kwargs={"credence_method": "jev"},
        ),
        ArmConfig(
            name="K-verbal",
            arm_type="bayesian_verbal",
            description="Clause-level with verbalized credences",
            kwargs={"credence_method": "verbal"},
        ),
    ])
    
    # Full grid additional arms
    full_grid_arms: List[ArmConfig] = field(default_factory=lambda: [
        ArmConfig(
            name="E",
            arm_type="check_only",
            description="GEPA + survival check, no critic",
        ),
        ArmConfig(
            name="F",
            arm_type="simba",
            description="dspy.SIMBA with reflection_lm",
        ),
    ])
    
    # Analysis settings
    bootstrap_samples: int = 10000
    confidence_level: float = 0.95
    holm_adjust: bool = True  # For co-primaries K-Jev and B
    
    # Results storage
    results_dir: str = "experiments/theory_driven/results"
    mlflow_tracking_uri: str = "experiments/theory_driven/mlruns"
    
    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_file(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    def get_pilot_config(self) -> "ExperimentConfig":
        """Return a pilot configuration with reduced scope."""
        pilot = ExperimentConfig(
            name=f"{self.name}-pilot",
            description=f"Pilot: {self.description}",
            budgets=self.pilot_budgets,
            n_seeds=self.pilot_seeds,
            train_size=self.train_size,
            val_size=self.val_size,
            model_config=self.model_config,
            benchmarks=[b for b in self.benchmarks if b.name == "hover"],  # HoVer only
            arms=self.arms,  # Pilot arms only
            results_dir=f"{self.results_dir}/pilot",
            mlflow_tracking_uri=f"{self.mlflow_tracking_uri}_pilot",
        )
        return pilot
    
    def get_full_grid_config(self) -> "ExperimentConfig":
        """Return full grid configuration with all arms and budgets."""
        full = ExperimentConfig(
            name=f"{self.name}-full",
            description=f"Full grid: {self.description}",
            budgets=self.budgets + [400],
            n_seeds=self.n_seeds,
            train_size=self.train_size,
            val_size=self.val_size,
            model_config=self.model_config,
            benchmarks=self.benchmarks,
            arms=self.arms + self.full_grid_arms,
            results_dir=f"{self.results_dir}/full",
            mlflow_tracking_uri=f"{self.mlflow_tracking_uri}_full",
        )
        return full
    
    def get_stage_config(self, stage_id: int) -> StageConfig:
        """Get configuration for a specific stage.
        
        Stage 0 (smoke): $1 cap, 1 benchmark, 2 arms, budget 50, 1 seed, 20-item test
        Stage 1 (pilot): $5 cap, HoVer, 5 arms, budgets 50/100, 2 seeds, 100-item test
        Stage 2 (cross-model): $25 cap, same as pilot but 2-3 models
        """
        if stage_id == 0:
            # Stage 0 uses free-tier model for $0 smoke test
            # WARNING: Free providers may log prompts and change without notice
            free_model = "openrouter/google/gemma-4-26b-a4b-it:free"
            
            return StageConfig(
                stage_id=0,
                name="smoke",
                description="$0 smoke test with free model",
                max_cost_dollars=1.0,  # Cap, but should be $0 with free model
                budgets=[50],
                n_seeds=1,
                test_size=self.test_size,
                benchmarks=["hotpotqa"],
                arms=["A", "B"],  # GEPA default + Popperian only for slim core
                models=[free_model],  # Use free model for stage 0
                requires_stage_gate=True,
                required_checks=[
                    "call_accounting_matches_provider",
                    "cap_aborts_correctly",
                    "results_populated",
                    "dashboard_accessible",
                    "metrics_parse",
                    "canary_no_drift",  # Canary check required
                ],
                run_canary=True,
                canary_model=free_model,
                allow_canary_drift=False,  # Block on drift (error should fail the stage)
            )
        elif stage_id == 1:
            return StageConfig(
                stage_id=1,
                name="pilot",
                description="Full pilot with all arms",
                max_cost_dollars=5.0,
                budgets=self.pilot_budgets,
                n_seeds=self.pilot_seeds,
                test_size=self.pilot_test_size,
                benchmarks=["hover"],
                arms=["A", "C", "B", "K-Jev", "K-verbal"],
                models=[self.model_config.task_model],
                requires_stage_gate=False,
                required_checks=[],
                run_canary=True,
                canary_model=self.model_config.task_model,  # Use paid model
                allow_canary_drift=False,  # Block on drift
            )
        elif stage_id == 2:
            # Cross-model: test on 2-3 cheap models
            models = [
                self.model_config.task_model,  # Default: qwen/qwen-3-8b
                "openai/gpt-4o-mini",  # Documented alternative from spec
                "anthropic/claude-3-haiku-20240307",  # Another cheap option
            ]
            # Cross-model: Qwen3.5-9B (default), GPT-4.1-mini, Grok-4.3
            models = [
                self.model_config.task_model,  # Default: Qwen3.5-9B
                self.model_config.baseline_check_model,  # GPT-4.1-mini
                self.model_config.strong_model,  # Grok-4.3
            ]
            return StageConfig(
                stage_id=2,
                name="cross-model",
                description="Pilot on multiple models: Qwen3.5-9B, GPT-4.1-mini, Grok-4.3",
                max_cost_dollars=25.0,
                budgets=self.pilot_budgets,
                n_seeds=self.pilot_seeds,
                test_size=self.pilot_test_size,
                benchmarks=["hover"],
                arms=["A", "C", "B", "K-Jev", "K-verbal"],
                models=models,
                requires_stage_gate=False,
                required_checks=[],
                run_canary=True,
                canary_model=None,  # Will run canaries for each model
                allow_canary_drift=True,  # Allow drift (comparing different models)
            )
        else:
            raise ValueError(f"Unknown stage_id: {stage_id}")
