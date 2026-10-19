"""Main experiment harness for running and managing experiments."""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import dspy

from .config import (
    ExperimentConfig,
    BudgetTracker,
    BudgetEnforcer,
    CostLimitExceeded,
    StageConfig,
)
from .arms import (
    BaseArm,
    GEPADefaultArm,
    ProseCriticArm,
    PopperianArm,
    BayesianPopperianArm,
)
from .benchmarks import BaseBenchmark, HoVerBenchmark, PUPABenchmark, SimpleQABenchmark


class ExperimentHarness:
    """Main harness for running theory-driven experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize harness.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LMs (only if API keys are set)
        self.task_lm = None
        self.reflection_lm = None
        self._init_lms()
    
    def _init_lms(self) -> None:
        """Initialize language models from config.
        
        Only initializes if API keys are present.
        Handles OpenRouter reasoning control and xAI reasoning_effort.
        """
        model_config = self.config.model_config
        
        # Determine which API key to use based on model
        api_key = None
        if "openrouter/" in model_config.task_model:
            api_key = os.getenv(model_config.openrouter_api_key_env)
        elif "xai/" in model_config.task_model or "grok" in model_config.task_model:
            api_key = os.getenv(model_config.xai_api_key_env)
        else:
            api_key = os.getenv(model_config.litellm_api_key_env)
        
        if not api_key and not model_config.vllm_endpoint:
            # No API keys or local endpoint - models will not be initialized
            # This is expected for offline tests
            return
        
        try:
            # Task LM
            if model_config.vllm_endpoint:
                # Use local vLLM endpoint
                self.task_lm = dspy.LM(
                    model=f"openai/{model_config.task_model}",
                    api_base=model_config.vllm_endpoint,
                    api_key="dummy",  # vLLM doesn't need real key
                )
            else:
                # Get LiteLLM kwargs (includes reasoning control)
                litellm_kwargs = model_config.get_litellm_kwargs(model_config.task_model)
                
                # Use LiteLLM with provider
                self.task_lm = dspy.LM(
                    model=model_config.task_model,
                    api_key=api_key,
                    **litellm_kwargs
                )
            
            # Reflection LM (may be same as task LM)
            if model_config.reflection_model == model_config.task_model:
                self.reflection_lm = self.task_lm
            else:
                # Determine API key for reflection model
                if "openrouter/" in model_config.reflection_model:
                    refl_api_key = os.getenv(model_config.openrouter_api_key_env)
                elif "xai/" in model_config.reflection_model:
                    refl_api_key = os.getenv(model_config.xai_api_key_env)
                else:
                    refl_api_key = api_key
                
                if model_config.vllm_endpoint:
                    self.reflection_lm = dspy.LM(
                        model=f"openai/{model_config.reflection_model}",
                        api_base=model_config.vllm_endpoint,
                        api_key="dummy",
                    )
                else:
                    litellm_kwargs = model_config.get_litellm_kwargs(model_config.reflection_model)
                    self.reflection_lm = dspy.LM(
                        model=model_config.reflection_model,
                        api_key=refl_api_key,
                        **litellm_kwargs
                    )
        
        except Exception as e:
            print(f"Warning: Could not initialize LMs: {e}")
            print("Continuing without LM initialization (tests may fail)")
    
    def create_arm(
        self,
        arm_name: str,
        budget_tracker: BudgetTracker
    ) -> BaseArm:
        """Create an arm instance.
        
        Args:
            arm_name: Name of arm to create (e.g., "A", "B", "K-Jev")
            budget_tracker: Budget tracker for this run
            
        Returns:
            Arm instance
        """
        # Find arm config
        arm_config = None
        for ac in self.config.arms + self.config.full_grid_arms:
            if ac.name == arm_name:
                arm_config = ac
                break
        
        if not arm_config:
            raise ValueError(f"Unknown arm: {arm_name}")
        
        # Create arm based on type
        if arm_config.arm_type == "gepa":
            return GEPADefaultArm(
                name=arm_config.name,
                description=arm_config.description,
                budget_tracker=budget_tracker,
                **arm_config.kwargs
            )
        elif arm_config.arm_type == "prose_critic":
            return ProseCriticArm(
                name=arm_config.name,
                description=arm_config.description,
                budget_tracker=budget_tracker,
                **arm_config.kwargs
            )
        elif arm_config.arm_type == "popperian":
            return PopperianArm(
                name=arm_config.name,
                description=arm_config.description,
                budget_tracker=budget_tracker,
                **arm_config.kwargs
            )
        elif arm_config.arm_type in ["bayesian_jev", "bayesian_verbal"]:
            credence_method = "jev" if "jev" in arm_config.arm_type else "verbal"
            return BayesianPopperianArm(
                name=arm_config.name,
                description=arm_config.description,
                budget_tracker=budget_tracker,
                credence_method=credence_method,
                **arm_config.kwargs
            )
        else:
            raise NotImplementedError(f"Arm type not implemented: {arm_config.arm_type}")
    
    def create_benchmark(self, benchmark_name: str) -> BaseBenchmark:
        """Create a benchmark instance.
        
        Args:
            benchmark_name: Name of benchmark (e.g., "hover", "pupa")
            
        Returns:
            Benchmark instance
        """
        if benchmark_name == "hover":
            return HoVerBenchmark()
        elif benchmark_name == "pupa":
            return PUPABenchmark()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    def run_single(
        self,
        benchmark_name: str,
        arm_name: str,
        budget: int,
        seed: int,
        test_size: Optional[int] = None,
        max_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a single experiment configuration.
        
        Args:
            benchmark_name: Benchmark to use
            arm_name: Arm to use
            budget: Total budget (units)
            seed: Random seed
            test_size: Test set size (None = full)
            max_cost: Maximum cost in dollars (None = no limit)
            
        Returns:
            Results dictionary
        """
        print(f"\nRunning: {benchmark_name} | {arm_name} | budget={budget} | seed={seed}")
        
        # Create budget tracker
        budget_tracker = BudgetTracker(
            max_units=budget,
            cost_per_million_input=self.config.model_config.cost_per_million_input,
            cost_per_million_output=self.config.model_config.cost_per_million_output,
        )
        
        # Create budget enforcer if max_cost specified
        enforcer = None
        if max_cost:
            enforcer = BudgetEnforcer(max_cost, budget_tracker)
        
        # Load benchmark
        benchmark = self.create_benchmark(benchmark_name)
        
        # Setup check
        setup_ok, setup_msg = benchmark.setup()
        if not setup_ok:
            print(f"Benchmark setup failed: {setup_msg}")
            return {
                "benchmark": benchmark_name,
                "arm": arm_name,
                "budget": budget,
                "seed": seed,
                "status": "setup_failed",
                "message": setup_msg,
            }
        
        # Load data
        trainset, valset, testset = benchmark.load_data(
            train_size=self.config.train_size,
            val_size=self.config.val_size,
            test_size=test_size,
        )
        
        # Get program and metric
        program = benchmark.get_program()
        metric = benchmark.get_metric()
        
        # Create arm
        arm = self.create_arm(arm_name, budget_tracker)
        
        # Check if we have LMs
        if self.task_lm is None or self.reflection_lm is None:
            return {
                "benchmark": benchmark_name,
                "arm": arm_name,
                "budget": budget,
                "seed": seed,
                "status": "no_lm",
                "message": "No LM configured (missing API keys)",
            }
        
        # Run optimization
        start_time = time.time()
        status = "success"
        message = ""
        
        try:
            # Configure DSPy
            dspy.settings.configure(lm=self.task_lm)
            
            # Compile
            optimized = arm.compile(
                program=program,
                metric=metric,
                trainset=trainset,
                valset=valset,
                critic_lm=self.reflection_lm,
                task_lm=self.task_lm,
            )
            
            # Check cost limit
            if enforcer:
                enforcer.check()
            
        except CostLimitExceeded as e:
            status = "cost_limit_exceeded"
            message = str(e)
            optimized = program
        
        except Exception as e:
            status = "error"
            message = str(e)
            optimized = program
        
        end_time = time.time()
        
        # Evaluate on test set
        test_scores = []
        if status == "success" and optimized:
            for ex in testset:
                try:
                    pred = optimized(question=ex.question)
                    score = metric(ex, pred, None, None, None).score
                    test_scores.append(score)
                except Exception as e:
                    print(f"Test evaluation error: {e}")
                    test_scores.append(0.0)
        
        # Collect results
        results = {
            "benchmark": benchmark_name,
            "arm": arm_name,
            "budget": budget,
            "seed": seed,
            "status": status,
            "message": message,
            "train_size": len(trainset),
            "val_size": len(valset),
            "test_size": len(testset),
            "runtime_seconds": end_time - start_time,
            "budget_stats": budget_tracker.get_stats(),
            "arm_stats": arm.get_stats(),
            "test_scores": test_scores,
            "test_mean": sum(test_scores) / len(test_scores) if test_scores else 0.0,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        filename = (
            f"{results['benchmark']}_"
            f"{results['arm']}_"
            f"b{results['budget']}_"
            f"s{results['seed']}_"
            f"{results['timestamp'].replace(':', '-')}.json"
        )
        
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def run_stage(self, stage: StageConfig) -> List[Dict[str, Any]]:
        """Run all experiments for a stage.
        
        Args:
            stage: Stage configuration
            
        Returns:
            List of results dictionaries
        """
        print(f"\n{'='*60}")
        print(f"Running Stage {stage.stage_id}: {stage.name}")
        print(f"Description: {stage.description}")
        print(f"Cost cap: ${stage.max_cost_dollars}")
        print(f"{'='*60}\n")
        
        all_results = []
        total_cost = 0.0
        
        # Iterate over configurations
        for model in stage.models:
            # Update model config
            self.config.model_config.task_model = model
            self.config.model_config.reflection_model = model
            self._init_lms()
            
            for benchmark_name in stage.benchmarks:
                for arm_name in stage.arms:
                    for budget in stage.budgets:
                        for seed in range(stage.n_seeds):
                            # Check if we've exceeded stage cost cap
                            if total_cost >= stage.max_cost_dollars:
                                print(f"\nStage cost cap reached: ${total_cost:.2f}")
                                return all_results
                            
                            # Run single experiment
                            results = self.run_single(
                                benchmark_name=benchmark_name,
                                arm_name=arm_name,
                                budget=budget,
                                seed=seed,
                                test_size=stage.test_size,
                                max_cost=stage.max_cost_dollars - total_cost,
                            )
                            
                            all_results.append(results)
                            
                            # Update total cost
                            if "budget_stats" in results:
                                total_cost += results["budget_stats"].get("total_cost", 0.0)
                            
                            print(f"Total stage cost so far: ${total_cost:.2f} / ${stage.max_cost_dollars:.2f}")
        
        print(f"\nStage {stage.stage_id} complete. Total cost: ${total_cost:.2f}")
        
        return all_results
