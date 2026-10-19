"""Experiment harness with per-run LM histories and held-out evaluation."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import dspy

from experiments.theory_driven.arms import (
    BaseArm, GEPADefaultArm, PopperianArm, ProseCriticArm,
)
from experiments.theory_driven.benchmarks import (
    BaseBenchmark, HotpotQABenchmark, HoVerBenchmark, PUPABenchmark,
)
from experiments.theory_driven.config import ExperimentConfig, ModelConfig, StageConfig
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.canary_runner import run_stage_canary
from experiments.theory_driven.events.run_trace import RunTrace
from experiments.theory_driven.provider_runtime import ProviderRuntime


def create_lms(config: ModelConfig, seed: int) -> dict[str, Any]:
    """Separate instances even for identical models: unambiguous role accounting."""
    def build(model: str):
        key_env = (config.openrouter_api_key_env if model.startswith("openrouter/")
                   else config.xai_api_key_env if model.startswith("xai/")
                   else config.litellm_api_key_env)
        key = os.getenv(key_env)
        if not key and not config.vllm_endpoint:
            raise ValueError(f"Missing credential: {key_env}")
        kwargs = config.get_litellm_kwargs(model)
        if config.vllm_endpoint:
            kwargs["api_base"] = config.vllm_endpoint
            model = f"openai/{model}"
        return dspy.LM(model=model, api_key=key or "dummy", cache=False, seed=seed, num_retries=0, **kwargs)
    return {"task": build(config.task_model),
            "reflection": build(config.reflection_model),
            "critic": build(config.reflection_model)}


class GuardedBudget(UnifiedBudget):
    def __init__(self, runtime: ProviderRuntime, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime = runtime

    def __call__(self, state) -> bool:
        return self.runtime.should_stop(state) or super().__call__(state)


class ExperimentHarness:
    def __init__(self, config: ExperimentConfig, *,
                 lm_factory: Callable = create_lms,
                 benchmark_factory: Callable | None = None):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.lm_factory = lm_factory
        self.benchmark_factory = benchmark_factory
        self.lms: dict[str, Any] = {}
        self.canary: dict[str, Any] | None = None

    def create_arm(self, arm_name: str, budget_tracker: UnifiedBudget,
                   seed: int = 0, budget: int | None = None) -> BaseArm:
        arm_config = next((a for a in self.config.arms + self.config.full_grid_arms
                           if a.name == arm_name), None)
        if arm_config is None:
            raise ValueError(f"Unknown arm: {arm_name}")
        classes = {"gepa": GEPADefaultArm, "popperian": PopperianArm,
                   "prose_critic": ProseCriticArm}
        if arm_config.arm_type not in classes:
            raise NotImplementedError(f"Pilot-only arm: {arm_name}")
        kwargs = dict(arm_config.kwargs)
        kwargs.update(seed=seed, max_metric_calls=budget if budget is not None else budget_tracker.max_units)
        return classes[arm_config.arm_type](name=arm_name, description=arm_config.description,
                                           budget_tracker=budget_tracker, **kwargs)

    def create_benchmark(self, benchmark_name: str) -> BaseBenchmark:
        if self.benchmark_factory is not None:
            return self.benchmark_factory(benchmark_name)
        classes = {"hotpotqa": HotpotQABenchmark, "hover": HoVerBenchmark, "pupa": PUPABenchmark}
        if benchmark_name not in classes:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        config = next((b for b in self.config.benchmarks if b.name == benchmark_name), None)
        kwargs = config.extra_kwargs if config else {}
        return classes[benchmark_name](name=benchmark_name, **kwargs)

    def run_single(self, benchmark_name: str, arm_name: str, budget: int, seed: int,
                   test_size: int | None = None, max_cost: float | None = None) -> dict[str, Any]:
        runtime = ProviderRuntime()
        trace = RunTrace(self.results_dir, self.config.name, benchmark_name, arm_name,
                         {"budget": budget, "seed": seed}, sanitizer=runtime.sanitize)
        call_path = f"lm_calls/{trace.run_id}_{arm_name}.jsonl"
        cost_cap = float("inf") if max_cost is None else max_cost
        tracker = GuardedBudget(runtime, budget, cost_cap)
        arm = None
        train, val, test = [], [], []
        self.lms = {}
        started = time.monotonic()
        status, message = "success", ""
        scores: list[float] = []
        optimization_calls = 0
        optimization_cost = 0.0
        optimization_complete = False
        try:
            # Factory creates fresh histories per seed/run; no cache or previous-run accumulation.
            self.lms = self.lm_factory(self.config.model_config, seed)
            if set(self.lms) != {"task", "reflection", "critic"}:
                raise ValueError("LM factory must return task, reflection and critic roles")
            if len({id(lm) for lm in self.lms.values()}) != len(self.lms):
                raise ValueError("Use distinct LM instances for truthful per-role accounting")
            if any(lm.history for lm in self.lms.values()):
                raise ValueError("Each run requires fresh LM histories")
            cost_cap = float("inf") if max_cost is None else max_cost
            self.lms = {role: runtime.wrap(lm, role) for role, lm in self.lms.items()}
            tracker = GuardedBudget(runtime, budget, cost_cap, *self.lms.values(), roles=self.lms)
            benchmark = self.create_benchmark(benchmark_name)
            ok, message = benchmark.setup()
            if not ok:
                raise RuntimeError(message)
            train, val, test = benchmark.load_data(seed=seed, train_size=self.config.train_size,
                                                   val_size=self.config.val_size,
                                                   test_size=self.config.test_size if test_size is None else test_size)
            program, metric = benchmark.get_program(), benchmark.get_metric()
            arm = self.create_arm(arm_name, tracker, seed=seed, budget=budget)
            with dspy.context(lm=self.lms["task"]):
                optimized = arm.compile(program=program, metric=metric, trainset=train,
                                        valset=val, task_lm=self.lms["task"], event_sink=trace.proposal,
                                        reflection_lm=self.lms["reflection" if arm_name == "A" else "critic"])
                runtime.raise_if_failed()
                optimization_calls, optimization_cost = tracker.used_calls(), tracker.used_usd()
                optimization_complete = True
                # Held-out scoring has its own call allocation, but shares histories and dollar cap.
                for example in test:
                    if tracker.used_usd() >= cost_cap:
                        status, message = "cost_limit_exceeded", "Cost cap reached before held-out evaluation completed"
                        break
                    prediction = optimized(**example.inputs())
                    runtime.raise_if_failed()
                    scores.append(float(metric(example, prediction, None, None, None).score))
                    if tracker.used_usd() > cost_cap:
                        status, message = "cost_limit_exceeded", "Held-out evaluation exceeded the dollar cap"
                        break
        except Exception as error:
            # Persist the failure; never mark a fallback/base-program run as success.
            status, message = "error", runtime.sanitize(f"{type(error).__name__}: {error}")
            runtime.attempts.append({"role": "harness", "error": message, "kind": "run_exception"})
            if not optimization_complete:
                optimization_calls, optimization_cost = tracker.used_calls(), tracker.used_usd()
        stats = tracker.get_stats()
        stats.update(optimization_calls=optimization_calls, optimization_cost=optimization_cost,
                     optimization_overshoot=max(0, optimization_calls - budget),
                     test_calls=tracker.used_calls() - optimization_calls,
                     cost_overshoot=max(0.0, tracker.used_usd() - cost_cap),
                     provider_failed=runtime.failed)
        runtime.write_jsonl(self.results_dir / call_path)
        artifacts = trace.finalize({"status": status, "message": message, "budget_stats": stats})
        result = {"benchmark": benchmark_name, "arm": arm_name, "budget": budget, "seed": seed,
                  "status": status, "message": message, "train_size": len(train), "val_size": len(val),
                  "test_size": len(test), "test_scores": scores,
                  "test_mean": sum(scores) / len(scores) if scores and status == "success" else None,
                  "budget_stats": stats, "arm_stats": trace._sanitize(arm.get_stats()) if arm else {},
                  "artifacts": {"events": artifacts["epistemic_events"], "lm_calls": call_path},
                  "canary": self.canary,
                  "runtime_seconds": time.monotonic() - started,
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        self._save_results(result)
        return result

    def _save_results(self, result: dict[str, Any]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        name = (f"{result['benchmark']}_{result['arm']}_b{result['budget']}_s{result['seed']}_"
                f"{result['timestamp'].replace(':', '-')}.json")
        (self.results_dir / name).write_text(json.dumps(result, indent=2))

    def _canary_lms(self, config: ModelConfig, seed: int):
        runtime = ProviderRuntime()
        return {role: runtime.wrap(lm, "canary" if role == "task" else role)
                for role, lm in self.lm_factory(config, seed).items()}

    def run_stage(self, stage: StageConfig) -> list[dict[str, Any]]:
        results = []
        self.canary = run_stage_canary(self.config, stage, self._canary_lms)
        if not self.canary["passed"]:
            raise RuntimeError(f"Stage canary failed: {self.canary['message']}")
        cost = self.canary["total_cost"]
        for model in stage.models:
            self.config.model_config.task_model = model
            self.config.model_config.reflection_model = model
            for benchmark in stage.benchmarks:
                for arm in stage.arms:
                    for budget in stage.budgets:
                        for seed in range(stage.n_seeds):
                            if cost >= stage.max_cost_dollars:
                                raise RuntimeError("Stage dollar cap reached before all runs completed")
                            result = self.run_single(benchmark, arm, budget, seed,
                                                     test_size=stage.test_size,
                                                     max_cost=stage.max_cost_dollars - cost)
                            results.append(result)
                            if result["status"] != "success":
                                raise RuntimeError(f"Arm {arm} aborted: {result['message']}")
                            cost += result["budget_stats"]["total_cost"]
        return results
