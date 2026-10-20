"""Experiment harness with per-run LM histories and held-out evaluation."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import dspy

from experiments.theory_driven.arms import (
    BaseArm, GEPADefaultArm, PopperianArm, ProseCriticArm, MillianArm,
)
from experiments.theory_driven.benchmarks import (
    BaseBenchmark, HotpotQABenchmark, HoVerBenchmark, PUPABenchmark,
)
from experiments.theory_driven.config import ExperimentConfig, ModelConfig, StageConfig
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.canary_runner import run_stage_canary
from experiments.theory_driven.events.run_trace import RunTrace
from experiments.theory_driven.provider_runtime import ProviderRuntime
from experiments.theory_driven.config.variants import parse_arm_variant
from experiments.theory_driven.config.resume import fingerprint, completed_result, any_result_cost


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
                   seed: int = 0, budget: int | None = None, *,
                   survival_policy: str | None = None,
                   min_survival_questions: int | None = None,
                   held_out_regression_samples: int = 0) -> BaseArm:
        base_name, variant_policy = parse_arm_variant(arm_name)
        survival_policy = variant_policy or survival_policy
        arm_config = next((a for a in self.config.arms + self.config.full_grid_arms
                           if a.name == base_name), None)
        if arm_config is None:
            raise ValueError(f"Unknown arm: {arm_name}")
        classes = {"gepa": GEPADefaultArm, "popperian": PopperianArm,
                   "prose_critic": ProseCriticArm, "millian": MillianArm}
        if arm_config.arm_type not in classes:
            raise NotImplementedError(f"Pilot-only arm: {arm_name}")
        kwargs = dict(arm_config.kwargs)
        if arm_config.arm_type in {"popperian", "millian"}:
            if survival_policy is not None:
                kwargs["survival_policy"] = survival_policy
            if min_survival_questions is not None:
                kwargs["min_survival_questions"] = min_survival_questions
            if held_out_regression_samples > 0:
                kwargs["held_out_regression_samples"] = held_out_regression_samples
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
                   test_size: int | None = None, max_cost: float | None = None,
                   survival_policy: str | None = None,
                   min_survival_questions: int | None = None,
                   held_out_regression_samples: int = 0,
                   config_hash: str | None = None,
                   max_retries: int = 10,
                   backoff_seconds: float = 2.0,
                   max_delay_seconds: float = 120.0,
                   max_retry_wall_seconds: float = 900.0) -> dict[str, Any]:
        runtime = ProviderRuntime(
            full_prompt_log=True,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            max_delay_seconds=max_delay_seconds,
            max_retry_wall_seconds=max_retry_wall_seconds,
        )
        base_name, variant_policy = parse_arm_variant(arm_name)
        survival_policy = variant_policy or survival_policy
        arm_config = next((a for a in self.config.arms + self.config.full_grid_arms
                           if a.name == base_name), None)
        options = arm_config.kwargs if arm_config is not None else {}
        if survival_policy is None:
            survival_policy = options.get("survival_policy", "strict")
        if min_survival_questions is None:
            min_survival_questions = options.get("min_survival_questions", 3)
        policy_name = getattr(survival_policy, "name", survival_policy)
        trace = RunTrace(self.results_dir, self.config.name, benchmark_name, arm_name,
                         {"budget": budget, "seed": seed,
                          "survival_policy": policy_name,
                          "min_survival_questions": min_survival_questions}, sanitizer=lambda text: runtime.sanitize(text, truncate=False))
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
            if arm_config is not None and arm_config.arm_type == "millian":
                configured_defender = arm_config.kwargs.get("defender_lm")
                if configured_defender is not None and "defender" not in self.lms:
                    # Each run gets a fresh guarded copy, included before budget creation.
                    self.lms["defender"] = configured_defender.copy()
            if not {"task", "reflection", "critic"} <= set(self.lms) or set(self.lms) - {"task", "reflection", "critic", "defender"}:
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
            arm = self.create_arm(arm_name, tracker, seed=seed, budget=budget,
                                  survival_policy=survival_policy, min_survival_questions=min_survival_questions,
                                  held_out_regression_samples=held_out_regression_samples)
            with dspy.context(lm=self.lms["task"]):
                optimized = arm.compile(program=program, metric=metric, trainset=train,
                                        valset=val, task_lm=self.lms["task"], event_sink=trace.proposal,
                                        reflection_lm=self.lms["reflection" if arm_config and arm_config.arm_type == "gepa" else "critic"],
                                        **({"defender_lm": self.lms.get("defender")} if arm_config and arm_config.arm_type == "millian" else {}))
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
                  "arm_base": base_name, "config_hash": config_hash,
                  "status": status, "message": message, "train_size": len(train), "val_size": len(val),
                  "test_size": len(test), "test_scores": scores,
                  "test_mean": sum(scores) / len(scores) if scores and status == "success" else None,
                  "budget_stats": stats, "arm_stats": trace._sanitize(arm.get_stats()) if arm else {},
                  "artifacts": {"events": artifacts["epistemic_events"], "lm_calls": call_path},
                  "canary": self.canary,
                  "survival_policy": getattr(arm, "survival_policy", None),
                  "min_survival_questions": getattr(arm, "min_survival_questions", None),
                  "runtime_seconds": time.monotonic() - started,
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        self._save_results(result)
        return result

    def _save_results(self, result: dict[str, Any]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        name = (f"{result['benchmark']}_{result['arm']}_b{result['budget']}_s{result['seed']}_"
                f"{result['timestamp'].replace(':', '-')}.json")
        (self.results_dir / name).write_text(json.dumps(result, indent=2))

    def _canary_lms(self, config: ModelConfig, seed: int, stage: StageConfig):
        runtime = ProviderRuntime(
            max_retries=stage.max_retries,
            backoff_seconds=stage.backoff_seconds,
            max_delay_seconds=stage.max_delay_seconds,
            max_retry_wall_seconds=stage.max_retry_wall_seconds,
        )
        return {role: runtime.wrap(lm, "canary" if role == "task" else role)
                for role, lm in self.lm_factory(config, seed).items()}

    def run_stage(self, stage: StageConfig, resume_dir: str | Path | None = None) -> list[dict[str, Any]]:
        results = []
        self.canary = run_stage_canary(self.config, stage, lambda cfg, s: self._canary_lms(cfg, s, stage))
        if not self.canary["passed"]:
            raise RuntimeError(f"Stage canary failed: {self.canary['message']}")
        cost = self.canary["total_cost"]
        # Discover the entire stage first: count all runs (including aborted ones)
        # to ensure cumulative spend is tracked correctly.
        runs = []
        for model in stage.models:
            self.config.model_config.task_model = model
            self.config.model_config.reflection_model = model
            for benchmark in stage.benchmarks:
                for arm in stage.arms:
                    for budget in stage.budgets:
                        for seed in range(stage.n_seeds):
                            config_hash = fingerprint(self.config, stage, benchmark, arm, budget, seed)
                            previous = completed_result(Path(resume_dir), benchmark, arm, budget, seed,
                                                        config_hash, stage.test_size) if resume_dir else None
                            runs.append((model, benchmark, arm, budget, seed, config_hash, previous))
                            if resume_dir:
                                # Count all costs, including from aborted runs
                                run_cost = any_result_cost(Path(resume_dir), benchmark, arm, budget, seed, config_hash)
                                cost += run_cost
        for model, benchmark, arm, budget, seed, config_hash, previous in runs:
            self.config.model_config.task_model = model
            self.config.model_config.reflection_model = model
            if previous is not None:
                print(f"Resume: skipping {benchmark} | {arm} | budget={budget} | seed={seed}")
                results.append(previous)
                continue
            if cost >= stage.max_cost_dollars:
                raise RuntimeError("Stage dollar cap reached before all runs completed")
            result = self.run_single(benchmark, arm, budget, seed,
                                     test_size=stage.test_size,
                                     max_cost=stage.max_cost_dollars - cost,
                                     survival_policy=stage.survival_policy,
                                     min_survival_questions=stage.min_survival_questions,
                                     held_out_regression_samples=stage.held_out_regression_samples,
                                     config_hash=config_hash,
                                     max_retries=stage.max_retries,
                                     backoff_seconds=stage.backoff_seconds,
                                     max_delay_seconds=stage.max_delay_seconds,
                                     max_retry_wall_seconds=stage.max_retry_wall_seconds)
            results.append(result)
            cost += result["budget_stats"]["total_cost"]
            if cost > stage.max_cost_dollars:
                raise RuntimeError("Stage dollar cap exceeded")
            if result["status"] != "success":
                raise RuntimeError(f"Arm {arm} aborted: {result['message']}")
        if cost > stage.max_cost_dollars:
            raise RuntimeError("Stage dollar cap exceeded")
        return results
