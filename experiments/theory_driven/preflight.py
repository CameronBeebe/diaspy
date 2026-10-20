"""Offline stage-zero gate: static checks, socket-blocked tests, call plan."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.theory_driven.config import ExperimentConfig, StageConfig, CanarySuite
from experiments.theory_driven.config.variants import parse_arm_variant


class PreFlightGate:
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir.resolve()
        self.repo_root = self.experiment_dir.parents[1]
        self.results: dict[str, Any] = {}

    def _run(self, name: str, args: list[str]) -> bool:
        env = {key: value for key, value in os.environ.items()
               if not key.endswith("API_KEY") and not key.endswith("API_TOKEN")}
        env.update(LITELLM_LOCAL_MODEL_COST_MAP="True", HF_HUB_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1")
        result = subprocess.run([sys.executable, "-m", *args], cwd=self.repo_root,
                                env=env, capture_output=True, text=True, timeout=180)
        print(result.stdout, end="")
        print(result.stderr, end="")
        passed = result.returncode == 0
        self.results[name] = {"passed": passed, "returncode": result.returncode}
        return passed

    def _compute_planned_calls(self, stage: StageConfig, config: ExperimentConfig) -> dict[str, Any]:
        """Report shared caps/ranges, not invented exact adaptive role counts.

        Hotpot has one answer LM call per example; retrieval is local BM25.
        GEPA's adaptive optimization allocation cannot be known before execution.
        Test evaluation is a separate allocation and is included in actual totals.
        """
        multiplier = len(stage.models) * len(stage.benchmarks) * stage.n_seeds
        runs = multiplier * len(stage.arms) * len(stage.budgets)
        optimization = multiplier * len(stage.arms) * sum(stage.budgets)
        reflection = multiplier * sum(parse_arm_variant(a)[0] == "A" for a in stage.arms) * sum(stage.budgets)
        critic = multiplier * sum(parse_arm_variant(arm)[0] in {"B", "C", "M"} for arm in stage.arms) * sum(stage.budgets)
        test_calls = runs * stage.test_size
        canary_calls = len(CanarySuite.DEFAULT_CANARIES) if stage.run_canary else 0
        # Planning assumption, not a provider quote or hard billing bound: each call
        # uses <=8k input/2k output tokens at the configured planning token prices.
        # Allow a conservative 100-call final-iteration overshoot per run.
        planned_calls = optimization + test_calls + canary_calls + runs * 100
        cost_per_call = (8000 * config.model_config.cost_per_million_input
                         + 2000 * config.model_config.cost_per_million_output) / 1_000_000
        return {
            "estimated_worst_case_cost_usd": round(planned_calls * cost_per_call, 6),
            "estimate_assumptions": {"input_tokens_per_call": 8000, "output_tokens_per_call": 2000,
                                     "iteration_overshoot_calls_per_run": 100,
                                     "pricing": "configured planning rates; not a live quote",
                                     "not_a_hard_bound": True},
            "total_runs": runs,
            "optimization_call_cap": optimization,
            "proposer_calls_per_attempt": {a: {"A": 1, "B": 2, "C": 2, "M": 4}.get(parse_arm_variant(a)[0])
                                           for a in stage.arms},
            "per_variant_policies": {
                a: parse_arm_variant(a)[1] or stage.survival_policy
                for a in stage.arms
                if not a.startswith("A")  # Arm A has no proposer/evaluator, skip policy
            },
            "min_survival_questions": stage.min_survival_questions,
            "held_out_regression_samples": stage.held_out_regression_samples,
            "test_calls": test_calls,
            "canary_calls": canary_calls,
            "roles": {
                "canary": {"calls": canary_calls},
                "task": {"optimization_range": [0, optimization], "test": test_calls},
                "reflection": {"optimization_range": [0, reflection], "test": 0},
                "critic": {"optimization_range": [0, critic], "test": 0},
            },
            "max_cost": 0.0 if all(m.endswith(":free") for m in stage.models)
                        and (not stage.run_canary or (stage.canary_model or stage.models[0]).endswith(":free")) else None,
            "cost_cap": stage.max_cost_dollars,
            "note": "Role ranges share the optimization cap; they are not additive. "
                    "GEPA may overshoot by one iteration, reported separately. "
                    "Held-out evaluation adds task calls after optimization. "
                    "Canary calls run before the arm loop; transient retries add provider attempts.",
        }

    def run_dry_run(self, stage_id: int, config: ExperimentConfig | None = None):
        config = config or ExperimentConfig()
        stage = config.get_stage_config(stage_id)
        plan = self._compute_planned_calls(stage, config)
        print(f"Stage {stage_id}: {stage.models}, {stage.benchmarks}, arms {stage.arms}")
        print("Planned calls per role (adaptive ranges):")
        print(json.dumps(plan, indent=2))
        if plan["max_cost"] == 0:
            print("Model cost: $0.00 (free model); no LLM calls made by dry-run.")
        else:
            print("Model cost: unknown without usage/prices; no flat per-call estimate.")
        self.results["dry_run"] = {"passed": True, "plan": plan}
        return True, plan

    def run_all_checks(self, stage_id: int, require_confirm: bool = False) -> bool:
        if stage_id != 0:
            raise ValueError("Only stage 0 is supported by this gate")
        target = "experiments/theory_driven"
        checks = [
            self._run("ruff", ["ruff", "check", target]),
            self._run("pyright", ["pyright", target]),
            self._run("offline_tests", ["pytest", f"{target}/tests", "--disable-socket", "-q"]),
        ]
        checks.append(self.run_dry_run(stage_id)[0])
        return all(checks)

    def save_results(self, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(self.results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True, choices=[0])
    args = parser.parse_args()
    passed = PreFlightGate(Path(__file__).parent).run_all_checks(args.stage)
    print("PRE-FLIGHT GATE: PASSED" if passed else "PRE-FLIGHT GATE: FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
