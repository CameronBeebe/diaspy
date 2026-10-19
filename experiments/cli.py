#!/usr/bin/env python3
"""CLI for running experiments."""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import ExperimentConfig, BudgetConfig
from experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run theory-driven GEPA experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (all arms, tiny scale, DummyLM)
  python3 experiments/cli.py --smoke-test
  
  # Single arm test
  python3 experiments/cli.py --arms B --n 10 --seeds 0 --dummy
  
  # Full grid (requires real models)
  python3 experiments/cli.py --arms A0 A E C B F --n 10 25 50 100 --seeds 0 1 2 3 4 \\
      --task-model openrouter/qwen/qwen3-8b \\
      --reflection-model openrouter/qwen/qwen3-8b
"""
    )
    
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["A0", "A", "E", "C", "B", "F"],
        choices=["A0", "A", "E", "C", "B", "D", "F"],
        help="Arms to run"
    )
    
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="Number of labels"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(5)),
        help="Seeds to use"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory"
    )
    
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use DummyLM (offline, free)"
    )
    
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick smoke test: all arms, N=10, seed=0, tiny budget"
    )
    
    parser.add_argument(
        "--task-model",
        type=str,
        help="Task LM (e.g., openrouter/qwen/qwen3-8b). Required for real runs."
    )
    
    parser.add_argument(
        "--reflection-model",
        type=str,
        help="Reflection LM. Required for real runs."
    )
    
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=1000,
        help="Max program rollouts per run"
    )
    
    parser.add_argument(
        "--max-reflection-calls",
        type=int,
        default=60,
        help="Max reflection LM calls per run"
    )
    
    parser.add_argument(
        "--max-reflection-tokens",
        type=int,
        default=300_000,
        help="Max reflection tokens per run"
    )
    
    args = parser.parse_args()
    
    # Smoke test overrides
    if args.smoke_test:
        print("Running smoke test...")
        args.arms = ["A0", "A", "E", "C", "B", "F"]
        args.n = [10]
        args.seeds = [0]
        args.dummy = True
        args.max_rollouts = 50
        args.max_reflection_calls = 10
        args.max_reflection_tokens = 50_000
    
    # Check model requirements
    if not args.dummy and (not args.task_model or not args.reflection_model):
        print("ERROR: Real runs require --task-model and --reflection-model", file=sys.stderr)
        print("Or use --dummy for offline testing with DummyLM", file=sys.stderr)
        return 1
    
    # Build config
    config = ExperimentConfig(
        arms=args.arms,
        n_labels=args.n,
        seeds=args.seeds,
        results_dir=args.results_dir,
        task_model=args.task_model,
        reflection_model=args.reflection_model,
        budget=BudgetConfig(
            max_program_rollouts=args.max_rollouts,
            max_reflection_calls=args.max_reflection_calls,
            max_reflection_tokens=args.max_reflection_tokens
        )
    )
    
    print(f"Configuration:")
    print(f"  Arms: {', '.join(config.arms)}")
    print(f"  N: {config.n_labels}")
    print(f"  Seeds: {config.seeds}")
    print(f"  Budget: {config.budget.max_program_rollouts} rollouts, "
          f"{config.budget.max_reflection_calls} reflection calls")
    print(f"  Mode: {'DummyLM (offline)' if args.dummy else 'Real models'}")
    print(f"  Results: {config.results_dir}\n")
    
    # Confirm for real runs
    if not args.dummy and not args.smoke_test:
        resp = input("This will make real API calls. Continue? [y/N] ")
        if resp.lower() != 'y':
            print("Aborted.")
            return 0
    
    # Run
    runner = ExperimentRunner(config, use_dummy_lm=args.dummy)
    runner.run_grid()
    
    print(f"\n✓ Experiment complete. Results in {config.results_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Explore results: jupyter notebook notebooks/explore_results.ipynb")
    print(f"  2. View in MLflow: python3 experiments/mlflow_utils.py export {config.results_dir}")
    print(f"                     python3 experiments/mlflow_utils.py ui")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
