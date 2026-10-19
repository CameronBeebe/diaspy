"""Stage-zero entry point; pilot stages are intentionally unsupported."""
import argparse
import json
import os
from pathlib import Path

from experiments.theory_driven.config import ExperimentConfig, StageConfig
from experiments.theory_driven.harness import ExperimentHarness
from experiments.theory_driven.preflight import PreFlightGate
from experiments.theory_driven.provider_runtime import ProviderRuntime
from experiments.theory_driven.config.variants import parse_arm_variant


def run_stage0(config: ExperimentConfig, *, harness: ExperimentHarness | None = None,
               stage: StageConfig | None = None, resume_dir: str | Path | None = None) -> list[dict]:
    """Shared runtime path for injected offline LMs and real stage-zero execution."""
    stage = stage or config.get_stage_config(0)
    if stage.stage_id != 0 or {parse_arm_variant(a)[0] for a in stage.arms} != {"A", "B", "M"}:
        raise ValueError("Stage 0 requires A, B and M (policy variants allowed)")
    if len(stage.arms) != len(set(stage.arms)):
        raise ValueError("Stage arm variants must be unique")
    results = (harness or ExperimentHarness(config)).run_stage(stage, resume_dir=resume_dir)
    expected = (len(stage.models) * len(stage.benchmarks) * len(stage.arms)
                * len(stage.budgets) * stage.n_seeds)
    if len(results) != expected or any(r["status"] != "success" for r in results):
        raise RuntimeError(f"Stage 0 did not complete successfully: {results}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume completed matching arms from this results directory")
    parser.add_argument("--confirm", action="store_true", help="Explicitly allow model/network execution")
    args = parser.parse_args(argv)
    config = ExperimentConfig()
    config.model_config.provider_allow_fallbacks = os.getenv("OPENROUTER_ALLOW_FALLBACKS", "false").lower() == "true"
    config.model_config.provider_order = [p.strip() for p in os.getenv("OPENROUTER_PROVIDER_ORDER", "").split(",") if p.strip()]
    if args.resume is not None:
        config.results_dir = str(args.resume)
    gate = PreFlightGate(Path(__file__).parent)
    if args.dry_run:
        gate.run_dry_run(args.stage, config)
        return 0
    if not args.confirm:
        parser.error("Model execution requires --confirm; use --dry-run for offline planning")
    if not gate.run_all_checks(args.stage):
        return 1
    try:
        results = run_stage0(config, resume_dir=args.resume) if args.resume else run_stage0(config)
    except Exception as error:
        print(ProviderRuntime().sanitize(f"Stage 0 failed: {type(error).__name__}: {error}"))
        return 1
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
