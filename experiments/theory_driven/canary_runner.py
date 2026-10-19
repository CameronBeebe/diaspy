"""Reusable live-stage canary gate, independent of the experiment harness."""
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from experiments.theory_driven.config.canary import CanarySuite, safe_canary_error

from experiments.theory_driven.config import ExperimentConfig, StageConfig
from experiments.theory_driven.provider_runtime import ProviderRuntime


def _sanitize_payload(value: Any, sanitizer: ProviderRuntime) -> Any:
    """Redact complete strings before any drift diagnostic truncates them."""
    if isinstance(value, str):
        return sanitizer.sanitize(value)
    if isinstance(value, dict):
        return {sanitizer.sanitize(key): _sanitize_payload(item, sanitizer)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item, sanitizer) for item in value]
    return value


def run_stage_canary(
    config: ExperimentConfig, stage: StageConfig, lm_factory: Callable
) -> dict[str, Any]:
    """Run once before the arm loop; callers must block when ``passed`` is false.

    ``lm_factory(ModelConfig, seed)`` supplies fresh role LMs; only ``task``
    is called. The canary model overrides task/reflection models on a copy.
    Artifact paths are relative to ``config.results_dir``. Provider errors
    always block; ``allow_canary_drift`` only relaxes reference comparison.
    The first successful run establishes a labelled baseline, not a drift
    validation. Call only for live runs; disabled stages create no artifacts.
    """
    if not stage.run_canary:
        return {"passed": True, "status": "skipped", "message": "Canary disabled",
                "artifacts": {}, "total_cost": 0.0}
    root = Path(config.results_dir)
    directory = root / "canaries"
    suite = CanarySuite(directory / "references")
    model = stage.canary_model or stage.models[0]
    model_config = replace(config.model_config, task_model=model, reflection_model=model)
    lm = None
    results = []
    reference = {}
    sanitizer = ProviderRuntime()
    try:
        reference = suite.load_reference(model)
        lm = lm_factory(model_config, 42)["task"]
        sanitizer = getattr(lm, "runtime", None) or sanitizer
        results = suite.run_canaries(lm, model, temperature=0.0, seed=42)
        error = next((r.error for r in results if r.error), None)
    except Exception as exc:
        error = safe_canary_error(exc)
    # Sanitize historical references before check_drift slices expected excerpts;
    # afterwards a partial credential can no longer be matched for redaction.
    safe_reference = _sanitize_payload(reference, sanitizer)
    if safe_reference != reference:
        suite.get_reference_file(model).write_text(json.dumps(safe_reference, indent=2))
    reference = safe_reference
    # Hashes were computed on raw output and remain the drift comparison keys.
    results = [replace(entry, **_sanitize_payload(entry.to_dict(), sanitizer)) for entry in results]
    suite.results = results
    issues = []
    passed, status, message = False, "error", error
    if not error and not reference.get("prompts"):
        suite.save_reference(model, results)
        passed, status, message = True, "baseline", "Canary baseline created (no prior reference)"
    elif not error:
        drift, issues = suite.check_drift(results, reference)
        passed = not drift or stage.allow_canary_drift
        status = "drift_allowed" if drift and passed else "drift" if drift else "passed"
        message = "Canary drift allowed" if drift and passed else "Canary drift detected" if drift else "Canary matches reference"
    run_id = f"stage_{stage.stage_id}_{uuid4().hex}"
    results_path = directory / f"{run_id}_results.json"
    summary_path = directory / f"{run_id}_summary.json"
    suite.save_results(results_path)
    result = {
        "passed": passed, "status": status, "message": message, "issues": issues,
        "model": model,
        "total_cost": sum(float(call.get("cost") or 0.0) for call in lm.history) if lm is not None else 0.0,
        "artifacts": {
            "results": str(results_path.relative_to(root)),
            "summary": str(summary_path.relative_to(root)),
        },
    }
    if suite.get_reference_file(model).exists():
        result["artifacts"]["reference"] = str(suite.get_reference_file(model).relative_to(root))
    runtime = getattr(lm, "runtime", None)
    if runtime is not None:
        calls_path = directory / f"{run_id}_lm_calls.jsonl"
        runtime.write_jsonl(calls_path)
        result["artifacts"]["lm_calls"] = str(calls_path.relative_to(root))
    result = _sanitize_payload(result, sanitizer)
    summary_path.write_text(json.dumps(result, indent=2))
    return result
