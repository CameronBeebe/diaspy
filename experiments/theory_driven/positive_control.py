"""Synthetic proposer-level positive control, not a benchmark or efficacy study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Importing DSPy must not fetch a remote pricing map, even for the no-call plan.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import dspy

from experiments.theory_driven.arms.millian import MillianProposer
from experiments.theory_driven.arms.popperian import PopperianProposer
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.config.experiment_config import ModelConfig
from experiments.theory_driven.harness import create_lms
from experiments.theory_driven.provider_runtime import ProviderFailure, ProviderRuntime


CONTRACT = (
    "Read the supplied synthetic JSON record. Return one field value as the answer, "
    "with exact spelling and no explanation or punctuation. "
)
CLEAN = CONTRACT + "Return the requested_field value from record."
DEFECTIVE = CONTRACT + "Ignore requested_field and always return the city value from record."
SOURCE = "synthetic/manual-positive-control-v1 (not real-world measurements)"
SCOPE = "proposer-level; no GEPA compile or final-selection claim"
CELL_CALL_CAP = 20
SOFT_USD_CAP = 0.25
SETTINGS = {
    "max_provider_calls": 80, "max_calls_per_cell": CELL_CALL_CAP, "max_retries": 0,
    "max_rewrite_attempts": 1, "task_max_tokens": 128, "reflection_max_tokens": 1024,
    "seed": 0, "temperature": 0, "n": 1, "cache": False,
    "soft_cost_usd": SOFT_USD_CAP, "cost_is_hard_cap": False,
    "survival_policy": "strict", "min_survival_questions": 3,
    "data_split": {"reflection": 3, "hidden": 3},
    "warning": "Soft stop uses reported costs; missing/delayed costs and the final call may exceed $0.25.",
    "claim": "Diagnostic only; no efficacy or generalization claim.",
}


class DiagnosticRuntime(ProviderRuntime):
    """Thin pre/post-call limits; provider requests/retry/logging remain in ProviderRuntime."""

    def __init__(self):
        super().__init__(full_prompt_log=True, max_retries=0,
                         timeout_seconds=30, max_retry_wall_seconds=40)

    def call(self, lm, role, prompt, messages, kwargs):
        with self._lock:
            self.raise_if_failed()
            cell = role.rsplit(":", 1)[0]
            used = sum(row["role"].rsplit(":", 1)[0] == cell for row in self.attempts)
            if len(self.attempts) >= 80 or used >= CELL_CALL_CAP:
                self.failure = ProviderFailure("diagnostic call cap reached")
            if sum(row.get("cost") or 0 for row in self.attempts) >= SOFT_USD_CAP:
                self.failure = ProviderFailure("diagnostic soft cost cap reached; not a hard billing cap")
            self.raise_if_failed()
            response = super().call(lm, role, prompt, messages, kwargs)
            if sum(row.get("cost") or 0 for row in self.attempts) >= SOFT_USD_CAP:
                self.failure = ProviderFailure("diagnostic soft cost cap reached; not a hard billing cap")
            self.raise_if_failed()
            return response


def evaluate(program, examples, lm):
    rows = []
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        for example in examples:
            prediction = program(**example.inputs())
            metric = exact_metric(example, prediction)
            rows.append({"Inputs": dict(example.inputs()),
                         "Generated Outputs": {"answer": prediction.answer},
                         "Feedback": metric.feedback, "score": metric.score})
    return {"rows": rows, "scores": [row["score"] for row in rows]}


def run_diagnostic(output: Path, *, lm_factory, model="scripted-offline-test-double"):
    """One fresh actual B/M proposer per cell; injected LMs enable socket-blocked tests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = DiagnosticRuntime()
    result = {"source": SOURCE, "scope": SCOPE, "model": model,
              "settings": SETTINGS, "status": "running", "cells": [], "calls": 0}
    train, hidden = examples()
    start = 0
    try:
        for arm in ("B", "M"):
            for condition, instruction in (("defective", DEFECTIVE), ("clean", CLEAN)):
                start = len(runtime.attempts)
                cell = {"arm": arm, "condition": condition, "original_instruction": instruction,
                        "proposal_events": [], "status": "running"}
                result["cells"].append(cell)
                lms = {role: runtime.wrap(lm, f"{arm}:{condition}:{role}")
                       for role, lm in lm_factory(arm, condition).items()}
                budget = UnifiedBudget(CELL_CALL_CAP, SOFT_USD_CAP, roles=lms)
                program = RecordQA(instruction)
                observed = evaluate(program, train, lms["task"])
                cell["reflective_batch"] = observed["rows"]
                cls = PopperianProposer if arm == "B" else MillianProposer
                options: dict[str, Any] = {"defender_lm": lms["reflection"]} if arm == "M" else {}
                proposer = cls(
                    program, exact_metric, {ex.question: ex for ex in train},
                    lms["critic"], lms["task"], budget, max_attempts=1,
                    survival_policy="strict", min_survival_questions=3,
                    event_sink=cell["proposal_events"].append,
                    **options,
                )
                with dspy.context(adapter=dspy.ChatAdapter()):
                    updates = proposer({"answer": instruction}, {"answer": observed["rows"]}, ["answer"])
                proposer.structured_adapter.raise_if_failed()
                runtime.raise_if_failed()
                selected = updates.get("answer", instruction)
                cell["selected_instruction"] = selected
                cell["instruction_changed"] = selected != instruction
                # Hidden records are first used only after the proposal is frozen.
                cell["parent_hidden"] = evaluate(program, hidden, lms["task"])
                cell["selected_hidden"] = evaluate(RecordQA(selected), hidden, lms["task"])
                parent_scores = cell["parent_hidden"]["scores"]
                selected_scores = cell["selected_hidden"]["scores"]
                if condition == "defective":
                    if all(score == 1 for score in observed["scores"]):
                        observation = "defect_not_manifest_in_parent; diagnostic_inconclusive"
                    elif not cell["instruction_changed"]:
                        observation = "defect_observed_no_admitted_repair"
                    elif (sum(selected_scores) > sum(parent_scores)
                          and all(new >= old for new, old in zip(selected_scores, parent_scores))):
                        observation = "admitted_repair_improved_fixed_hidden_records"
                    else:
                        observation = "admitted_rewrite_without_hidden_improvement"
                else:
                    observation = ("clean_preserved" if not cell["instruction_changed"]
                                   and all(score == 1 for score in observed["scores"] + parent_scores + selected_scores)
                                   else "clean_control_changed_or_failed")
                cell["observation"] = observation
                cell["calls"] = len(runtime.attempts) - start
                cell["status"] = "success"
        result["status"] = "success"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = runtime.sanitize(f"{type(error).__name__}: {error}", truncate=False)
        if result["cells"]:
            result["cells"][-1]["status"] = "failed"
            result["cells"][-1]["calls"] = len(runtime.attempts) - start
    finally:
        result["calls"] = len(runtime.attempts)
        result["reported_cost_usd"] = sum(row.get("cost") or 0 for row in runtime.attempts)
        result["cost_metadata_complete"] = all(row.get("cost") is not None for row in runtime.attempts)
        runtime.write_jsonl(output / "calls.jsonl")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def live_factory(model):
    def build(arm, condition):
        config = ModelConfig(task_model=model, reflection_model=model,
                             enable_reasoning=False, provider_allow_fallbacks=False)
        lms = create_lms(config, seed=0)
        for role, lm in lms.items():
            lm.kwargs.update(max_tokens=128 if role == "task" else 1024, temperature=0, n=1)
            lm.cache = False
            lm.num_retries = 0
        return lms
    return build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Authorize bounded paid provider calls")
    parser.add_argument("--model", default="openrouter/qwen/qwen3.5-9b")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/theory_driven/results/positive-control"))
    args = parser.parse_args(argv)
    plan = {"mode": "live" if args.confirm else "dry-run", "source": SOURCE, "scope": SCOPE,
            "model": args.model, "cells": ["B:defective", "B:clean", "M:defective", "M:clean"],
            **SETTINGS, "output": str(args.output)}
    if not args.confirm:
        print(json.dumps(plan, indent=2))
        return 0
    result = run_diagnostic(args.output, lm_factory=live_factory(args.model), model=args.model)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
    return 0 if result["status"] == "success" else 1


def examples():
    """Fixed manually authored records; hidden entities never enter reflection."""
    rows = [
        ("Aster", "red", "Lima", "color"),
        ("Birch", "blue", "Oslo", "color"),
        ("Cedar", "green", "Kyoto", "city"),
        ("Dahlia", "amber", "Riga", "color"),
        ("Elm", "violet", "Perth", "color"),
        ("Fir", "teal", "Accra", "city"),
    ]
    result = []
    for entity, color, city, field in rows:
        record = {"color": color, "city": city}
        question = json.dumps({"source": SOURCE, "entity": entity,
                               "record": record, "requested_field": field}, sort_keys=True)
        result.append(dspy.Example(question=question, answer=record[field]).with_inputs("question"))
    return result[:3], result[3:]


class RecordQA(dspy.Module):
    def __init__(self, instruction: str):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")
        assert self.answer.signature is not None
        self.answer.signature = self.answer.signature.with_instructions(instruction)

    def forward(self, question):
        return self.answer(question=question)


def exact_metric(gold, prediction, trace=None, pred_name=None, pred_trace=None):
    """No normalization, judge LM, or invented evidence; gold stays in scoring."""
    score = float(prediction.answer == gold.answer)
    return dspy.Prediction(
        score=score,
        feedback=(f"Synthetic task contract: return record[requested_field], exactly; "
                  f"all needed evidence is in the input. Exact match score={score}; "
                  f"expected answer={gold.answer!r}; observed answer={prediction.answer!r}. "
                  "Gold is evaluator-only, not a task input. No retrieval is needed."),
    )


if __name__ == "__main__":
    raise SystemExit(main())
