"""Persist sparse, per-arm epistemic events as each outcome becomes known."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.theory_driven.events.emitter import EventEmitter
from experiments.theory_driven.events.storage import EventStorage
from experiments.theory_driven.events.schema import (
    AcceptRejectDecision, Conjecture, EpistemicEvent, Refutation, TestOutcome, TestSelection,
)


class RunTrace:
    """A unique run with artifacts relative to its results directory."""

    def __init__(
        self,
        results_dir: Path,
        experiment_name: str,
        benchmark_name: str,
        arm_name: str,
        metadata: dict[str, Any],
        sanitizer: Callable[[str], str] | None = None,
    ):
        self.results_dir = Path(results_dir)
        self.sanitizer = sanitizer or (lambda text: text)
        self.run_id = uuid4().hex
        self.arm_name = arm_name
        self.storage = EventStorage(self.results_dir)
        self.emitter = EventEmitter(experiment_name, benchmark_name, self.run_id, arm_name)
        self.emitter.emit_initialization({**metadata, "arm_name": arm_name})
        self.event_path = self._persist()

    def _persist(self) -> str:
        # Sanitize the complete payload before EventStorage touches disk.
        events = [
            EpistemicEvent.model_validate(self._sanitize(event.model_dump(mode="json")))
            for event in self.emitter.get_events()
        ]
        path = self.storage.save_events(events, self.run_id, self.arm_name)
        return str(path.relative_to(self.results_dir))

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.sanitizer(value)
        if isinstance(value, dict):
            return {self.sanitizer(str(key)): self._sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        return value

    def proposal(self, record: dict[str, Any]) -> None:
        """Persist survival admission, never GEPA's later acceptance decision."""
        if record.get("kind") == "millian_debate":
            # The typed debate record contains both rivals, not a fictitious single step.
            event = EpistemicEvent(
                event_id=uuid4().hex, event_type="optimization_step",
                experiment_name=self.emitter.experiment_name,
                benchmark_name=self.emitter.benchmark_name, run_id=self.run_id,
                metadata={"decision_scope": "proposal_survival_admission", "debate": record},
            )
            self.emitter.events.append(event)
            self._persist()
            return
        survived = bool(record["survived"])
        questions = record["tested_questions"]
        reason = (
            "Survival passed: proposal admitted for GEPA evaluation; "
            "not a GEPA final candidate acceptance."
            if survived else
            "Survival failed: proposal rejected before GEPA evaluation."
        )
        event = self.emitter.emit_step(
            conjecture=Conjecture(
                candidate_id=uuid4().hex,
                module_name=record["predictor"],
                old_instruction=record["old_instruction"],
                new_instruction=record["new_instruction"],
                instruction_diff=self.emitter.compute_instruction_diff(
                    record["old_instruction"], record["new_instruction"],
                ),
                generated_by=self.arm_name,
                generation_reason="Typed refutations followed by mechanical survival check",
            ),
            test_selection=TestSelection(
                test_type="survival_check",
                # Questions, including archived cases, have no stable dataset IDs.
                example_ids=[], example_count=len(questions),
                selection_rationale="Mechanically corroborated counterexamples, archive and padding",
                test_cost_units=len(questions),
            ),
            test_outcome=TestOutcome(
                passed=survived, score=float(survived), example_scores=[],
                survival_check=True, survived=survived,
                feedback="Admission policy outcome; per-question scores are in event metadata",
            ),
            decision=AcceptRejectDecision(
                accepted=survived, decision_reason=reason,
                survival_check_passed=survived,
            ),
            calls_consumed=record["calls_consumed"],
            cost_consumed=record["cost_consumed"],
            refutations=[Refutation.model_validate(item) for item in record["refutations"]],
        )
        event.metadata = {
            "decision_scope": "proposal_survival_admission",
            "attempt": record["attempt"], "tested_questions": questions,
            **{key: record[key] for key in ("policy", "evidence", "rationale", "severity_usage")
               if key in record},
        }
        self._persist()

    def finalize(self, metadata: dict[str, Any]) -> dict[str, str]:
        """Persist the outcome, including errors, and return relative artifacts."""
        self.emitter.emit_finalization({**metadata, "arm_name": self.arm_name})
        self._persist()
        return {"epistemic_events": self.event_path}
