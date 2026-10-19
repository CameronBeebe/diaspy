"""Observe stock GEPA proposals and its actual candidate-pool decisions.

The default proposer is delegated to verbatim. GEPA's public observation
callbacks supply minibatch scores and admission, NOT an inferred comparison or
final-best selection. No callback mutates GEPA state or makes an LM call.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from dspy.teleprompt.gepa import InstructionProposer

logger = logging.getLogger(__name__)


class GEPAProposalTrace:
    """Delegating proposer plus GEPA callbacks for the stock single-mutation arm.

    ``accepted`` means insertion into GEPA's candidate pool. It is None until an
    authoritative decision callback arrives; failure/skip is not rejection.
    Events are emitted at iteration end (also on fatal compile exit), once per
    proposer invocation. Multi-proposal strategies with ambiguous callback
    identity deliberately retain unknown decisions rather than misattribute them.
    """

    def __init__(self, event_sink: Callable[[dict[str, Any]], None] | None = None):
        self.delegate = InstructionProposer()
        self.event_sink = event_sink
        self.log: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {}
        self._selected_count = 0

    def on_iteration_start(self, event: dict[str, Any]) -> None:
        self._context = {"iteration": event["iteration"]}
        self._selected_count = 0

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        self._selected_count += 1
        self._context.update(parent_candidate_id=event["candidate_idx"],
                             parent_validation_score=event["score"])

    def on_minibatch_sampled(self, event: dict[str, Any]) -> None:
        self._context["minibatch_ids"] = list(event["minibatch_ids"])

    def on_evaluation_start(self, event: dict[str, Any]) -> None:
        if event["candidate_idx"] is not None and event["capture_traces"]:
            self._context["minibatch_questions"] = [item.get("question") for item in event["inputs"]]

    def on_evaluation_end(self, event: dict[str, Any]) -> None:
        if event["candidate_idx"] is not None:
            self._context["parent_scores"] = list(event["scores"])
        elif (record := self._proposal()) is not None:
            record["proposed_scores"] = list(event["scores"])

    def __call__(
        self, candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        record = {
            "kind": "gepa_proposal", **deepcopy(self._context),
            "parent_instructions": dict(candidate),
            "components": list(components_to_update),
            "proposed_instructions": None, "proposed_candidate": None,
            "proposed_scores": None, "accepted": None, "candidate_id": None,
            "validation_score": None, "validation_scores": None,
            "status": "pending", "reason": "no_gepa_decision_observed",
        }
        self._pending.append(record)
        # Pass the original objects through: do not reimplement reflection,
        # filter evidence, normalize text, or evaluate any extra candidates.
        try:
            updates = self.delegate(candidate, reflective_dataset, components_to_update)
        except Exception as exc:
            record.update(status="proposal_failed", reason="instruction_proposer_failed",
                          error=f"{type(exc).__name__}: {exc}")
            raise
        record["proposed_instructions"] = dict(updates)
        record["proposed_candidate"] = {**candidate, **updates}
        if not updates:
            record.update(status="no_updates", reason="instruction_proposer_returned_no_updates")
        return updates

    def _proposal(self) -> dict[str, Any] | None:
        proposals = [record for record in self._pending if record["proposed_instructions"]]
        if self._selected_count == 1 and len(proposals) == 1:
            return proposals[0]
        return None

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        record = self._proposal()
        if record is not None and event["candidate"] == record["proposed_candidate"]:
            record["validation_score"] = event["average_score"]
            record["validation_scores"] = dict(event["scores_by_val_id"])

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        if (record := self._proposal()) is not None:
            record.update(accepted=True, status="accepted", reason="gepa_candidate_accepted",
                          candidate_id=event["new_candidate_idx"],
                          parent_ids=list(event["parent_ids"]),
                          proposed_score_sum=event["new_score"])

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        if (record := self._proposal()) is not None:
            record.update(accepted=False, status="rejected", reason=event["reason"],
                          parent_score_sum=event["old_score"],
                          proposed_score_sum=event["new_score"])

    def on_iteration_end(self, event: dict[str, Any]) -> None:
        self.flush()

    def on_error(self, event: dict[str, Any]) -> None:
        for record in self._pending:
            if record["accepted"] is None and record["status"] == "pending":
                exc = event["exception"]
                record.update(status="interrupted", reason="gepa_error_before_decision",
                              error=f"{type(exc).__name__}: {exc}")

    def flush(self) -> None:
        """Persist pending observations, including when compile raises."""
        pending, self._pending = self._pending, []
        for record in pending:
            snapshot = deepcopy(record)
            self.log.append(snapshot)
            if self.event_sink is not None:
                try:
                    self.event_sink(deepcopy(snapshot))
                except Exception:
                    # Instrumentation must not turn a successful stock run into
                    # a failed optimization. Keep the record in self.log too.
                    logger.exception("GEPA proposal event sink failed")
