"""Arm M: typed Millian debate, with shared score-based survival policies.

The defender's stance steers only the rewrite prompt. Admission and selection
use observed task scores, never rhetorical stance or self-reported severity.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Literal

import dspy
from dspy.clients.base_lm import BaseLM
from pydantic import BaseModel, Field

from experiments.theory_driven.arms.base import BaseArm
from experiments.theory_driven.arms.popperian import (
    Critique, FailedAttempt, Refutation, Rewrite, normalize_instruction,
)
from experiments.theory_driven.arms.survival import AdmissionPolicy, Evidence, SurvivalEvaluator, get_policy
from experiments.theory_driven.config.budget_tracker import BudgetTracker, UnifiedBudget


class Rebuttal(BaseModel):
    """An argument about a refutation, not a numeric confidence estimate."""

    refutation_index: int
    stance: Literal["concede", "contest"]
    argument: str
    protected_ids: list[int]


class Defense(dspy.Signature):
    """Give the strongest defense of the current instruction against each
    refutation. Concede genuine failures; contest mistaken objections. Cite
    numbered examples the parent handles that the proposed fix might endanger.
    """

    current_instruction: str = dspy.InputField()
    refutations: list[Refutation] = dspy.InputField()
    examples_with_feedback: str = dspy.InputField()
    rebuttals: list[Rebuttal] = dspy.OutputField()


class DialecticalRewrite(dspy.Signature):
    """Rewrite after considering both objections and their strongest defenses.
    Fix conceded refutations while preserving contested and protected cases.
    State which refutations you reject and why; do not merely follow the defender.
    """

    current_instruction: str = dspy.InputField()
    refutations: list[Refutation] = dspy.InputField()
    rebuttals: list[Rebuttal] = dspy.InputField()
    failed_attempts: list[FailedAttempt] = dspy.InputField(
        desc="All failed rival rewrites this round with observed per-question scores. Do not repeat them.")
    new_instruction: str = dspy.OutputField()
    rejected_refutations: str = dspy.OutputField(desc="Rejected refutations and explanations, or none")


class RewriteRecord(BaseModel):
    """One independently scored rival on the common evidence vector."""

    new_instruction: str
    rejected_refutations: str = ""
    evidence: list[dict[str, Any]]
    admitted: bool
    rationale: dict[str, Any]
    total_gain: float


class DebateAttempt(BaseModel):
    attempt: int
    prosecution: RewriteRecord
    millian: RewriteRecord
    winner: Literal["prosecution", "millian"] | None


class DebateRecord(BaseModel):
    """One JSON-exportable record per component's debate round, including retries."""

    kind: Literal["millian_debate"] = "millian_debate"
    policy: str
    predictor: str
    old_instruction: str
    refutations: list[Refutation]
    rebuttals: list[Rebuttal]
    severity_descriptive_only: Literal[True] = True
    joint_set: list[dict[str, Any]]
    prosecution: RewriteRecord
    millian: RewriteRecord
    attempts: list[DebateAttempt]
    winner: Literal["prosecution", "millian"] | None
    received_view_survived: bool
    rationale: dict[str, Any]
    role_calls: dict[str, int]
    role_costs: dict[str, float]
    calls_consumed: int
    cost_consumed: float
    survived: bool
    new_instruction: str
    tested_questions: list[str] = Field(default_factory=list)


def _evidence_dicts(evidence: Sequence[Evidence]) -> list[dict[str, Any]]:
    rows = [asdict(item) for item in evidence]
    for row in rows:
        for refutation in row["refutations"]:
            refutation["severity_descriptive_only"] = True
    return rows


class MillianProposer:
    """GEPA proposer: prosecution, defense, rival rewrites, mechanical admission.

    ``max_attempts`` caps pairs of rival rewrites per round. The critique,
    defense, and joint parent evaluation are shared across those attempts.
    ``severity_tau`` is accepted for compatibility but never filters M's evidence.
    """

    def __init__(
        self,
        program: dspy.Module,
        metric: Callable,
        gold_by_question: dict[str, Any],
        critic_lm: BaseLM,
        task_lm: BaseLM,
        budget_tracker: BudgetTracker | UnifiedBudget,
        max_attempts: int = 2,
        severity_tau: float | None = None,
        archive_cap: int = 6,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        survival_policy: str | AdmissionPolicy = "strict",
        min_survival_questions: int = 3,
        eps: float = 0,
        defender_lm: BaseLM | None = None,
    ):
        self.program = program
        self.metric = metric
        self.gold = gold_by_question
        self.critic_lm = critic_lm
        self.task_lm = task_lm
        self.defender_lm = defender_lm if defender_lm is not None else critic_lm
        self.budget_tracker = budget_tracker
        self.max_attempts = max_attempts
        self.event_sink = event_sink
        self.policy = get_policy(survival_policy, eps=eps)
        self.evaluator = SurvivalEvaluator(
            program, metric, gold_by_question, task_lm,
            min_survival_questions=min_survival_questions, archive_cap=archive_cap, eps=eps,
        )
        self.critique = dspy.Predict(Critique)
        self.defense = dspy.Predict(Defense)
        self.rewrite = dspy.Predict(Rewrite)
        self.dialectical_rewrite = dspy.Predict(DialecticalRewrite)
        self.log: list[dict[str, Any]] = []

    @property
    def archive(self) -> dict[str, float]:
        return self.evaluator.archive

    @contextmanager
    def _account(self, role: str, calls: dict[str, int], costs: dict[str, float]):
        # Snapshot identities, not role labels: task/reflection/defender may alias.
        lms = {id(lm): lm for lm in (self.task_lm, self.critic_lm, self.defender_lm)}
        offsets = {identity: len(lm.history) for identity, lm in lms.items()}
        try:
            yield
        finally:
            entries = [entry for identity, lm in lms.items() for entry in lm.history[offsets[identity]:]]
            calls[role] = calls.get(role, 0) + len(entries)
            costs[role] = costs.get(role, 0.0) + sum(float(entry.get("cost") or 0) for entry in entries)

    def _score(self, candidate: dict[str, str], name: str, instruction: str,
               evidence: list[Evidence], explanation: str = "") -> tuple[RewriteRecord, list[Evidence]]:
        scored = self.evaluator.evaluate(candidate, name, instruction, evidence)
        admitted, rationale = self.policy(scored)
        # A question may carry both counterexample and archive/defense roles.
        # Those impose separate constraints, not extra weight in the tie-break.
        gains = {item.question: item.rewrite_score - item.parent_score
                 for item in scored if item.rewrite_score is not None}
        gain = sum(gains.values())
        return RewriteRecord(
            new_instruction=instruction, rejected_refutations=explanation,
            evidence=_evidence_dicts(scored), admitted=admitted, rationale=rationale, total_gain=gain,
        ), scored

    def _score_unless_duplicate(self, candidate: dict[str, str], name: str, instruction: str,
                                evidence: list[Evidence], seen: set[str],
                                explanation: str = "") -> tuple[RewriteRecord, list[Evidence]]:
        if normalize_instruction(instruction) in seen:
            return RewriteRecord(new_instruction=instruction, rejected_refutations=explanation,
                                 evidence=[], admitted=False, rationale={"reason": "duplicate_rewrite"},
                                 total_gain=0), []
        return self._score(candidate, name, instruction, evidence, explanation)

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            out[name] = candidate[name]
            if self.max_attempts <= 0:
                continue
            calls: dict[str, int] = {}
            costs: dict[str, float] = {}
            batch = list(reflective_dataset[name])
            shown = "\n".join(f"[{i}] {json.dumps(row, default=str)}" for i, row in enumerate(batch))
            with self._account("prosecutor", calls, costs), dspy.context(lm=self.critic_lm):
                refutations = self.critique(
                    current_instruction=candidate[name], examples_with_feedback=shown,
                ).refutations
            with self._account("defender", calls, costs), dspy.context(lm=self.defender_lm):
                rebuttals = self.defense(
                    current_instruction=candidate[name], refutations=refutations, examples_with_feedback=shown,
                ).rebuttals
            protected = sorted({index for rebuttal in rebuttals for index in rebuttal.protected_ids})
            with self._account("parent", calls, costs):
                corroborated, parent_scores = self.evaluator.corroborate(candidate, refutations, batch)
            if not corroborated:
                reason = {"reason": "no_corroborated_counterexample", "decision": "received_view_survived"}
                skipped = RewriteRecord(new_instruction=candidate[name], evidence=[], admitted=False,
                                        rationale=reason, total_gain=0)
                record = DebateRecord(
                    policy=self.policy.name, predictor=name, old_instruction=candidate[name],
                    refutations=refutations, rebuttals=rebuttals, joint_set=[],
                    prosecution=skipped, millian=skipped, attempts=[], winner=None,
                    received_view_survived=True, rationale=reason, role_calls=calls, role_costs=costs,
                    calls_consumed=0, cost_consumed=0, survived=False, new_instruction=candidate[name],
                ).model_dump(mode="json")
                self.log.append(record)
                if self.event_sink is not None:
                    self.event_sink(record)
                continue
            evidence: list[Evidence] = []
            prepared = False
            self.evaluator.last_metadata = {}
            attempts: list[DebateAttempt] = []
            failed_attempts: list[FailedAttempt] = []
            seen = {normalize_instruction(candidate[name])}
            winner: Literal["prosecution", "millian"] | None = None
            for attempt in range(1, self.max_attempts + 1):
                with self._account("prosecution_rewrite", calls, costs), dspy.context(lm=self.critic_lm):
                    prosecution = self.rewrite(current_instruction=candidate[name], refutations=refutations,
                                               failed_attempts=failed_attempts)
                with self._account("millian_rewrite", calls, costs), dspy.context(lm=self.critic_lm):
                    millian = self.dialectical_rewrite(
                        current_instruction=candidate[name], refutations=refutations, rebuttals=rebuttals,
                        failed_attempts=failed_attempts,
                    )
                if not prepared and any(normalize_instruction(text) not in seen for text in
                                        (prosecution.new_instruction, millian.new_instruction)):
                    with self._account("parent", calls, costs):
                        evidence = self.evaluator.prepare(candidate, refutations, batch,
                                                          protected_ids=protected, parent_scores=parent_scores)
                    prepared = True
                with self._account("prosecution_survival", calls, costs):
                    prosecution_record, prosecution_evidence = self._score_unless_duplicate(
                        candidate, name, prosecution.new_instruction, evidence, seen,
                    )
                with self._account("millian_survival", calls, costs):
                    if normalize_instruction(millian.new_instruction) == normalize_instruction(prosecution.new_instruction):
                        millian_record = prosecution_record.model_copy(deep=True, update={
                            "new_instruction": millian.new_instruction,
                            "rejected_refutations": millian.rejected_refutations,
                            "rationale": {**prosecution_record.rationale, "evaluation_reused": True},
                        })
                        millian_evidence = prosecution_evidence
                    else:
                        millian_record, millian_evidence = self._score_unless_duplicate(
                            candidate, name, millian.new_instruction, evidence, seen, millian.rejected_refutations,
                        )
                seen.update(normalize_instruction(text) for text in
                            (prosecution.new_instruction, millian.new_instruction))
                for rival, rows in ((prosecution_record, prosecution_evidence), (millian_record, millian_evidence)):
                    if not rival.admitted:
                        failed_attempts.append(FailedAttempt.from_evidence(
                            rival.new_instruction, rows, rival.rationale["reason"], self.evaluator.eps))
                passing = [(label, record, rows) for label, record, rows in (
                    ("prosecution", prosecution_record, prosecution_evidence),
                    ("millian", millian_record, millian_evidence),
                ) if record.admitted]
                if passing:
                    # Stable max makes prosecution the deterministic final tie-break.
                    label, winning, winning_evidence = max(
                        passing, key=lambda entry: (entry[1].total_gain, -len(entry[1].new_instruction)),
                    )
                    winner = "prosecution" if label == "prosecution" else "millian"
                    out[name] = winning.new_instruction
                    self.evaluator.admit_archive(winning_evidence)
                attempts.append(DebateAttempt(
                    attempt=attempt, prosecution=prosecution_record, millian=millian_record, winner=winner,
                ))
                if winner is not None:
                    break
            final = attempts[-1]
            record = DebateRecord(
                policy=self.policy.name, predictor=name, old_instruction=candidate[name],
                refutations=refutations, rebuttals=rebuttals, joint_set=_evidence_dicts(evidence),
                prosecution=final.prosecution, millian=final.millian, attempts=attempts, winner=winner,
                received_view_survived=winner is None,
                rationale={"decision": "received_view_survived" if winner is None else "rewrite_admitted",
                           **self.evaluator.last_metadata,
                           "selection_order": ["total_gain", "shorter_instruction", "prosecution"]},
                role_calls=calls, role_costs=costs, calls_consumed=sum(calls.values()),
                cost_consumed=sum(costs.values()), survived=winner is not None, new_instruction=out[name],
                tested_questions=[item.question for item in evidence if not item.role.startswith("uncorroborated")],
            ).model_dump(mode="json")
            self.log.append(record)
            if self.event_sink is not None:
                self.event_sink(record)
        return out


class MillianArm(BaseArm):
    """Independent GEPA arm using a Millian proposer, not a mode of arm B."""

    def __init__(
        self, name: str, description: str, budget_tracker: BudgetTracker | UnifiedBudget,
        max_metric_calls: int = 200, reflection_minibatch_size: int = 3,
        use_merge: bool = False, seed: int = 0, max_attempts: int = 2,
        severity_tau: float | None = None, archive_cap: int = 6,
        survival_policy: str | AdmissionPolicy = "strict", min_survival_questions: int = 3,
        eps: float = 0, defender_lm: BaseLM | None = None, **kwargs,
    ):
        super().__init__(name, description, budget_tracker, **kwargs)
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.use_merge = use_merge
        self.seed = seed
        self.max_attempts = max_attempts
        self.severity_tau = severity_tau
        self.archive_cap = archive_cap
        self.policy = get_policy(survival_policy, eps=eps)
        self.survival_policy = self.policy.name
        self.min_survival_questions = min_survival_questions
        self.eps = eps
        self.defender_lm = defender_lm
        self.proposer: MillianProposer | None = None

    def compile(
        self, program: dspy.Module, metric: Callable, trainset: list, valset: list,
        task_lm=None, reflection_lm=None, defender_lm=None,
        event_sink: Callable[[dict[str, Any]], None] | None = None, **compile_kwargs,
    ) -> dspy.Module:
        task_lm = task_lm if task_lm is not None else dspy.settings.lm
        reflection_lm = reflection_lm if reflection_lm is not None else dspy.settings.lm
        if task_lm is None or reflection_lm is None:
            raise ValueError("Both task_lm and reflection_lm are required")
        defender_lm = defender_lm if defender_lm is not None else self.defender_lm
        self.proposer = MillianProposer(
            program, metric, {ex.question: ex for ex in trainset + valset},
            reflection_lm, task_lm, self.budget_tracker, max_attempts=self.max_attempts,
            severity_tau=self.severity_tau, archive_cap=self.archive_cap,
            survival_policy=self.policy, min_survival_questions=self.min_survival_questions,
            eps=self.eps, defender_lm=defender_lm, event_sink=event_sink,
        )
        gepa = dspy.GEPA(
            metric=metric, instruction_proposer=self.proposer, max_metric_calls=self.max_metric_calls,
            reflection_minibatch_size=self.reflection_minibatch_size, use_merge=self.use_merge,
            seed=self.seed, track_stats=True, num_threads=1,
            gepa_kwargs={**self.kwargs, "stop_callbacks": [self.budget_tracker]},
        )
        with dspy.context(lm=task_lm):
            return gepa.compile(program, trainset=trainset, valset=valset)

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats["survival_policy"] = self.policy.name
        stats["min_survival_questions"] = self.min_survival_questions
        if self.proposer is not None:
            stats["proposer_log"] = self.proposer.log
            stats["archive_size"] = len(self.proposer.archive)
        return stats
