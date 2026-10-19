"""Observed-score evidence and pluggable admission policies shared by B and M.

Refutations (including severity) are descriptive provenance, not policy inputs.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol
import logging
from statistics import median

import dspy
from dspy.clients.base_lm import BaseLM


@dataclass
class Evidence:
    """One question's observed scores and the refutations that cited it."""

    question: str
    role: str
    parent_score: float
    rewrite_score: float | None
    refutations: list[dict[str, Any]]
    reference_score: float | None = None


class AdmissionPolicy(Protocol):
    """Extension point for score-based or future audited calibrated policies."""

    name: str

    def __call__(self, evidence: Sequence[Evidence]) -> tuple[bool, dict[str, Any]]:
        """Return an admission decision and a JSON-serializable explanation."""
        ...


@dataclass
class ScorePolicy:
    name: str
    eps: float = 0.0

    def __call__(self, evidence: Sequence[Evidence]) -> tuple[bool, dict[str, Any]]:
        counterexamples = [e for e in evidence if e.role == "counterexample"]
        fixed = sum(e.rewrite_score is not None and e.rewrite_score > e.parent_score + self.eps
                    for e in counterexamples)
        regressions: dict[str, list[Evidence]] = {}
        for row in evidence:
            if row.role in {"regression", "archive", "defense"}:
                regressions.setdefault(row.question, []).append(row)

        def kept(e: Evidence) -> bool:
            reference = e.parent_score if e.reference_score is None else e.reference_score
            return e.rewrite_score is not None and e.rewrite_score >= reference - self.eps

        # Each question gets one vote, but must satisfy every applicable reference.
        n_kept = sum(all(kept(e) for e in rows) for rows in regressions.values())
        vetoed = [question for question, rows in regressions.items()
                  if any(e.role == "archive" and not kept(e) for e in rows)]
        if self.name == "strict":
            admitted = fixed == len(counterexamples) and n_kept == len(regressions)
        else:
            # Empty regression groups are vacuously kept; a single real fix can pass.
            admitted = fixed * 2 > len(counterexamples) and (
                not regressions or n_kept * 2 > len(regressions))
        reason = "passed" if admitted else "insufficient_fixes_or_regressions"
        if self.name == "veto" and vetoed:
            admitted, reason = False, "archive_veto"
        if not counterexamples:
            admitted, reason = False, "no_corroborated_counterexample"
        return admitted, {"reason": reason, "counterexamples_fixed": fixed,
                          "counterexamples_total": len(counterexamples),
                          "regressions_kept": n_kept, "regressions_total": len(regressions),
                          "vetoed_questions": vetoed if self.name == "veto" else []}


def get_policy(name_or_policy: str | AdmissionPolicy, eps: float = 0.0) -> AdmissionPolicy:
    """Resolve a built-in policy or accept a caller-supplied policy object."""
    if not isinstance(name_or_policy, str):
        return name_or_policy
    if name_or_policy not in {"strict", "majority", "veto"}:
        raise ValueError(f"Unknown survival policy: {name_or_policy}")
    return ScorePolicy(name_or_policy, eps)


class SurvivalEvaluator:
    """Build observed evidence once and evaluate all questions for each rewrite."""

    def __init__(self, program: dspy.Module, metric: Callable,
                 gold_by_question: dict[str, Any], task_lm: BaseLM,
                 min_survival_questions: int = 3, severity_tau: float | None = None,
                 archive_cap: int = 6, eps: float = 0.0):
        self.program = program
        self.metric = metric
        self.gold = gold_by_question
        self.task_lm = task_lm
        self.min_survival_questions = min_survival_questions
        self.severity_tau = severity_tau
        self.archive_cap = archive_cap
        self.eps = eps
        self.archive: dict[str, float] = {}
        if severity_tau is not None:
            logging.getLogger(__name__).warning(
                "severity_tau is deprecated: explicit legacy filtering is enabled")

    def _scores(self, candidate: dict[str, str], questions: Sequence[str]) -> dict[str, float]:
        program = self.program.deepcopy()
        for name, predictor in program.named_predictors():
            signature = predictor.signature
            assert signature is not None
            predictor.signature = signature.with_instructions(candidate[name])
        scores = {}
        with dspy.context(lm=self.task_lm):
            for question in dict.fromkeys(questions):
                gold = self.gold[question]
                prediction = program(**gold.inputs())
                result = self.metric(gold, prediction, None, None, None)
                scores[question] = float(result.score)
        return scores

    def prepare(self, candidate: dict[str, str], refutations: list, batch: list,
                protected_ids: list[int] | None = None) -> list[Evidence]:
        """Observe the full parent minibatch; preserve dropped citations for tracing."""
        questions = [example["Inputs"]["question"] for example in batch]
        scores = self._scores(candidate, [*questions, *self.archive])
        citations: dict[str, list[dict[str, Any]]] = {}
        for refutation in refutations:
            ref = refutation.model_dump() if hasattr(refutation, "model_dump") else dict(refutation)
            if self.severity_tau is not None and ref["severity"] < self.severity_tau:
                continue
            for i in ref["counterexample_ids"]:
                if 0 <= i < len(questions):
                    refs = citations.setdefault(questions[i], [])
                    if ref not in refs:
                        refs.append(ref)
        evidence = []
        included: set[str] = set()
        for question in dict.fromkeys(questions):
            if question not in citations:
                continue
            role = "counterexample" if scores[question] < 1 else "uncorroborated-dropped"
            evidence.append(Evidence(question, role, scores[question], None, citations[question]))
            if role == "counterexample":
                included.add(question)
        if questions and protected_ids:
            threshold = median(scores[q] for q in questions)
            for i in sorted(set(protected_ids)):
                if not 0 <= i < len(questions):
                    continue
                question = questions[i]
                role = "defense" if scores[question] >= threshold else "uncorroborated_defense"
                evidence.append(Evidence(question, role, scores[question], None,
                                         citations.get(question, [])))
                if role == "defense":
                    included.add(question)
        # Keep an archive row even if also cited: its historical reference is distinct.
        for question, reference in self.archive.items():
            evidence.append(Evidence(question, "archive", scores[question], None,
                                     citations.get(question, []), reference))
            included.add(question)
        for question in dict.fromkeys(questions):
            if len(included) >= self.min_survival_questions:
                break
            if question not in included:
                evidence.append(Evidence(question, "regression", scores[question], None,
                                         citations.get(question, [])))
                included.add(question)
        return evidence

    def evaluate(self, candidate: dict[str, str], predictor_name: str, instruction: str,
                 evidence: list[Evidence]) -> list[Evidence]:
        """Return fresh evidence; never short-circuit or mutate the shared joint set."""
        active = {"counterexample", "archive", "regression", "defense"}
        scores = self._scores({**candidate, predictor_name: instruction},
                              [e.question for e in evidence if e.role in active])
        return [replace(e, rewrite_score=scores[e.question] if e.role in active else None)
                for e in evidence]

    def admit_archive(self, evidence: Sequence[Evidence]) -> None:
        """Record fixed counterexamples only; call only after policy admission.

        Existing archive references never decrease, even if a permissive policy
        admits a rewrite that regresses an older fix.
        """
        for row in evidence:
            if (row.role == "counterexample" and row.rewrite_score is not None
                    and row.rewrite_score > row.parent_score + self.eps):
                self.archive[row.question] = max(
                    self.archive.get(row.question, row.rewrite_score), row.rewrite_score)
        while len(self.archive) > max(0, self.archive_cap):
            del self.archive[next(iter(self.archive))]
