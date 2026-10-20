"""Arm B: Typed Popperian proposer for GEPA.

Based on popperian_gepa_skeleton.py, adapted for real LMs.
"""

import json
from dataclasses import asdict
from typing import Callable, List, Dict, Any, Optional
import pydantic
import dspy
from dspy.clients.base_lm import BaseLM
from collections.abc import Mapping, Sequence
from experiments.theory_driven.arms.base import BaseArm
from experiments.theory_driven.arms.survival import AdmissionPolicy, Evidence, SurvivalEvaluator, get_policy
from experiments.theory_driven.config.budget_tracker import BudgetTracker, UnifiedBudget


class Refutation(pydantic.BaseModel):
    """One falsifiable claim about why the current instruction fails."""
    claim: str  # "instruction fails because ..."
    counterexample_ids: List[int]  # indices into the reflective minibatch
    severity: float = pydantic.Field(ge=0, le=1)
    proposed_fix: str


class Critique(dspy.Signature):
    """Conjecture-and-refutation: state each way the instruction fails as a
    falsifiable claim, cite the example ids that break it, and rate severity.
    """
    current_instruction: str = dspy.InputField()
    examples_with_feedback: str = dspy.InputField(
        desc="numbered examples: inputs, outputs, feedback"
    )
    refutations: List[Refutation] = dspy.OutputField()


def normalize_instruction(instruction: str) -> str:
    return " ".join(instruction.split())


class AttemptQuestion(pydantic.BaseModel):
    question_id: str
    role: str
    parent_score: float
    rewrite_score: float | None
    fixed_or_kept: bool


class FailedAttempt(pydantic.BaseModel):
    instruction: str
    per_question: List[AttemptQuestion]
    rejection_reason: str

    @classmethod
    def from_evidence(cls, instruction: str, evidence: Sequence[Evidence], reason: str,
                      eps: float = 0.0) -> "FailedAttempt":
        rows = []
        for row in evidence:
            reference = row.parent_score if row.reference_score is None else row.reference_score
            passed = row.rewrite_score is not None and (
                row.rewrite_score > row.parent_score + eps if row.role == "counterexample"
                else row.role in {"regression", "archive", "defense"} and not row.vacuous
                and row.rewrite_score >= reference - eps)
            rows.append(AttemptQuestion(question_id=row.question, role=row.role,
                                        parent_score=row.parent_score, rewrite_score=row.rewrite_score,
                                        fixed_or_kept=passed))
        return cls(instruction=instruction, per_question=rows, rejection_reason=reason)


class Rewrite(dspy.Signature):
    """Rewrite the instruction so that it survives every refutation. Address each one."""
    current_instruction: str = dspy.InputField()
    refutations: List[Refutation] = dspy.InputField()
    failed_attempts: List[FailedAttempt] = dspy.InputField(
        desc="All failed rewrites this round and their observed per-question refutations. Do not repeat them.")
    new_instruction: str = dspy.OutputField()


class PopperianProposer:
    """GEPA ProposalFn: critic -> typed refutations -> rewrite -> mechanical survival check.
    
    Extension points used (per robustness audit requirements):
    - GEPA instruction_proposer: Custom proposal function (documented in GEPA API)
    - Mechanical survival check: Re-runs counterexamples with new instruction
    - Refutation archive: Regression tests from observed past fixes
    """
    
    def __init__(
        self,
        program: dspy.Module,
        metric: Callable,
        gold_by_question: Dict[str, Any],
        critic_lm: BaseLM,
        task_lm: BaseLM,
        budget_tracker: BudgetTracker | UnifiedBudget,
        max_attempts: int = 2,
        severity_tau: float | None = None,
        archive_cap: int = 6,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        survival_policy: str | AdmissionPolicy = "strict",
        min_survival_questions: int = 3,
        eps: float = 0.0,
    ):
        """Initialize Popperian proposer.
        
        Args:
            program: DSPy program being optimized
            metric: Evaluation metric
            gold_by_question: Mapping from question to gold example
            critic_lm: LM for critique and rewrite
            task_lm: LM for task execution
            budget_tracker: History-based budget stopper
            max_attempts: Maximum rewrite attempts per iteration
            severity_tau: Deprecated explicit legacy citation filter (default: disabled)
            archive_cap: Maximum archive size
            event_sink: Immediate survival-admission event callback (not GEPA acceptance)
        """
        self.program = program
        self.metric = metric
        self.gold = gold_by_question
        self.critic_lm = critic_lm
        self.task_lm = task_lm
        self.budget_tracker = budget_tracker
        self.max_attempts = max_attempts
        self.policy = get_policy(survival_policy, eps)
        self.evaluator = SurvivalEvaluator(
            program, metric, gold_by_question, task_lm, min_survival_questions,
            severity_tau, archive_cap, eps)
        self.last_evidence: list[Evidence] = []
        self.last_rationale: dict[str, Any] = {}
        self.event_sink = event_sink
        
        self.critique = dspy.Predict(Critique)
        self.rewrite = dspy.Predict(Rewrite)
        self.log: List[Dict[str, Any]] = []
        
    @property
    def archive(self) -> dict[str, float]:
        """Compatibility view of the evaluator's scored archive."""
        return self.evaluator.archive

    def survives(
        self,
        predictor_name: str,
        instruction: str,
        refutations: List[Refutation],
        batch: list,
        candidate: Dict[str, str],
        evidence: list[Evidence] | None = None,
    ) -> tuple[bool, List[str]]:
        """Evaluate the entire survival set against observed parent/reference scores."""
        if evidence is None:
            evidence = self.evaluator.prepare(candidate, refutations, batch)
        self.last_evidence = self.evaluator.evaluate(candidate, predictor_name, instruction, evidence)
        admitted, self.last_rationale = self.policy(self.last_evidence)
        if admitted:
            self.evaluator.admit_archive(self.last_evidence)
        questions = list(dict.fromkeys(e.question for e in self.last_evidence
                                      if e.rewrite_score is not None))
        return admitted, questions

    def __call__(
        self,
        candidate: Dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: List[str]
    ) -> Dict[str, str]:
        """GEPA proposal function interface.
        
        Args:
            candidate: Current instruction dict {component_name: instruction}
            reflective_dataset: Minibatch dict {component_name: examples}
            components_to_update: List of component names to update
            
        Returns:
            Updated instruction dict
        """
        out = {}
        
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            
            batch = list(reflective_dataset[name])
            shown = "\n".join(
                f"[{i}] {json.dumps(ex, default=str)}"
                for i, ex in enumerate(batch)
            )
            
            # Critique with reflection LM
            # Attribute the shared critique to the first rewrite attempt only.
            calls_before = self.budget_tracker.total_units
            cost_before = self.budget_tracker.total_cost
            with dspy.context(lm=self.critic_lm):
                critique_result = self.critique(
                    current_instruction=candidate[name],
                    examples_with_feedback=shown
                )
                refutations = critique_result.refutations
            
            
            if self.max_attempts <= 0:
                out[name] = candidate[name]
                continue
            critique_calls = self.budget_tracker.total_units - calls_before
            critique_cost = self.budget_tracker.total_cost - cost_before
            parent_before = self.budget_tracker.total_units
            parent_cost_before = self.budget_tracker.total_cost
            corroborated, parent_scores = self.evaluator.corroborate(candidate, refutations, batch)
            if not corroborated:
                record = {
                    "policy": self.policy.name, "attempt": 0, "predictor": name,
                    "old_instruction": candidate[name], "new_instruction": candidate[name],
                    "refutations": [r.model_dump() for r in refutations], "n_refutations": len(refutations),
                    "evidence": [], "checked_ids": [], "tested_questions": [], "survived": False,
                    "rationale": {"reason": "no_corroborated_counterexample"},
                    "severity_usage": "descriptive_only", "calls_consumed": 0, "cost_consumed": 0.0,
                    "critique_calls_consumed": critique_calls, "critique_cost_consumed": critique_cost,
                    "parent_calls_consumed": self.budget_tracker.total_units - parent_before,
                    "parent_cost_consumed": self.budget_tracker.total_cost - parent_cost_before,
                }
                self.log.append(record)
                if self.event_sink is not None:
                    self.event_sink(record)
                out[name] = candidate[name]
                continue
            # Rewrite attempts
            new = candidate[name]
            ok = False
            evidence: list[Evidence] | None = None
            seen = {normalize_instruction(candidate[name])}
            self.evaluator.last_metadata = {}
            failed_attempts: list[FailedAttempt] = []
            for attempt in range(1, self.max_attempts + 1):
                with dspy.context(lm=self.critic_lm):
                    rewrite_result = self.rewrite(
                        current_instruction=candidate[name],
                        refutations=refutations,
                        failed_attempts=failed_attempts,
                    )
                    new = rewrite_result.new_instruction
                
                
                # Survival check
                normalized = normalize_instruction(new)
                if normalized in seen:
                    ok, ids = False, []
                    self.last_evidence = []
                    self.last_rationale = {"reason": "duplicate_rewrite"}
                else:
                    if evidence is None:
                        evidence = self.evaluator.prepare(candidate, refutations, batch, parent_scores=parent_scores)
                    ok, ids = self.survives(name, new, refutations, batch, candidate, evidence)
                seen.add(normalized)
                
                record = {
                    "policy": self.policy.name,
                    "evidence": [asdict(e) for e in self.last_evidence],
                    "severity_usage": "descriptive_only",
                    "rationale": {**self.last_rationale, **self.evaluator.last_metadata},
                    "legacy_severity_tau": self.evaluator.severity_tau,
                    "attempt": attempt,
                    "predictor": name,
                    "old_instruction": candidate[name],
                    "n_refutations": len(refutations),
                    "refutations": [r.model_dump() for r in refutations],
                    "checked_ids": ids,
                    "tested_questions": ids,
                    "survived": ok,
                    "new_instruction": new,
                    "calls_consumed": self.budget_tracker.total_units - calls_before,
                    "cost_consumed": self.budget_tracker.total_cost - cost_before,
                }
                self.log.append(record)
                if self.event_sink is not None:
                    self.event_sink(record)
                calls_before = self.budget_tracker.total_units
                cost_before = self.budget_tracker.total_cost
                
                if ok:
                    break
                failed_attempts.append(FailedAttempt.from_evidence(
                    new, self.last_evidence, self.last_rationale["reason"], self.evaluator.eps))
            
            # Return new instruction if it survived, otherwise return old instruction
            # This rejects failing rewrites immediately instead of letting GEPA test them
            if ok:
                out[name] = new
            else:
                # Keep parent instruction - reject the failing rewrite
                out[name] = candidate[name]
        
        return out


class PopperianArm(BaseArm):
    """Arm B: GEPA + typed Popperian proposer."""
    
    def __init__(
        self,
        name: str,
        description: str,
        budget_tracker: BudgetTracker | UnifiedBudget,
        max_metric_calls: int = 200,
        reflection_minibatch_size: int = 3,
        use_merge: bool = False,
        seed: int = 0,
        max_attempts: int = 2,
        severity_tau: float | None = None,
        archive_cap: int = 6,
        survival_policy: str | AdmissionPolicy = "strict",
        min_survival_questions: int = 3,
        eps: float = 0.0,
        **kwargs
    ):
        """Initialize Popperian arm."""
        super().__init__(name, description, budget_tracker, **kwargs)
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.use_merge = use_merge
        self.seed = seed
        self.max_attempts = max_attempts
        self.severity_tau = severity_tau
        self.archive_cap = archive_cap
        self.policy = get_policy(survival_policy, eps)
        self.survival_policy = self.policy.name
        self.min_survival_questions = min_survival_questions
        self.eps = eps
        self.proposer: Optional[PopperianProposer] = None
    
    def compile(
        self,
        program: dspy.Module,
        metric: Callable,
        trainset: list,
        valset: list,
        task_lm=None,
        reflection_lm=None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        **compile_kwargs
    ) -> dspy.Module:
        """Compile with Popperian proposer.
        
        Args:
            program: Program to optimize
            metric: Evaluation metric
            trainset: Training examples
            valset: Validation examples
            reflection_lm: LM for critique/rewrite
            task_lm: LM for task execution
            event_sink: Immediate proposal survival-admission callback
            **compile_kwargs: Additional args
            
        Returns:
            Optimized program
        """
        task_lm = task_lm if task_lm is not None else dspy.settings.lm
        reflection_lm = reflection_lm if reflection_lm is not None else dspy.settings.lm
        if task_lm is None or reflection_lm is None:
            raise ValueError("Both task_lm and reflection_lm are required")
        # Build gold lookup
        gold_by_question = {ex.question: ex for ex in trainset + valset}
        
        # Create proposer
        self.proposer = PopperianProposer(
            program=program,
            metric=metric,
            gold_by_question=gold_by_question,
            critic_lm=reflection_lm,
            task_lm=task_lm,
            budget_tracker=self.budget_tracker,
            max_attempts=self.max_attempts,
            severity_tau=self.severity_tau,
            archive_cap=self.archive_cap,
            event_sink=event_sink,
            survival_policy=self.policy,
            min_survival_questions=self.min_survival_questions,
            eps=self.eps,
        )
        
        # Create GEPA with custom proposer
        gepa = dspy.GEPA(
            metric=metric,
            instruction_proposer=self.proposer,
            max_metric_calls=self.max_metric_calls,
            reflection_minibatch_size=self.reflection_minibatch_size,
            use_merge=self.use_merge,
            seed=self.seed,
            track_stats=True,
            num_threads=1,
            gepa_kwargs={
                **self.kwargs,
                "stop_callbacks": [self.budget_tracker],
            }
        )
        
        # Compile
        with dspy.context(lm=task_lm if task_lm is not None else dspy.settings.lm):
            optimized = gepa.compile(
                program,
                trainset=trainset,
                valset=valset,
            )
        
        return optimized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get arm statistics including proposer log."""
        stats = super().get_stats()
        stats["survival_policy"] = self.survival_policy
        stats["min_survival_questions"] = self.min_survival_questions
        if self.proposer:
            stats["proposer_log"] = self.proposer.log
            stats["archive_size"] = len(self.proposer.archive)
        return stats
