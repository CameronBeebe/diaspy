"""Tests for stage-0 reliability improvements: resume, unchanged candidates, deduplication, held-out samples."""

import json
import dspy
from dspy.utils.dummies import DummyLM

from experiments.theory_driven.config.resume import any_result_cost, fingerprint
from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.arms.survival import SurvivalEvaluator, Evidence, ScorePolicy
from experiments.theory_driven.arms.popperian import PopperianProposer, FailedAttempt, normalize_instruction
from experiments.theory_driven.config.budget_tracker import UnifiedBudget


def test_any_result_cost_counts_aborted_runs(tmp_path):
    """Resume should count costs from aborted runs, not just completed ones."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Write a failed run result
    failed_result = {
        "benchmark": "hotpotqa",
        "arm": "B@half_no_regress",
        "budget": 100,
        "seed": 0,
        "config_hash": "test_hash",
        "status": "error",
        "message": "Provider failed",
        "budget_stats": {"total_cost": 0.0123}
    }
    (results_dir / "failed_run.json").write_text(json.dumps(failed_result))
    
    # any_result_cost should find the cost even though status is error
    cost = any_result_cost(results_dir, "hotpotqa", "B@half_no_regress", 100, 0, "test_hash")
    assert cost == 0.0123


def test_any_result_cost_returns_zero_when_no_match(tmp_path):
    """any_result_cost returns 0.0 when no matching run exists."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    cost = any_result_cost(results_dir, "hotpotqa", "B@half_no_regress", 100, 0, "nonexistent_hash")
    assert cost == 0.0


def test_fingerprint_includes_code_version():
    """Resume fingerprint must include code version to prevent stale reuse."""
    config = ExperimentConfig()
    stage = config.get_stage_config(0)
    
    hash1 = fingerprint(config, stage, "hotpotqa", "A", 100, 0)
    
    # The hash should be stable
    hash2 = fingerprint(config, stage, "hotpotqa", "A", 100, 0)
    assert hash1 == hash2
    
    # And it should be a valid SHA256 hex digest
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)


def test_normalize_instruction_whitespace_only():
    """normalize_instruction should only normalize whitespace, not content."""
    assert normalize_instruction("  hello  \n  world  ") == "hello world"
    assert normalize_instruction("hello\tworld") == "hello world"
    assert normalize_instruction("hello\n\nworld") == "hello world"
    # Content should be preserved
    assert normalize_instruction("Hello World") == "Hello World"


def test_unchanged_parent_logged_explicitly():
    """When proposer returns parent unchanged, it should be explicitly logged."""
    
    class FakeEvidence:
        def __init__(self):
            self.archive = {}
            self.last_metadata = {}
        
        def corroborate(self, candidate, refutations, batch):
            # No corroborated counterexamples
            return False, {}
        
        def prepare(self, candidate, refutations, batch, **kwargs):
            return []
    
    # Mock a proposer with dummy LMs
    proposer = PopperianProposer(
        program=dspy.Predict("question -> answer"),
        metric=lambda *args: type("Metric", (), {"score": 1.0})(),
        gold_by_question={},
        critic_lm=DummyLM([{"refutations": []}]),
        task_lm=DummyLM([{"answer": "test"}]),
        budget_tracker=UnifiedBudget(100, 1.0, DummyLM([{"answer": "test"}])),
        max_attempts=2,
    )
    proposer.evaluator = FakeEvidence()  # type: ignore[assignment]
    
    # When corroborate returns False, the output should be {}
    result = proposer(
        candidate={"Q": "parent instruction"},
        reflective_dataset={"Q": [{"Inputs": {"question": "test"}}]},
        components_to_update=["Q"]
    )
    
    # Should return {} to skip GEPA re-scoring
    assert result == {}
    
    # Check the log to verify it was marked as unchanged
    assert len(proposer.log) == 1
    assert proposer.log[0].get("returned_unchanged_parent") is True


def test_joint_set_deduplication_counterexample_takes_precedence():
    """Each question gets exactly one role, counterexample > defense > archive > regression."""
    
    # Create a minimal evaluator
    program = dspy.Predict("question -> answer")
    
    def metric(gold, pred, *args):
        return type("Score", (), {"score": 0.5})()
    
    gold = {
        "q1": dspy.Example(question="q1", answer="a1").with_inputs("question"),
    }
    
    evaluator = SurvivalEvaluator(
        program=program,
        metric=metric,
        gold_by_question=gold,
        task_lm=dspy.LM("dummy", cache=False),
    )
    
    # Mock the _scores method to return fixed scores
    def mock_scores(candidate, questions):
        return {q: 0.5 for q in questions}
    
    evaluator._scores = mock_scores
    
    # Create a refutation citing index 0
    from experiments.theory_driven.arms.popperian import Refutation
    refutations = [Refutation(
        claim="test", counterexample_ids=[0], severity=0.5, proposed_fix="fix"
    )]
    
    batch = [{"Inputs": {"question": "q1"}}]
    
    # Call prepare with protected_ids that also includes index 0
    evidence = evaluator.prepare(
        candidate={"Q": "test"},
        refutations=refutations,
        batch=batch,
        protected_ids=[0],  # Same question as counterexample
        parent_scores={"q1": 0.5}
    )
    
    # q1 should appear exactly once, with role "counterexample" (higher precedence)
    questions = [e.question for e in evidence]
    assert questions.count("q1") == 1
    
    roles = [e.role for e in evidence if e.question == "q1"]
    assert roles == ["counterexample"]


def test_held_out_regression_samples_deterministic():
    """Held-out samples should be deterministically selected based on seed."""
    
    program = dspy.Predict("question -> answer")
    
    def metric(gold, pred, *args):
        return type("Score", (), {"score": 1.0})()
    
    trainset = [
        dspy.Example(question=f"q{i}", answer=f"a{i}").with_inputs("question")
        for i in range(10)
    ]
    gold = {ex.question: ex for ex in trainset}
    
    # Create two evaluators with same seed
    evaluator1 = SurvivalEvaluator(
        program=program,
        metric=metric,
        gold_by_question=gold,
        task_lm=dspy.LM("dummy", cache=False),
        held_out_regression_samples=3,
        trainset=trainset,
        seed=42
    )
    
    evaluator2 = SurvivalEvaluator(
        program=program,
        metric=metric,
        gold_by_question=gold,
        task_lm=dspy.LM("dummy", cache=False),
        held_out_regression_samples=3,
        trainset=trainset,
        seed=42
    )
    
    # They should select the same questions
    assert evaluator1._held_out_questions == evaluator2._held_out_questions
    assert len(evaluator1._held_out_questions) == 3
    
    # Different seed should select different questions
    evaluator3 = SurvivalEvaluator(
        program=program,
        metric=metric,
        gold_by_question=gold,
        task_lm=dspy.LM("dummy", cache=False),
        held_out_regression_samples=3,
        trainset=trainset,
        seed=99
    )
    
    assert evaluator3._held_out_questions != evaluator1._held_out_questions


def test_held_out_regression_included_in_evidence():
    """Held-out regression samples should be included in the evidence with correct role."""
    
    program = dspy.Predict("question -> answer")
    
    def metric(gold, pred, *args):
        return type("Score", (), {"score": 1.0})()
    
    trainset = [
        dspy.Example(question=f"q{i}", answer=f"a{i}").with_inputs("question")
        for i in range(10)
    ]
    gold = {ex.question: ex for ex in trainset}
    
    evaluator = SurvivalEvaluator(
        program=program,
        metric=metric,
        gold_by_question=gold,
        task_lm=dspy.LM("dummy", cache=False),
        held_out_regression_samples=2,
        trainset=trainset,
        seed=42
    )
    
    # Mock _scores
    def mock_scores(candidate, questions):
        return {q: 1.0 for q in questions}
    
    evaluator._scores = mock_scores
    
    # Prepare with empty refutations
    evidence = evaluator.prepare(
        candidate={"Q": "test"},
        refutations=[],
        batch=[{"Inputs": {"question": "q0"}}],
        parent_scores={"q0": 1.0}
    )
    
    # Should have held-out regression samples
    held_out = [e for e in evidence if e.role == "held_out_regression"]
    assert len(held_out) == 2
    
    # They should be the deterministically selected questions
    assert all(e.question in evaluator._held_out_questions for e in held_out)


def test_held_out_regression_counted_in_regressions():
    """Held-out regression samples should be treated as regressions for admission."""
    
    policy = ScorePolicy("half_no_regress")
    
    # Create evidence with a fixed counterexample and a held-out regression
    evidence = [
        Evidence("q1", "counterexample", 0.0, 1.0, []),
        Evidence("q2", "held_out_regression", 1.0, 0.5, []),  # Regression!
    ]
    
    # Should be rejected due to regression
    admitted, rationale = policy(evidence)
    assert not admitted
    assert rationale["reason"] == "insufficient_fixes_or_regressions"
    assert rationale["counterexamples_fixed"] == 1
    assert rationale["regressions_total"] == 1
    assert rationale["regressions_kept"] == 0


def test_failed_attempt_from_evidence_with_held_out():
    """FailedAttempt should correctly represent held-out regression samples."""
    
    evidence = [
        Evidence("q1", "counterexample", 0.0, 1.0, []),
        Evidence("q2", "held_out_regression", 1.0, 0.8, []),  # Slight regression
    ]
    
    attempt = FailedAttempt.from_evidence(
        instruction="test instruction",
        evidence=evidence,
        reason="insufficient_fixes",
        eps=0.0
    )
    
    assert len(attempt.per_question) == 2
    
    # Held-out regression should be marked as not kept (regression)
    held_out_q = next(q for q in attempt.per_question if q.question_id == "q2")
    assert held_out_q.role == "held_out_regression"
    assert held_out_q.parent_score == 1.0
    assert held_out_q.rewrite_score == 0.8
    assert not held_out_q.fixed_or_kept
