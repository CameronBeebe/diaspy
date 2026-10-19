"""Policy evidence is preserved in emitted survival metadata."""
import json
from experiments.theory_driven.events.run_trace import RunTrace


def test_survival_event_preserves_policy_and_score_vector(tmp_path):
    trace = RunTrace(tmp_path, "experiment", "hotpotqa", "B", {})
    evidence = [{"question": "q", "role": "counterexample", "parent_score": 0.3,
                 "rewrite_score": 0.6, "refutations": [{"severity": 0.0}],
                 "severity_usage": "descriptive_only"}]
    trace.proposal({"attempt": 1, "predictor": "answer", "old_instruction": "old",
                    "new_instruction": "new", "tested_questions": ["q"], "survived": True,
                    "refutations": [], "calls_consumed": 4, "cost_consumed": 0.0,
                    "policy": "majority", "evidence": evidence,
                    "rationale": {"fixed": "1/1"}, "severity_usage": "descriptive_only"})
    rows = [json.loads(line) for line in (tmp_path / trace.event_path).read_text().splitlines()]
    metadata = rows[-1]["metadata"]
    assert metadata["policy"] == "majority"
    assert metadata["evidence"] == evidence
    assert metadata["rationale"] == {"fixed": "1/1"}
    assert metadata["severity_usage"] == "descriptive_only"


def test_millian_debate_round_is_persisted_once(tmp_path):
    trace = RunTrace(tmp_path, "experiment", "hotpotqa", "M", {})
    record = {"kind": "millian_debate", "policy": "majority", "winner": None,
              "decision": "received_view_survived", "refutations": [], "rebuttals": [],
              "rewrites": {}, "evidence": [], "calls_by_role": {"critic": 4, "task": 9},
              "cost_by_role": {"critic": 0.0, "task": 0.0}}
    trace.proposal(record)
    rows = [json.loads(line) for line in (tmp_path / trace.event_path).read_text().splitlines()]
    assert len(rows) == 2
    assert rows[-1]["event_type"] == "optimization_step"
    assert rows[-1]["metadata"]["debate"] == record
    assert rows[-1]["metadata"]["decision_scope"] == "proposal_survival_admission"
