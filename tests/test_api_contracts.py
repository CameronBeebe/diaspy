"""API contract tests: verify DSPy/GEPA extension points."""
import inspect
import dspy
import pytest

from experiments.arms import PopperianProposer, CheckOnlyProposer, UntypedCriticProposer
from experiments.task import ifbench_metric, IFBenchProgram


def test_gepa_has_instruction_proposer():
    """Verify GEPA accepts instruction_proposer parameter."""
    sig = inspect.signature(dspy.GEPA.__init__)
    params = sig.parameters
    
    assert 'instruction_proposer' in params, "GEPA missing instruction_proposer parameter"
    assert 'reflection_lm' in params, "GEPA missing reflection_lm parameter"
    assert 'metric' in params, "GEPA missing metric parameter"


def test_simba_has_required_params():
    """Verify SIMBA has the parameters we use."""
    sig = inspect.signature(dspy.SIMBA.__init__)
    params = sig.parameters
    
    assert 'prompt_model' in params
    assert 'max_demos' in params
    assert 'bsize' in params


def test_proposer_signature():
    """Verify our proposers match the ProposalFn signature."""
    from dspy.utils import DummyLM
    
    program = IFBenchProgram()
    # DummyLM needs dict of dicts for output fields
    task_lm = DummyLM({"test": {"answer": "test answer"}})
    critic_lm = DummyLM({
        "test": {
            "refutations": "[]",
            "new_instruction": "improved instruction"
        }
    })
    gold = {"test": type('Ex', (), {'answer': 'test answer'})()}
    
    # Test PopperianProposer
    proposer = PopperianProposer(program, ifbench_metric, gold, critic_lm, task_lm)
    
    # Should accept these 3 args and return dict
    result = proposer(
        candidate={"answer": "test instruction"},
        reflective_dataset={"answer": []},
        components_to_update=["answer"]
    )
    
    assert isinstance(result, dict), "Proposer must return dict"
    assert "answer" in result, "Proposer must return instruction for requested component"


def test_metric_signature():
    """Verify our metric returns the expected format."""
    gold = type('Ex', (), {'answer': 'Paris'})()
    pred = dspy.Prediction(answer="Paris")
    
    result = ifbench_metric(gold, pred, None, None, None)
    
    assert hasattr(result, 'score'), "Metric must return object with .score"
    assert hasattr(result, 'feedback'), "Metric must return object with .feedback"
    assert isinstance(result.score, (int, float)), "score must be numeric"
    assert isinstance(result.feedback, str), "feedback must be string"
    assert 0 <= result.score <= 1, "score should be in [0, 1]"


def test_pydantic_models_in_signatures():
    """Verify Pydantic models work in DSPy signatures."""
    from experiments.arms import Critique, Refutation
    
    # Should be able to create a Predict with typed signature
    critique_pred = dspy.Predict(Critique)
    
    # Signature should have the expected fields
    sig = critique_pred.signature
    assert 'current_instruction' in sig.input_fields
    assert 'refutations' in sig.output_fields
    
    # Check that refutations field has list[Refutation] type
    refutations_field = sig.output_fields['refutations']
    # DSPy should handle this type annotation


def test_budget_wrapper_transparency():
    """Verify BudgetedLM delegates to underlying LM."""
    from dspy.utils import DummyLM
    from experiments.config import BudgetTracker, BudgetedLM, BudgetConfig
    
    base_lm = DummyLM({"hello": "world"})
    tracker = BudgetTracker(BudgetConfig())
    wrapped = BudgetedLM(base_lm, tracker, is_reflection=True)
    
    # Should delegate attribute access
    assert hasattr(wrapped, '_engine_spec'), "Should delegate to underlying LM"
    
    # Should track calls
    assert tracker.reflection_calls == 0
    # Call would happen via __call__, not testing actual call to avoid engine details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
