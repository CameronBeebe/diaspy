"""Policy settings must reach the actual arm constructor through the harness."""
import pytest
from dspy.utils import DummyLM
from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.config.budget_tracker import UnifiedBudget
from experiments.theory_driven.harness import ExperimentHarness


@pytest.mark.parametrize("arm_name", ["B", "M"])
def test_policy_reaches_arm_constructor(arm_name, tmp_path):
    config = ExperimentConfig(results_dir=str(tmp_path))
    harness = ExperimentHarness(config)
    budget = UnifiedBudget(50, 1.0, DummyLM([]))
    arm = harness.create_arm(arm_name, budget, survival_policy="veto", min_survival_questions=5)
    assert getattr(arm, "survival_policy") == "veto"
    assert getattr(arm, "min_survival_questions") == 5
    with pytest.raises(ValueError):
        harness.create_arm(arm_name, budget, survival_policy="unknown")


def test_configured_defender_is_guarded_and_budgeted(tmp_path, monkeypatch):
    import dspy
    from experiments.theory_driven.tests.test_harness import FixtureBenchmark
    config = ExperimentConfig(train_size=1, val_size=1, test_size=1, results_dir=str(tmp_path))
    defender = DummyLM([{"answer": "defense"}])
    next(a for a in config.arms if a.name == "M").kwargs["defender_lm"] = defender
    lms = {role: DummyLM([{"answer": "gold"}]) for role in ("task", "reflection", "critic")}
    harness = ExperimentHarness(config, lm_factory=lambda *args: lms,
                                benchmark_factory=lambda name: FixtureBenchmark())
    from experiments.theory_driven.arms.millian import MillianArm
    def compile(self, program, defender_lm=None, **kwargs):
        assert defender_lm is not None
        assert hasattr(defender_lm, "runtime")
        with dspy.context(lm=defender_lm):
            dspy.Predict("question -> answer")(question="defend")
        return program
    monkeypatch.setattr(MillianArm, "compile", compile)
    result = harness.run_single("hotpotqa", "M", 50, 0)
    assert result["status"] == "success", result["message"]
    assert result["budget_stats"]["role_calls"]["defender"] == 1
    assert result["budget_stats"]["total_units"] == 2


def test_initialization_metadata_uses_resolved_arm_options(tmp_path, monkeypatch):
    import json
    from experiments.theory_driven.tests.test_harness import FixtureBenchmark
    config = ExperimentConfig(train_size=1, val_size=1, test_size=1, results_dir=str(tmp_path))
    next(a for a in config.arms if a.name == "B").kwargs.update(
        survival_policy="veto", min_survival_questions=0)
    harness = ExperimentHarness(config, lm_factory=lambda *args: {
        role: DummyLM([]) for role in ("task", "reflection", "critic")
    }, benchmark_factory=lambda name: FixtureBenchmark())
    def fail_compile(*args, **kwargs):
        raise RuntimeError("stop after construction")
    from experiments.theory_driven.arms import PopperianArm
    monkeypatch.setattr(PopperianArm, "compile", fail_compile)
    result = harness.run_single("hotpotqa", "B", 50, 0)
    event = json.loads((tmp_path / result["artifacts"]["events"]).read_text().splitlines()[0])
    assert event["metadata"]["survival_policy"] == "veto"
    assert event["metadata"]["min_survival_questions"] == 0
