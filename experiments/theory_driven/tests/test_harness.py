"""Runtime failures and held-out accounting are not converted into successes."""
import dspy
from dspy.utils import DummyLM

from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.harness import ExperimentHarness


class FixtureBenchmark:
    def setup(self):
        return True, "offline fixture"

    def load_data(self, seed, train_size, val_size, test_size):
        self.test_size = test_size
        return tuple([dspy.Example(question=f"{split}{i}", answer="gold").with_inputs("question")
                      for i in range(size)]
                     for split, size in zip("tve", (train_size, val_size, test_size), strict=True))

    def get_program(self):
        return dspy.Predict("question -> answer")

    def get_metric(self):
        return lambda gold, pred, *args: dspy.Prediction(score=float(gold.answer == pred.answer), feedback="gold")


def test_compile_failure_keeps_test_mean_none(monkeypatch, tmp_path):
    lms = {role: DummyLM([{"answer": "gold"}]) for role in ("task", "reflection", "critic")}
    benchmark = FixtureBenchmark()
    harness = ExperimentHarness(ExperimentConfig(train_size=1, val_size=1, test_size=2,
                                                 results_dir=str(tmp_path)),
                                lm_factory=lambda config, seed: lms,
                                benchmark_factory=lambda name: benchmark)

    class FailingArm:
        def compile(self, **kwargs):
            raise RuntimeError("intentional compile failure")

        def get_stats(self):
            return {}

    monkeypatch.setattr(harness, "create_arm", lambda *args, **kwargs: FailingArm())
    result = harness.run_single("hotpotqa", "A", 3, 9)
    assert result["status"] == "error"
    assert result["test_mean"] is None
    assert result["test_scores"] == []
    assert benchmark.test_size == 2
    assert result["budget_stats"]["total_units"] == 0
    assert "intentional compile failure" in result["message"]


def test_last_evaluation_cost_overrun_is_not_success(monkeypatch, tmp_path):
    lms = {role: DummyLM([]) for role in ("task", "reflection", "critic")}
    harness = ExperimentHarness(ExperimentConfig(train_size=1, val_size=1, test_size=1,
                                                 results_dir=str(tmp_path)),
                                lm_factory=lambda config, seed: lms,
                                benchmark_factory=lambda name: FixtureBenchmark())

    class CostlyEvaluationArm:
        def compile(self, **kwargs):
            def evaluate(**inputs):
                lms["task"].history.append({"cost": 0.2})
                return dspy.Prediction(answer="gold")
            return evaluate

        def get_stats(self):
            return {}

    monkeypatch.setattr(harness, "create_arm", lambda *args, **kwargs: CostlyEvaluationArm())
    result = harness.run_single("hotpotqa", "A", 3, 9, max_cost=0.1)
    assert result["status"] == "cost_limit_exceeded"
    assert result["test_mean"] is None
    assert result["budget_stats"]["test_calls"] == 1
    assert result["budget_stats"]["cost_overshoot"] == 0.1


def test_lm_construction_disables_cache_for_each_role(monkeypatch):
    from experiments.theory_driven.harness import create_lms
    from experiments.theory_driven.config import ModelConfig
    calls = []

    def constructor(**kwargs):
        calls.append(kwargs)
        return DummyLM([])

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")
    monkeypatch.setattr(dspy, "LM", constructor)
    lms = create_lms(ModelConfig(), 17)
    assert len(lms) == len(calls) == 3
    assert all(call["cache"] is False and call["seed"] == 17 for call in calls)
    assert all(call["num_retries"] == 0 for call in calls)  # Guard owns 429 retries, not DSPy.
    assert all(call["extra_body"]["reasoning"] == {"enabled": False} for call in calls)
    assert len({id(lm) for lm in lms.values()}) == 3
