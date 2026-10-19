"""HoVer pilot interface; real data and retrieval are not implemented."""

import dspy

from experiments.theory_driven.benchmarks.base import BaseBenchmark, Splits


class HoVerBenchmark(BaseBenchmark):
    def __init__(self, name: str = "hover"):
        super().__init__(name)

    def setup(self) -> tuple[bool, str]:
        return (
            False,
            "HoVer is pilot-only; real data and retrieval are not implemented.",
        )

    def load_data(
        self,
        seed: int = 0,
        train_size: int = 20,
        val_size: int = 20,
        test_size: int = 50,
    ) -> Splits:
        raise NotImplementedError(
            "HoVer is pilot-only; no synthetic benchmark data is supplied"
        )

    def get_program(self) -> dspy.Module:
        raise NotImplementedError("HoVer is pilot-only; retrieval is not implemented")

    def get_metric(self):
        def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            expected = set(getattr(gold, "gold_passages", []))
            retrieved = set(getattr(pred, "retrieved_passages", []))
            found = expected & retrieved
            score = len(found) / len(expected) if expected else 0.0
            return dspy.Prediction(
                score=float(score),
                feedback=f"Found: {sorted(found)}. Missing: {sorted(expected - retrieved)}.",
            )

        return metric
