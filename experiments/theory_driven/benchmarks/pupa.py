"""PUPA pilot interface; real data and judge are not implemented."""

import dspy

from experiments.theory_driven.benchmarks.base import BaseBenchmark, Splits


class PUPABenchmark(BaseBenchmark):
    def __init__(self, name: str = "pupa"):
        super().__init__(name)

    def setup(self) -> tuple[bool, str]:
        return False, "PUPA is pilot-only; real data and judge are not implemented."

    def load_data(
        self,
        seed: int = 0,
        train_size: int = 20,
        val_size: int = 20,
        test_size: int = 50,
    ) -> Splits:
        raise NotImplementedError(
            "PUPA is pilot-only; no synthetic benchmark data is supplied"
        )

    def get_program(self) -> dspy.Module:
        raise NotImplementedError("PUPA is pilot-only; task program is not implemented")

    def get_metric(self):
        """Offline exact-match diagnostic, not the unimplemented PUPA judge."""

        def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            score = float(gold.answer.strip() == getattr(pred, "answer", "").strip())
            return dspy.Prediction(
                score=score,
                feedback=f"Exact-match diagnostic only. Gold answer: {gold.answer}.",
            )

        return metric
