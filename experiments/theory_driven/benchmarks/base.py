"""Shared benchmark interface for stage 0 and explicitly unavailable pilots."""

from abc import ABC, abstractmethod
from typing import Callable

import dspy

Splits = tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]


class BaseBenchmark(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load_data(
        self,
        seed: int = 0,
        train_size: int = 20,
        val_size: int = 20,
        test_size: int = 50,
    ) -> Splits:
        """Return disjoint seeded train/validation/test examples."""
        raise NotImplementedError

    @abstractmethod
    def get_program(self) -> dspy.Module:
        raise NotImplementedError

    @abstractmethod
    def get_metric(self) -> Callable[..., dspy.Prediction]:
        """Return metric(gold, pred, trace=None, pred_name=None, pred_trace=None)."""
        raise NotImplementedError

    @abstractmethod
    def setup(self) -> tuple[bool, str]:
        raise NotImplementedError
