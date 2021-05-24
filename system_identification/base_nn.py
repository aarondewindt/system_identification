import re
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Union, Optional
from pathlib import Path

import numpy as np
from .utils.vdom import hyr


@dataclass
class TrainingParameters:
    epochs: float
    goal: float
    min_grad: float
    mu: float

    def _repr_html_(self):
        return hyr(title="TrainingParameters", root_type=type(self), content=asdict(self))


@dataclass
class TrainingLog:
    error: float

    def _repr_html_(self):
        return hyr(title="TrainingLog", root_type=type(self), content=asdict(self))


class BaseNeuralNetwork(ABC):
    name = "base_neural_network"
    activation_functions = tuple()
    training_algorithm = ""

    @abstractmethod
    def evaluate(self, inputs: np.ndarray):
        ...

    @abstractmethod
    def train(self,
              inputs: np.ndarray,
              reference_outputs: np.ndarray,
              epochs: int=None,
              method: str="",
              train_log_freq=1):
        ...

    @abstractmethod
    def evaluate_error(self, inputs, reference_outputs):
        ...

    @classmethod
    def saves_idxs(cls, log_dir: Union[Path, str]):
        log_dir = Path(log_dir)
        idxs = set()
        for path in log_dir.glob(f"{cls.name}_*.pickle"):
            if match := re.match(rf"{cls.name}_(\d+).pickle", path.name):
                idxs.add(int(match.group(1)))
        return sorted(idxs)

    @classmethod
    def load(cls, log_dir: Union[Path, str], idx: Optional[int] = None):
        if idx is None:
            if saves_idxs := cls.saves_idxs(log_dir):
                idx = saves_idxs[-1]
            else:
                return None

        log_dir = Path(log_dir)
        if (path := (log_dir / f"{cls.name}_{idx}.pickle")).exists():
            with open(path, "rb") as f:
                return pickle.load(f)

    def save(self, log_dir: Union[Path, str]):
        if saves_idxs := self.saves_idxs(log_dir):
            idx = saves_idxs[-1] + 1
        else:
            idx = 0

        with (log_dir / f"{self.name}_{idx}.pickle").open("wb") as f:
            pickle.dump(self, f)

    @abstractmethod
    def save_matlab(self):
        ...