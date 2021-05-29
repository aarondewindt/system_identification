from typing import Deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np
import pandas as pd
from .utils.vdom import hyr


@dataclass
class TrainingLog:
    epoch: int
    grad: float
    error: float

    def _repr_html_(self):
        return hyr(title="TrainingLog", root_type=type(self), content=asdict(self))


class BaseModel(ABC):
    name = "base_model"
    activation_functions = tuple()
    training_algorithm = ""

    def __init__(self):
        self._training_log: Deque[TrainingLog] = deque()

    @abstractmethod
    def evaluate(self, inputs: np.ndarray):
        ...

    @abstractmethod
    def train(self,
              inputs: np.ndarray,
              reference_outputs: np.ndarray,
              epochs: int=None,
              method: str="",
              goal=0,
              min_grad=1e-10,
              train_log_freq=1):
        ...

    def log(self, epoch, grad, error):
        self._training_log.append(TrainingLog(epoch, grad, error))

    @property
    def training_log(self):
        """
        `xarray.Dataset` of the training log.
        :return:
        """
        return pd.DataFrame([asdict(log_entry) for log_entry in self._training_log])

    @abstractmethod
    def save_matlab(self):
        ...

    def evaluate_error(self, inputs, reference_outputs):
        """
        Evaluated the model and calculates the absolute sum of the errors.

        :param inputs: Must have shape (n_samples, n_inputs, 1).
        :param reference_outputs: Must have shape (n_samples, n_outputs, 1). Expected ouputs
                                  for the inputs.
        :return: Absolute sum of the errors.
        """
        return np.sum(np.abs(reference_outputs - self.evaluate(inputs))) / inputs.shape[0]
