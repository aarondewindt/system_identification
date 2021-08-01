from typing import Deque, Optional, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from functools import cached_property
import statsmodels.stats.api as sms

import numpy as np
import xarray as xr
from .utils.vdom import hyr


@dataclass
class TrainingLog:
    epoch: int
    grad: float
    error_training_data: np.ndarray
    error_validation_data: np.ndarray
    weights: Dict[str, np.ndarray]

    def _repr_html_(self):
        return hyr(title="TrainingLog", root_type=type(self), content=asdict(self))


class BaseModel(ABC):
    name = "base_model"
    activation_functions = tuple()
    training_algorithm = ""

    def __init__(self, n_inputs, n_outputs, input_range, description):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input_range = input_range
        self.description = description
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
              train_log_freq=1,
              validation_inputs: Optional[np.ndarray] = None,
              validation_outputs: Optional[np.ndarray] = None,
              ):
        ...

    def log(self, epoch, grad, error_training_data, error_validation_data, weights):
        # Clear cached log dataset if necessary
        if 'training_log' in self.__dict__:
            del self.__dict__['training_log']

        self._training_log.append(
            TrainingLog(epoch, grad, error_training_data, error_validation_data, weights)
        )

    @cached_property
    def training_log(self):
        """
        `xarray.Dataset` of the training log with the results of Jarque-Bera tests
        on the training and validation data errors during each epoch.

        :return:
        """
        epoch = deque()
        gradient = deque()
        error_training_data = deque()
        error_validation_data = deque()
        error_training_jb = deque()
        error_training_jbp = deque()
        error_training_skew = deque()
        error_training_kurt = deque()
        error_validation_jb = deque()
        error_validation_jbp = deque()
        error_validation_skew = deque()
        error_validation_kurt = deque()

        weights = defaultdict(deque)

        for log_entry in self._training_log:
            epoch.append(log_entry.epoch)
            gradient.append(log_entry.grad)
            error_training_data.append(log_entry.error_training_data)
            error_validation_data.append(log_entry.error_validation_data)

            jb, jbp, skew, kurt = sms.jarque_bera(log_entry.error_training_data.squeeze())
            error_training_jb.append(jb)
            error_training_jbp.append(jbp)
            error_training_skew.append(skew)
            error_training_kurt.append(kurt)

            for key, value in log_entry.weights.items():
                weights[key].append(value)

            if log_entry.error_validation_data.size > 1:
                jb, jbp, skew, kurt = sms.jarque_bera(log_entry.error_validation_data.squeeze())
                error_validation_jb.append(jb)
                error_validation_jbp.append(jbp)
                error_validation_skew.append(skew)
                error_validation_kurt.append(kurt)
            else:
                error_validation_jb.append(np.nan)
                error_validation_jbp.append(np.nan)
                error_validation_skew.append(np.nan)
                error_validation_kurt.append(np.nan)

        # Xarray requires all dimensions to be named, so I'm using this iterator
        # to get names for the unnamed dimensions.
        names = iter("ijklmnopqrstuvwxyz")

        return xr.Dataset(
            data_vars={
                "gradient": (("epoch",), gradient,),
                "error_training_data": (("epoch", next(names)), np.array(error_training_data)[..., 0, 0],),
                "error_validation_data": (("epoch", next(names)), np.array(error_validation_data)[..., 0, 0],),
                "error_training_jb": (("epoch",), error_training_jb),
                "error_training_jbp": (("epoch",), error_training_jbp),
                "error_training_skew": (("epoch",), error_training_skew),
                "error_training_kurt": (("epoch",), error_training_kurt),
                "error_validation_jb": (("epoch",), error_validation_jb),
                "error_validation_jbp": (("epoch",), error_validation_jbp),
                "error_validation_skew": (("epoch",), error_validation_skew),
                "error_validation_kurt": (("epoch",), error_validation_kurt),
                **{
                    key: (("epoch", *[next(names) for _ in range(value[0].ndim)]), value)
                    for key, value in weights.items()
                }
            },
            coords={"epoch": epoch},
        )

    @abstractmethod
    def save_matlab(self):
        ...

    def evaluate_errors(self, training_inputs, training_outputs, validation_inputs, validation_outputs):
        """
        Evaluated the model and calculates the absolute sum of the errors.

        :param training_inputs: Must have shape (n_samples, n_inputs, 1).
        :param training_outputs: Must have shape (n_samples, n_outputs, 1). Expected ouputs
                                  for the inputs.
        :return: Absolute sum of the errors.
        """

        training_data_errors = training_outputs - self.evaluate(training_inputs)

        if (validation_inputs is not None) and (validation_outputs is not None):
            validation_data_errors = validation_outputs - self.evaluate(validation_inputs)
        else:
            validation_data_errors = np.array((((np.nan,),),))

        return training_data_errors, validation_data_errors

