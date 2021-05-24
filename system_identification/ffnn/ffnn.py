from typing import Tuple, Union, Sequence, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from time import time
import pickle
import re

from tqdm.auto import trange
from numba import jit
import numpy as np
import pandas as pd


@dataclass
class TrainingParameters:
    epochs: float
    goal: float
    min_grad: float
    mu: float


@dataclass
class TrainingLog:
    error: float


class FeedForwardNeuralNetwork:
    def __init__(self,
                 input_weights: np.ndarray,
                 output_weights: np.ndarray,
                 bias_weights_0: np.ndarray,
                 bias_weights_1: np.ndarray,
                 range: np.ndarray,
                 training_parameters: TrainingParameters,
                 log_dir: Union[Path, str],
                 ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.name = "feedforward"
        self.activation_functions = ("tansig", 'purelin')
        self.training_algorithm = "trainlm"

        self.n_inputs = input_weights.shape[1]
        self.n_hidden = input_weights.shape[0]
        self.n_output = output_weights.shape[0]

        self.input_weights = input_weights
        self.output_weights = output_weights
        self.bias_weights_0 = bias_weights_0
        self.bias_weights_1 = bias_weights_1
        self.range = range
        self.training_parameters = training_parameters

        self._training_log: List[TrainingLog] = []

    def __repr__(self):
        return f"<FeedForwardNeuralNetwork\n" \
               f" n_inputs={self.n_inputs}\n" \
               f" n_hidden={self.n_hidden}\n" \
               f" n_output={self.n_output}>"

    @classmethod
    def new(cls,
            n_inputs: float,
            n_outputs: float,
            n_hidden: float,
            range: Union[np.ndarray, Sequence[float]],
            training_parameters: TrainingParameters,
            log_dir: Union[Path, str]):

        range = np.atleast_2d(range)
        assert range.shape == (n_inputs, 2)

        return cls(
            input_weights=np.random.rand(n_hidden, n_inputs),
            output_weights=np.random.rand(n_outputs, n_hidden),
            bias_weights_0=np.random.rand(n_hidden, 1),
            bias_weights_1=np.random.rand(n_outputs, 1),
            range=range,
            training_parameters=training_parameters,
            log_dir=log_dir,
        )

    @classmethod
    def saves_idxs(cls, log_dir: Union[Path, str], name="feedforward"):
        log_dir = Path(log_dir)
        idxs = set()
        for path in log_dir.glob(f"feedforward_*.pickle"):
            if match := re.match(rf"feedforward_(\d+).pickle", path.name):
                idxs.add(int(match.group(1)))
        return sorted(idxs)

    @classmethod
    def load(cls, log_dir: Union[Path, str], idx: Optional[int]=None):
        if idx is None:
            saves_idxs = cls.saves_idxs(log_dir)
            if len(saves_idxs):
                idx = saves_idxs[-1]
            else:
                return None

        log_dir = Path(log_dir)
        if (path := (log_dir / f"feedforward_{idx}.pickle")).exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            nn = cls(
                input_weights=np.array(data['input_weights']),
                output_weights=np.array(data['output_weights']),
                bias_weights_0=np.array(data['bias_weights_0']),
                bias_weights_1=np.array(data['bias_weights_1']),
                range=np.array(data['range']),
                training_parameters=data['training_parameters'],
                log_dir=log_dir)
            nn._training_log = data['training_log']
            return nn

    def save(self):
        # Look for available index
        idx = 0
        path = self.log_dir / f"{self.name}_{idx}.pickle"
        while path.exists():
            idx += 1
            path = self.log_dir / f"{self.name}_{idx}.pickle"

        with open(path, "wb") as f:
            pickle.dump({
                "input_weights": self.input_weights.tolist(),
                "output_weights": self.output_weights.tolist(),
                "bias_weights0": self.bias_weights_0.tolist(),
                "bias_weights1": self.bias_weights_1.tolist(),
                "range": self.range.tolist(),
                "training_parameters": self.training_parameters,
                "training_log": self._training_log
            }, f)

    def save_matlab(self):
        mdict = {
            "name": self.name,
            "trainFunct": self.activation_functions,
            "trainAlg": self.training_algorithm,

            "IW": self.input_weights,
            "LW": self.output_weights,
            "b": self.bias_weights,
            "range": self.range,
            "trainParam": {
                "epochs": self.training_parameters.epochs,
                "goal": self.training_parameters.goal,
                "min_grad": self.training_parameters.min_grad,
                "mu": self.training_parameters.mu,
            },
        }

    def evaluate(self, input: np.ndarray):
        """
        Shape of input must be: (n_samples, n_inputs, 1).
        Output shape will be: (n_samples, n_outputs, 1).

        :param input:
        :return:
        """
        # Evaluate neural network
        hidden = np.tanh(self.input_weights @ input + self.bias_weights_0)
        return self.output_weights @ hidden + self.bias_weights_1

    def train(self, inputs, reference_outputs, epochs=None, method="back_propagation", train_log_freq=10):
        epochs = epochs or self.training_parameters.epochs

        if method == "back_propagation":
            train_function = self.back_propagation
        else:
            raise ValueError(f"Unknown training method {method}.")

        for i in trange(epochs):
            train_function(inputs, reference_outputs)

            if i % train_log_freq:
                self._training_log.append(TrainingLog(self.evaluate_error(inputs, reference_outputs)))

    def back_propagation(self,
                         inputs: Union[np.ndarray, Sequence[float]],
                         reference_outputs: Union[np.ndarray, Sequence[float]]):
        mu = self.training_parameters.mu

        hidden_input = self.input_weights @ inputs + self.bias_weights_0
        hidden_output: np.ndarray = np.tanh(hidden_input)
        output = self.output_weights @ hidden_output + self.bias_weights_1

        errors = reference_outputs - output

        derror_dbiasweight_1 = np.sum(-errors, 0)
        derror_dow = np.einsum("soj,shj->oh", -errors, hidden_output)

        # Hidden layer activation function derivative
        dvk_dwjk = (1 / np.cosh(hidden_input)) ** 2

        derror_vj = np.einsum("soj,oh,shj->shj", -errors, self.output_weights, dvk_dwjk)
        derror_biasweight_0 = np.sum(derror_vj, 0)
        derror_diw = np.einsum("shj,sij->hi", derror_vj, inputs)

        n_samples = inputs.shape[0]
        self.bias_weights_0 -= mu * derror_biasweight_0 / n_samples
        self.bias_weights_1 -= mu * derror_dbiasweight_1 / n_samples
        self.input_weights -= mu * derror_diw / n_samples
        self.output_weights -= mu * derror_dow / n_samples

    def evaluate_error(self, inputs, reference_outputs):
        return np.sum(np.abs(reference_outputs - self.evaluate(inputs)))

    @property
    def training_log(self):
        return pd.DataFrame([asdict(log_entry) for log_entry in self._training_log])
