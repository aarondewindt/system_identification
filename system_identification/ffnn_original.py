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
    delta_sum: float


class FeedForwardNeuralNetwork:
    def __init__(self,
                 input_weights: np.ndarray,
                 output_weights: np.ndarray,
                 bias_weights: Tuple[np.ndarray, np.ndarray],
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
        self.bias_weights = list(bias_weights)
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
            bias_weights=(
                np.random.rand(n_hidden, 1),
                np.random.rand(n_outputs, 1)
            ),
            range=range,
            training_parameters=training_parameters,
            log_dir=log_dir,
        )

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
                "bias_weights0": self.bias_weights[0].tolist(),
                "bias_weights1": self.bias_weights[1].tolist(),
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

    def evaluate(self, input: Union[np.ndarray, Sequence[float]]):
        # Make sure the input is a numpy column vector
        input = np.asarray(input)[:, None]

        # Evaluate neural network
        hidden = np.tanh(self.input_weights @ input + self.bias_weights[0])
        return self.output_weights @ hidden + self.bias_weights[1]

    def back_propagation(self,
                         inputs: Union[np.ndarray, Sequence[float]],
                         reference_outputs: Union[np.ndarray, Sequence[float]],
                         epochs=None,
                         dt_save=60):
        epochs = epochs or self.training_parameters.epochs

        inputs = np.atleast_2d(inputs)
        reference_outputs = np.atleast_2d(reference_outputs)

        assert inputs.shape[0] == reference_outputs.shape[0]

        n_samples = inputs.shape[0]

        self.save()
        t_last_save = time()

        with trange(epochs) as progress_bar:
            for epoch_idx in progress_bar:
                derror_biasweight_0 = np.zeros((self.n_hidden, 1))

                epoch_error = 0
                epoch_delta_bias_weights_0 = np.zeros((self.n_hidden, 1))
                epoch_delta_bias_weights_1 = np.zeros((self.n_output, 1))
                epoch_delta_input_weights = np.zeros(self.input_weights.shape)
                epoch_delta_output_weights = np.zeros(self.output_weights.shape)

                for input, reference_output in zip(inputs, reference_outputs):
                    input = input[:, None]
                    reference_output = reference_output[:, None]

                    update_error, delta_bias_weights_0, delta_input_weights, \
                    delta_bias_weights_1, delta_output_weights = calc_deltas(
                        input,
                        reference_output,
                        self.training_parameters.mu,
                        self.bias_weights[0],
                        self.bias_weights[1],
                        self.input_weights,
                        self.output_weights)

                    epoch_error += update_error
                    epoch_delta_bias_weights_0 += delta_bias_weights_0
                    epoch_delta_bias_weights_1 += delta_bias_weights_1
                    epoch_delta_input_weights += delta_input_weights
                    epoch_delta_output_weights += delta_output_weights

                epoch_delta = \
                    np.sum(abs(delta_bias_weights_0)) \
                    + np.sum(abs(delta_bias_weights_1)) \
                    + np.sum(abs(delta_input_weights)) \
                    + np.sum(abs(delta_output_weights))

                self.bias_weights[0] += epoch_delta_bias_weights_0 / n_samples
                self.bias_weights[1] += epoch_delta_bias_weights_1 / n_samples
                self.input_weights += epoch_delta_input_weights / n_samples
                self.output_weights += epoch_delta_output_weights / n_samples

                if (time() - t_last_save) > dt_save:
                    t_last_save = time()
                    self.save()

                self._training_log.append(TrainingLog(epoch_error, epoch_delta))
                progress_bar.set_postfix(msg=f"{np.sum(delta_input_weights):5.2e} {np.sum(delta_output_weights):5.2e}")

        self.save()

    def evaluate_error(self, inputs, reference_outputs):
        inputs = np.atleast_2d(inputs)
        reference_outputs = np.atleast_2d(reference_outputs)
        error = 0
        for input, reference_output in zip(inputs, reference_outputs):
            output = self.evaluate(input)
            error += np.sum(abs(reference_output - output))
        return error

    @property
    def training_log(self):
        return pd.DataFrame([asdict(log_entry) for log_entry in self._training_log])


@jit(nopython=True)
def calc_deltas(input, reference_output, mu,
                bias_weights_0, bias_weights_1, input_weights, output_weights):
    hidden_input = input_weights @ input + bias_weights_0
    hidden_output = np.tanh(hidden_input)
    output = output_weights @ hidden_output + bias_weights_1

    errors = reference_output - output
    update_error = np.sum(np.abs(errors))

    derror_biasweight_1 = -errors
    derror_dow = -errors @ hidden_output.T

    # Hidden layer activation function derivative
    dvk_dwjk = (1 / np.cosh(hidden_input)) ** 2

    n_hidden = input_weights.shape[0]
    n_output = output_weights.shape[0]
    derror_biasweight_0 = np.zeros((n_hidden, 1))
    for j in range(n_hidden):
        for k in range(n_output):
            derror_biasweight_0[j] = -errors[k] * output_weights[k, j] * dvk_dwjk[j]

    derror_diw = derror_biasweight_0 @ input.T

    delta_bias_weights_0 = -mu * derror_biasweight_0
    delta_input_weights = -mu * derror_diw
    delta_bias_weights_1 = -mu * derror_biasweight_1
    delta_output_weights = -mu * derror_dow

    return update_error, delta_bias_weights_0, delta_input_weights, delta_bias_weights_1, delta_output_weights