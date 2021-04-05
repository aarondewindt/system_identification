from typing import Tuple, Union, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import time
import pickle

from tqdm.auto import trange
import numpy as np


@dataclass
class TrainingParameters:
    epochs: float
    goal: float
    min_grad: float
    mu: float


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
        input = np.asarray(input)
        input = input.reshape(input.shape[0], -1)

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

        self.save()
        t_last_save = time()

        with trange(epochs) as progress_bar:
            for epoch_idx in progress_bar:
                derror_biasweight_0 = np.zeros((self.n_hidden, 1))
                for input, reference_output in zip(inputs, reference_outputs):

                    input = input.reshape(input.shape[0], -1)
                    reference_output = reference_output.reshape(reference_output.shape[0], -1)

                    hidden_input = self.input_weights @ input + self.bias_weights[0]
                    hidden_output = np.tanh(hidden_input)
                    output = self.output_weights @ hidden_output + self.bias_weights[1]

                    errors = reference_output - output

                    derror_biasweight_1 = -errors
                    derror_dow = -errors @ hidden_output.T

                    dvk_dwjk = (1 / np.cosh(hidden_input)) ** 2

                    for j, k in product(range(self.n_hidden), range(self.n_output)):
                        derror_biasweight_0[j] = -errors[k] * self.output_weights[k, j] * dvk_dwjk[j]

                    derror_diw = derror_biasweight_0 @ input.T

                    delta_bias_weights_0 = -self.training_parameters.mu * derror_biasweight_0
                    self.bias_weights[0] += delta_bias_weights_0
                    delta_input_weights = -self.training_parameters.mu * derror_diw
                    self.input_weights += delta_input_weights

                    delta_bias_weights_1 = -self.training_parameters.mu * derror_biasweight_1
                    self.bias_weights[1] += delta_bias_weights_1

                    delta_output_weights = -self.training_parameters.mu * derror_dow
                    self.output_weights += delta_output_weights

                    if (time() - t_last_save) > dt_save:
                        t_last_save = time()
                        self.save()

                progress_bar.set_postfix(msg=f"{np.sum(delta_input_weights):5.2e} {np.sum(delta_output_weights):5.2e}")
