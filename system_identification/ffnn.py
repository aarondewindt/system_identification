from typing import Union, Sequence, Deque
from dataclasses import asdict
from pathlib import Path
from collections import deque

from tqdm.auto import trange
import numpy as np
import pandas as pd

from .base_nn import TrainingLog, TrainingParameters, BaseNeuralNetwork
from .utils.vdom import hyr


class FeedForwardNeuralNetwork(BaseNeuralNetwork):
    name = "feedforward"
    activation_functions = ("tansig", 'purelin')
    training_algorithm = "trainlm"

    def __init__(self,
                 weights_0: np.ndarray,
                 weights_1: np.ndarray,
                 range: np.ndarray,
                 training_parameters: TrainingParameters,
                 log_dir: Union[Path, str]):
        """
        Dense feedforward neural network.

        ..note: Do not use this constructor to create new FeedForwardNeuralNetwork instances. Use the
                FeedForwardNeuralNetwork.new(...) factory function instead.

        :param weights_0: (n_hidden x (n_input + 1)) matrix. Weights between the input and hidden layer.
        :param weights_1: (n_output x (n_hidden + 1)) matrix. Weights between the hidden and output layer.
        :param range: (n_input x 2) matrix with the range of each input in each row.
        :param training_parameters: Training parameters.
        :param log_dir: Directory to store network training history.
        """

        assert weights_0.shape[0] == (weights_1.shape[1] - 1)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.weights_0 = weights_0
        self.weights_1 = weights_1

        self.n_inputs = weights_0.shape[1] - 1
        self.n_hidden = weights_0.shape[0]
        self.n_outputs = weights_1.shape[0]

        # self.input_weights = input_weights
        # self.output_weights = output_weights
        # self.bias_weights_0 = bias_weights_0
        # self.bias_weights_1 = bias_weights_1
        self.range = range
        self.training_parameters = training_parameters

        self._training_log: Deque[TrainingLog] = deque()

    def __repr__(self):
        return f"<FeedForwardNeuralNetwork\n" \
               f" n_inputs={self.n_inputs}\n" \
               f" n_hidden={self.n_hidden}\n" \
               f" n_outputs={self.n_outputs}>"

    def _repr_html_(self):
        return hyr(title="Feed Forward Neural Network", root_type=type(self), content={
            "n_inputs": self.n_inputs,
            "n_hidden": self.n_hidden,
            "n_output": self.n_outputs,
            "weights_0": self.weights_0,
            "weights_1": self.weights_1,
            "input_weights": self.input_weights,
            "output_weights": self.output_weights,
            "bias_weights": [self.bias_weights_0, self.bias_weights_1],
            "training_parameters": self.training_parameters,
            "log_dir": self.log_dir,
        }).to_html()


    @property
    def input_weights(self):
        return self.weights_0[:, 1:]

    @property
    def output_weights(self):
        return self.weights_1[:, 1:]

    @property
    def bias_weights_0(self):
        return self.weights_0[:, 1, None]

    @property
    def bias_weights_1(self):
        return self.weights_1[:, 1, None]

    @classmethod
    def new(cls,
            n_inputs: float,
            n_outputs: float,
            n_hidden: float,
            range: Union[np.ndarray, Sequence[float]],
            training_parameters: TrainingParameters,
            log_dir: Union[Path, str]):
        """
        Factory function used to create new FeedForwardNeuralNetwork instances.

        :param n_inputs: Number of inputs
        :param n_outputs: Number of outputs
        :param n_hidden: Number of cells in the hidden layer.
        :param range: Range of each input.
        :param training_parameters: Training parameters.
        :param log_dir: Directory to store network training history.
        :return: New FeedForwardNeuralNetwork instance.
        """

        range = np.atleast_2d(range)
        assert range.shape == (n_inputs, 2), "There must be one row in range per input."

        return cls(
            weights_0=np.random.rand(n_hidden, n_inputs + 1),
            weights_1=np.random.rand(n_outputs, n_hidden + 1),
            range=range,
            training_parameters=training_parameters,
            log_dir=log_dir,
        )

    def evaluate(self, inputs: np.ndarray):
        """
        Shape of input must be: (n_samples, n_inputs, 1).
        Output shape will be: (n_samples, n_outputs, 1).

        :param input:
        :return:
        """
        # Evaluate hidden layer
        # Insert ones for bias input.
        inputs = np.insert(inputs, 0, 1, 1)
        hidden = np.tanh(self.weights_0 @ inputs)

        # Evaluate output layer
        hidden = np.insert(hidden, 0, 1, 1)
        return self.weights_1 @ hidden

    def train(self, inputs, reference_outputs, epochs=None, method="trainbp", train_log_freq=10, **kwargs):
        epochs = epochs or self.training_parameters.epochs

        inputs_with_bias = np.insert(inputs, 0, 1, 1)

        if method == "trainbp":
            train_function = self.back_propagation
        elif method == "trainlm":
            train_function = self.levenberg_marquardt
        else:
            raise ValueError(f"Unknown training method {method}.")

        for i in trange(epochs):
            train_function(inputs_with_bias, reference_outputs, **kwargs)

            if not (i % train_log_freq):
                self._training_log.append(TrainingLog(self.evaluate_error(inputs, reference_outputs)))

    def back_propagation(self,
                         inputs: Union[np.ndarray, Sequence[float]],
                         reference_outputs: Union[np.ndarray, Sequence[float]],
                         eta: float,
                         alpha: float
                         ):
        hidden_preact = self.weights_0 @ inputs  # Hidden layer value before activation
        hidden_posact = np.tanh(hidden_preact)  # Hidden layer value after activation.

        hidden_posact = np.insert(hidden_posact, 0, 1, 1)
        output = self.weights_1 @ hidden_posact

        errors = reference_outputs - output

        dcost_dweights_1 = np.einsum("qkm,qjm->kj", -errors, hidden_posact)

        # Hidden layer activation function derivative
        dvk_dwjk = (1 / np.cosh(hidden_preact)) ** 2

        dcost_dweights_0 = np.einsum("qkm,kj,qjm,qim->ji", -errors, self.weights_1[:, 1:], dvk_dwjk, inputs)

        n_samples = inputs.shape[0]
        self.weights_0 -= eta * dcost_dweights_0 / n_samples
        self.weights_1 -= eta * dcost_dweights_1 / n_samples

    def levenberg_marquardt(self,
                            inputs: np.ndarray,
                            reference_outputs: np.ndarray,
                            mu):
        # q: n_samples
        # i: n_input + 1
        # j: n_hidden (later n_hidden+1)
        # k: n_output

        # (q, j, 1) = (j, i) @ (q, i, 1)
        hidden_preact = self.weights_0 @ inputs  # Hidden layer value before activation. vj
        hidden_posact = np.tanh(hidden_preact)  # Hidden layer value after activation. yj

        hidden_posact = np.insert(hidden_posact, 0, 1, 1)  # Add bias, j is now (n_hidden + 1)
        # (q, k, 1) = (k, j) @ (q, j, 1)
        output = self.weights_1 @ hidden_posact

        errors = np.sum(reference_outputs - output, axis=(1))

        derror_dweights_1 = -hidden_posact[..., 0]  # (q, j)
        if self.n_outputs > 1:
            derror_dweights_1 = np.hstack([derror_dweights_1] * self.n_outputs)

        dvk_dwjk = (1 / np.cosh(hidden_preact)) ** 2  # (q, j, 1),  j is now n_hidden

        derror_dweights_0 = np.einsum("kj,qjm,qim->qji", -self.weights_1[:, 1:], dvk_dwjk, inputs)  # (q, j, i)
        derror_dweights_0 = derror_dweights_0.reshape((derror_dweights_0.shape[0], -1))  # (q, x)

        j = np.hstack((derror_dweights_0, derror_dweights_1))

        delta_weights = np.linalg.inv(j.T @ j + mu * np.eye(j.shape[1])) @ j.T @ errors

        delta_weights_0 = delta_weights[:(self.n_hidden * (self.n_inputs + 1))].reshape((self.n_hidden, self.n_inputs + 1))
        delta_weights_1 = delta_weights[(self.n_hidden * (self.n_inputs + 1)):].reshape((self.n_outputs, self.n_hidden + 1))

        self.weights_0 -= delta_weights_0
        self.weights_1 -= delta_weights_1

    def evaluate_error(self, inputs, reference_outputs):
        return np.sum(np.abs(reference_outputs - self.evaluate(inputs))) / inputs.shape[0]

    @property
    def training_log(self):
        return pd.DataFrame([asdict(log_entry) for log_entry in self._training_log])

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
