from typing import Union, Sequence, Deque, Optional
from dataclasses import asdict
from pathlib import Path

from tqdm.auto import trange
import numpy as np
import pandas as pd

from .base_model import BaseModel
from .utils.vdom import hyr


class FeedForwardNeuralNetwork(BaseModel):
    name = "feedforward"
    activation_functions = ("tansig", 'purelin')

    def __init__(self,
                 weights_0: np.ndarray,
                 weights_1: np.ndarray,
                 range: np.ndarray,
                 log_dir: Union[Path, str],
                 description: str):
        """
        Dense feedforward neural network.

        ..note: Do not use this constructor to create new FeedForwardNeuralNetwork instances. Use the
                FeedForwardNeuralNetwork.new(...) factory function instead.

        :param weights_0: (n_hidden x (n_input + 1)) matrix. Weights between the input and hidden layer.
        :param weights_1: (n_output x (n_hidden + 1)) matrix. Weights between the hidden and output layer.
        :param range: (n_input x 2) matrix with the range of each input in each row.
        :param log_dir: Directory to store network training history.
        """

        super().__init__(
            n_inputs=weights_0.shape[1] - 1,
            n_outputs=weights_1.shape[0],
            range=range,
            description=description,
        )

        assert weights_0.shape[0] == (weights_1.shape[1] - 1)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.weights_0 = weights_0  #: Input weights, including bias weights on in the first column.
        self.weights_1 = weights_1  #: Outputs weights, including bias weights on in the first column.

        self.n_hidden = weights_0.shape[0]
        self.range = range

        self.epochs = 0

        self.training_parameters = {
            "training_algorithm": None,
            "goal": None,
            "min_grad": None,
        }

    def __repr__(self):
        return f"<FeedForwardNeuralNetwork\n" \
               f" n_inputs={self.n_inputs}\n" \
               f" n_hidden={self.n_hidden}\n" \
               f" n_outputs={self.n_outputs}>"

    def _repr_html_(self):
        return hyr(title="Feed Forward Neural Network", root_type=type(self), content={
            "description": self.description,
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
            n_inputs: int,
            n_outputs: int,
            n_hidden: int,
            range: Union[np.ndarray, Sequence[float]],
            log_dir: Union[Path, str]):
        """
        Factory function used to create new FeedForwardNeuralNetwork instances.

        :param n_inputs: Number of inputs
        :param n_outputs: Number of outputs
        :param n_hidden: Number of cells in the hidden layer.
        :param range: Range of each input.
        :param log_dir: Directory to store network training history.
        :return: New FeedForwardNeuralNetwork instance.
        """

        range = np.atleast_2d(range)
        assert range.shape == (n_inputs, 2), "There must be one row in range per input."

        return cls(
            weights_0=np.random.rand(n_hidden, n_inputs + 1),
            weights_1=np.random.rand(n_outputs, n_hidden + 1),
            range=range,
            log_dir=log_dir,
            description="Feedforward neural network with random initial weights."
        )

    def evaluate(self, inputs: np.ndarray):
        """
        Shape of input must be: (n_samples, n_inputs, 1).
        Output shape will be: (n_samples, n_outputs, 1).

        :param inputs: Inputs to evaluate.
        :return:
        """
        # Evaluate hidden layer
        # Insert ones for bias input.
        inputs = np.insert(inputs, 0, 1, 1)
        hidden = np.tanh(self.weights_0 @ inputs)

        # Evaluate output layer
        hidden = np.insert(hidden, 0, 1, 1)
        return self.weights_1 @ hidden

    def train(self,
              inputs: np.ndarray,
              reference_outputs: np.ndarray,
              epochs: int=1,
              method: str="trainlm",
              goal: float=0.,
              min_grad: float=1e-10,
              train_log_freq: int=1,
              evaluation_inputs: Optional[np.ndarray]=None,
              evaluation_reference_outputs: Optional[np.ndarray]=None,
              **kwargs):
        """
        Train feedforward neural network.

        :param inputs: Training input data. Must have shape (n_samples, n_inputs, 1).
        :param reference_outputs: Training output data. Must have shape (n_samples, n_outputs, 1).
        :param epochs: Maximum number of training epochs.
        :param method: Training algorithm. Valid values are `trainbp` for back propagation and `trainlm`
                       for Levenberg Marquardt.
        :param goal: Target maximum error. The training will stop once the evaluated error is under the goal.
        :param min_grad: Minimum gradient. The training will stop once the absolute sum gradients is below
                         this value.
        :param train_log_freq: Number of epochs between error evaluation and logging steps.
        :param evaluation_inputs: Optional evaluation input data. By default the training data is used.
        :param evaluation_reference_outputs: Optional evaluation input data. By default the training data is used.
        :param kwargs: Extra parameters passed to the training algorith functions.
                       Both accept `alpha`. `trainbp` accepts `eta` and `trainlm` accepts `mu`. `alpha` may be
                       set to `None` to make it non-adaptive.
        """

        assert (evaluation_inputs is None) == (evaluation_reference_outputs is None), \
            "Both the evaluation inputs and reference outputs must be given or omitted."

        evaluation_inputs = evaluation_inputs if evaluation_inputs is None else inputs
        evaluation_reference_outputs = evaluation_reference_outputs if evaluation_reference_outputs is None else reference_outputs

        inputs = np.insert(inputs, 0, 1, 1)

        if method == "trainbp":
            if "eta" not in kwargs:
                raise ValueError("Parameter `eta` required for method `trainbp`.")
            if "alpha" not in kwargs:
                raise ValueError("Parameter `alpha` required for method `trainbp`.")
            train_function = self._back_propagation

        elif method == "trainlm":
            if "mu" not in kwargs:
                raise ValueError("Parameter `mu` required for method `trainlm`.")
            train_function = self._levenberg_marquardt

        else:
            raise ValueError(f"Unknown training method {method}.")

        if self.training_parameters["training_algorithm"] is None:
            self.training_parameters["training_algorithm"] = method
        else:
            if self.training_parameters["training_algorithm"] != method:
                raise ValueError("Changing training algorithm on trained model not allowed.")

        self.training_parameters["goal"] = goal
        self.training_parameters["min_grad"] = min_grad
        self.training_parameters["epochs"] = epochs
        self.training_parameters.update(kwargs)

        for i, grad in zip(trange(epochs - self.epochs), train_function(inputs, reference_outputs, **kwargs)):
            if self.epochs >= epochs:
                break

            self.epochs += 1

            if grad <= self.training_parameters["min_grad"]:
                break

            if not (i % train_log_freq):
                error = self.evaluate_error(evaluation_inputs, evaluation_reference_outputs)
                if error < self.training_parameters["goal"]:
                    break

                self.log(self.epochs, grad, error)

    def _back_propagation(self,
                          inputs: np.ndarray,
                          reference_outputs: np.ndarray,
                          eta: float,
                          alpha: Optional[float],
                          **_
                          ):
        """
        Generator that performs one backpropagation step on each iteration.

        ..note: Do not call this function directly. Use train(...) instead.

        :param inputs: Must have shape (n_samples, n_input + 1, 1). It's assumed
                       that the bias input is already included.
        :param reference_outputs: Outputs for each input. Must have shape (n_samples, n_output, 1)
        :param eta: Learning rate.
        :param alpha: Adaptive factor. If None, then the learning rate will not be adapted.
        :yields: Absolute sum of the gradients on each iteration.
        """
        # q: n_samples
        # i: n_input + 1
        # j: n_hidden (later n_hidden+1)
        # k: n_output
        # m: 1

        last_error = None

        # Infinite loop. This is a generator that will indefinitely update the
        # weights on each iteration. The train(...) function handles the end
        # conditions.
        while True:
            # Evaluate network to get the hidden layer, output and error values.
            # ==================================================================

            # (q, j, 1) = (j, i) @ (q, i, 1)   # j is n_hidden
            hidden_preact = self.weights_0 @ inputs  # Hidden layer value before activation
            hidden_posact = np.tanh(hidden_preact)  # Hidden layer value after activation.

            # (q, k, 1) = (k, j) @ (q, j, 1)
            hidden_posact = np.insert(hidden_posact, 0, 1, 1)  # Add bias, j is now (n_hidden + 1)
            output = self.weights_1 @ hidden_posact
            errors = reference_outputs - output

            # Calculate cost function derivatives.
            # ====================================
            dcost_dweights_1 = np.einsum("qkm,qjm->kj", -errors, hidden_posact)

            # Hidden layer activation function derivative
            dvk_dwjk = (1 / np.cosh(hidden_preact)) ** 2

            # j is now n_hidden
            dcost_dweights_0 = np.einsum("qkm,kj,qjm,qim->ji", -errors, self.weights_1[:, 1:], dvk_dwjk, inputs)

            n_samples = inputs.shape[0]
            self.weights_0 -= eta * dcost_dweights_0 / n_samples
            self.weights_1 -= eta * dcost_dweights_1 / n_samples

            # Adapt learning rate
            error = np.sum(np.abs(errors))
            if last_error and alpha:
                if error < last_error:
                    eta *= alpha
                else:
                    eta /= alpha
            last_error = error

            # Yield gradient absolute sum to let the train(...) function log and check for end conditions.
            yield np.sum(np.abs(dcost_dweights_0)) + np.sum(np.abs(dcost_dweights_0))

    def _levenberg_marquardt(self,
                             inputs: np.ndarray,
                             reference_outputs: np.ndarray,
                             mu: float,
                             alpha: Optional[float],
                             **_):
        """
        Generator that performs one Levenberg-Marquardt step on each iteration.

        ..note: Do not call this function directly. Use train(...) instead.

        :param inputs: Must have shape (n_samples, n_input + 1, 1). It's assumed
                       that the bias input is already included.
        :param reference_outputs: Outputs for each input. Must have shape (n_samples, n_output, 1)
        :param eta: Damping parameter.
        :param alpha: Adaptive factor. If None, then the damping parameter will not be adapted.
        :yields: Absolute sum of the gradients on each iteration.
        """
        # q: n_samples
        # i: n_input + 1
        # j: n_hidden (later n_hidden+1)
        # k: n_output
        # m: 1
        # x: n_output * (n_hidden + 1)
        # y: n_hidden * (n_input + 1)

        last_error = None
        while True:
            # Evaluate network to get hidden layer, output and error values.
            # ==============================================================
            # (q, j, 1) = (j, i) @ (q, i, 1)
            hidden_preact = self.weights_0 @ inputs  # Hidden layer value before activation. vj
            hidden_posact = np.tanh(hidden_preact)  # Hidden layer value after activation. yj

            hidden_posact = np.insert(hidden_posact, 0, 1, 1)  # Add bias, j is now (n_hidden + 1)
            # (q, k, 1) = (k, j) @ (q, j, 1)
            output = self.weights_1 @ hidden_posact

            errors = np.sum(reference_outputs - output, axis=1)

            # Calculate error derivatives
            # ===========================
            derror_dweights_1 = -hidden_posact[..., 0]  # (q, j)

            # The derivatives are the same through all outputs. So just repeat the same gradients
            # for each output if we have multiple.
            if self.n_outputs > 1:
                # (q, x)
                derror_dweights_1 = np.hstack([derror_dweights_1] * self.n_outputs)

            dvk_dwjk = (1 / np.cosh(hidden_preact)) ** 2  # (q, j, 1),  j is now n_hidden

            derror_dweights_0 = np.einsum("kj,qjm,qim->qji", -self.weights_1[:, 1:], dvk_dwjk, inputs)  # (q, j, i)

            # Flatten derivatives so each row contains the derivatives for each sample.
            derror_dweights_0 = derror_dweights_0.reshape((derror_dweights_0.shape[0], -1))  # (q, y)

            # Stack derivatives for the input and output gains.
            j = np.hstack((derror_dweights_0, derror_dweights_1))

            # Calculate weight updates.
            # =========================
            delta_weights = np.linalg.inv(j.T @ j + mu * np.eye(j.shape[1])) @ j.T @ errors

            # Unpack and restructure deltas so they can be applied to the weights.
            delta_weights_0 = \
                delta_weights[:(self.n_hidden * (self.n_inputs + 1))].reshape((self.n_hidden, self.n_inputs + 1))
            delta_weights_1 = \
                delta_weights[(self.n_hidden * (self.n_inputs + 1)):].reshape((self.n_outputs, self.n_hidden + 1))

            self.weights_0 -= delta_weights_0
            self.weights_1 -= delta_weights_1

            # Adapt mu
            error = np.sum(np.abs(errors))
            if last_error and alpha:
                if error < last_error:
                    mu *= alpha
                else:
                    mu /= alpha

            yield np.sum(np.abs(derror_dweights_0)) + np.sum(np.abs(derror_dweights_1))
            last_error = error

    def evaluate_error(self, inputs, reference_outputs):
        """
        Evaluated the neural network and calculates the absolute sum of the errors.

        :param inputs: Must have shape (n_samples, n_inputs, 1).
        :param reference_outputs: Must have shape (n_samples, n_outputs, 1). Expected ouputs
                                  for the inputs.
        :return: Absolute sum of the errors.
        """
        return np.sum(np.abs(reference_outputs - self.evaluate(inputs))) / inputs.shape[0]

    @property
    def training_log(self):
        """
        `xarray.Dataset` of the training log.
        :return:
        """
        return pd.DataFrame([asdict(log_entry) for log_entry in self._training_log])

    def save_matlab(self):
        """
        Saves the matlab structure given in the assignment.
        :return:
        """
        mdict = {
            "name": self.name,
            "trainFunct": self.activation_functions,
            "trainAlg": self.training_algorithm,
            "IW": self.input_weights,
            "LW": self.output_weights,
            "b": [self.bias_weights_0, self.bias_weights_1],
            "range": self.range,
            "trainParam": self.training_parameters,
        }

        raise NotImplementedError()
