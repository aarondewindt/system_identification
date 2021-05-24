from typing import Union, Sequence
from pathlib import Path

import numpy as np
import numba as nb
import numba.experimental


class Model:
    def __init__(self,
                 input_weights: np.ndarray,
                 output_weights: np.ndarray,
                 bias_weights_0: np.ndarray,
                 bias_weights_1: np.ndarray,
                 ):
        self.input_weights = input_weights
        self.output_weights = output_weights
        self.bias_weights_0 = bias_weights_0
        self.bias_weights_1 = bias_weights_1

    @classmethod
    def new(cls,
            n_inputs: float,
            n_outputs: float,
            n_hidden: float,
            range: Union[np.ndarray, Sequence[float]],
            training_parameters,
            log_dir: Union[Path, str]):

        range = np.atleast_2d(range)
        assert range.shape == (n_inputs, 2)

        return cls(
            input_weights=np.random.rand(n_hidden, n_inputs),
            output_weights=np.random.rand(n_outputs, n_hidden),
            bias_weights_0=np.random.rand(n_hidden, 1),
            bias_weights_1=np.random.rand(n_outputs, 1),
        )

    def evaluate(self, input: np.ndarray):
        # Evaluate neural network
        hidden = np.tanh(self.input_weights @ input + self.bias_weights_0)
        return self.output_weights @ hidden + self.bias_weights_1

    def back_propagation(self,
                         inputs,
                         reference_outputs,
                         mu):
        hidden_input = self.input_weights @ inputs + self.bias_weights_0
        hidden_output: np.ndarray = np.tanh(hidden_input)
        output = self.output_weights @ hidden_output + self.bias_weights_1

        errors = reference_outputs - output

        derror_dbiasweight_1 = np.sum(-errors, 0)
        derror_dow = np.einsum("soj,shj->oh", -errors, hidden_output)

        # Hidden layer activation function derivative
        dvk_dwjk = (1 / np.cosh(hidden_input)) ** 2

        derror_biasweight_0 = np.einsum("soj,oh,shj->shj", -errors, self.output_weights, dvk_dwjk)
        derror_diw = np.einsum("shj,sij->hi", derror_biasweight_0, inputs)
        derror_biasweight_0 = np.sum(derror_biasweight_0, 0)

        n_samples = inputs.shape[0]
        self.bias_weights_0 -= mu * derror_biasweight_0 / n_samples
        self.bias_weights_1 -= mu * derror_dbiasweight_1 / n_samples
        self.input_weights -= mu * derror_diw / n_samples
        self.output_weights -= mu * derror_dow / n_samples


