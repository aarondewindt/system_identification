from typing import Union, Sequence, Optional
from dataclasses import asdict
from pathlib import Path
from itertools import product

from tqdm.auto import trange
import numpy as np
import pandas as pd

from .base_model import BaseModel
from .utils.vdom import hyr
from .lsqr_model import LeastSquaresModel


class RadialBasisFunctionNeuralNetwork(BaseModel):
    name = "rbf"
    activation_functions = ("tansig", 'purelin')

    def __init__(self,
                 weights_a: np.ndarray,
                 weights_c: np.ndarray,
                 weights_w: np.ndarray,
                 range: np.ndarray,
                 log_dir: Union[Path, str],
                 description: str):
        """
        Dense neural network with radial basis function activation in the hidden layer.

        ..note: Do not use this constructor to create new RadialBasisFunctionNeuralNetwork instances. Use the
                RadialBasisFunctionNeuralNetwork.new(...) factory function instead.

        :param weights_0: (n_hidden x (n_input + 1)) matrix. Weights between the input and hidden layer.
        :param weights_1: (n_output x (n_hidden + 1)) matrix. Weights between the hidden and output layer.
        :param range: (n_input x 2) matrix with the range of each input in each row.
        :param log_dir: Directory to store network training history.
        """

        super().__init__(
            n_inputs=weights_c.shape[1],
            n_outputs=1,
            range=range,
            description=description,
        )

        assert weights_a.ndim == 1
        assert weights_w.shape[1] == self.n_inputs
        assert weights_w.shape[0] == weights_a.shape[0]

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.weights_a = weights_a  #: RBF amplitudes
        self.weights_c = weights_c  #: RBF coordinates
        self.weights_w = weights_w  #: RBF widths

        self.n_hidden = weights_a.shape[0]  #: Number of RBF's
        self.range = range  # Range of inputs.

        self.epochs = 0

        self.training_parameters = {
            "training_algorithm": None,
            "goal": None,
            "min_grad": None,
        }

    def __repr__(self):
        return f"<RadialBasisFunctionNeuralNetwork\n" \
               f" n_inputs={self.n_inputs}\n" \
               f" n_hidden={self.n_hidden}\n" \
               f" n_outputs={self.n_outputs}>"

    def _repr_html_(self):
        return hyr(title="Radial Basis Function Neural Network", root_type=type(self), content={
            "description": self.description,
            "n_inputs": self.n_inputs,
            "n_hidden": self.n_hidden,
            "n_output": self.n_outputs,
            "training_parameters": self.training_parameters,
            "log_dir": self.log_dir,
            "weights": {
                "weights_a": self.weights_a,
                "weights_c": self.weights_c,
                "weights_w": self.weights_w,
            }
        }).to_html()

    @classmethod
    def new(cls,
            n_inputs: int,
            n_hidden: int,
            range: Union[np.ndarray, Sequence[float]],
            log_dir: Union[Path, str]):
        """
        Factory function used to create new RadialBasisFunctionNeuralNetwork instances with random weights.

        :param n_inputs: Number of inputs
        :param n_outputs: Number of outputs
        :param n_hidden: Number of cells in the hidden layer.
        :param range: Range of each input.
        :param log_dir: Directory to store network training history.
        :return: New RadialBasisFunctionNeuralNetwork instance.
        """

        range = np.atleast_2d(range)
        assert range.shape == (n_inputs, 2), "There must be one row in range per input."

        # TODO: Initialize weights_c to place the rbf's across the input range instead of just between [0, 1)

        return cls(
            weights_a=np.random.rand(n_hidden,),
            weights_c=np.random.rand(n_hidden, n_inputs),
            weights_w=np.random.rand(n_hidden, n_inputs),
            range=range,
            log_dir=log_dir,
            description="Feedforward neural network with random initial weights."
        )

    @classmethod
    def new_grid_placement(cls,
                           n_inputs: int,
                           grid_size: Sequence[int],
                           input_range: Union[np.ndarray, Sequence[float]],
                           rbf_width: float,
                           rbf_amplitude: float,
                           log_dir: Union[Path, str]):

        # List containing vectors with the grid coordinates.
        # Eg. Assume 2 inputs, a (2x3) grid size and inputs range [(1, 4), (-1, 1)]
        # Then `rbf_coordinates` will contain these two vectors.
        # [1, 4] and [-1, 0, 1]
        rbf_coordinates = [np.linspace(*input_range[i], grid_size[i]).tolist() for i in range(n_inputs)]

        # Loop through all combinations of coordinates to get the
        # coordinates at each node in the grid.
        # Put these coordinates in each row of the weight matrix
        weights_c = np.array(list(product(*rbf_coordinates)))

        # The number of hidden cells, aka the number of rbf's, is now known.
        n_hidden = weights_c.shape[0]

        return cls(
            weights_a=np.ones((n_hidden,)) * rbf_amplitude,
            weights_c=weights_c,
            weights_w=np.ones((n_hidden, n_inputs)) * rbf_width,
            range=input_range,
            log_dir=log_dir,
            description="Radial basis function neural network with random initial weights."
        )

    def evaluate(self, inputs: np.ndarray):
        """
        Shape of input must be: (n_samples, n_inputs, 1).
        Output shape will be: (n_samples, n_outputs, 1).

        :param inputs: Inputs to evaluate.
        :return:
        """
        # Transpose inputs so it has shape (n_samples, 1, n_inputs)
        # This is to make sure the centers can be properly broadcasted
        # during the following computation.
        inputs = inputs.transpose((0, 2, 1))

        # sum(wij**2 * (xi - cij)**2)
        vj = np.einsum("ji,sji->sj", self.weights_w**2, (inputs - self.weights_c)**2)

        # a * exp(-vj)
        return np.einsum("j,sj->s", self.weights_a, np.exp(-vj)).reshape(-1, 1, 1)

    def train(self,
              inputs: np.ndarray,
              reference_outputs: np.ndarray,
              epochs: int=1,
              method: str="trainlsqr",
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
        :param method: Training algorithm. Valid values are `trainlsqr` to calculate the amplitudes using
                       least squares and `trainlm` for Levenberg Marquardt.
        :param goal: Target maximum error. The training will stop once the evaluated error is under the goal.
        :param min_grad: Minimum gradient. The training will stop once the absolute sum gradients is below
                         this value.
        :param train_log_freq: Number of epochs between error evaluation and logging steps.
        :param evaluation_inputs: Optional evaluation input data. By default the training data is used.
        :param evaluation_reference_outputs: Optional evaluation input data. By default the training data is used.
        :param kwargs: Extra parameters passed to the training algorithm functions.
                       trainlm` accepts `mu` and `alpha`. `alpha` may be set to `None` to make it non-adaptive.
        """

        assert (evaluation_inputs is None) == (evaluation_reference_outputs is None), \
            "Both the evaluation inputs and reference outputs must be given or omitted."

        evaluation_inputs = evaluation_inputs or inputs
        evaluation_reference_outputs = evaluation_reference_outputs or reference_outputs

        if method == "trainlsqr":
            self._least_squares(inputs, reference_outputs, evaluation_inputs, evaluation_reference_outputs)

    def _least_squares(self,
                       inputs: np.ndarray,
                       reference_outputs: np.ndarray,
                       evaluation_inputs: np.ndarray,
                       evaluation_reference_outputs: np.ndarray):
        def rbf_func(inp):
            vj = np.einsum("ji,ji->j", self.weights_w**2, (inp - self.weights_c)**2)
            return self.weights_a * np.exp(-vj)

        # Use the least squared model to solve this.
        lsq_model = LeastSquaresModel(
            func=rbf_func,
            n_inputs=self.n_inputs,
            range=self.range,
            description="",
        )

        lsq_model.train(
            inputs=inputs,
            reference_outputs=reference_outputs,
            evaluation_inputs=evaluation_inputs,
            evaluation_reference_outputs=evaluation_reference_outputs,
        )

        # Sanity check
        assert lsq_model.coefficients.shape == self.weights_a.shape

        self.weights_a = lsq_model.coefficients
        self._training_log = lsq_model._training_log

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

        raise NotImplementedError()
