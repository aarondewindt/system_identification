from typing import Optional
from functools import partial
from itertools import product

import numpy as np
from scipy.optimize import lsq_linear

from .base_model import BaseModel
from .utils.vdom import hyr


class LeastSquaresModel(BaseModel):
    name = "polynomial_model"
    activation_functions = None
    training_algorithm = "lstsqr"

    def __init__(self, func, n_inputs, range, description):
        """

        :param func: Function used to calculate the rows of the rows of the A matrix.
        """
        super().__init__(n_inputs, 1, range, description)

        self.func = func
        self.coefficients = None

    @classmethod
    def new_polynomial(cls,
                       n_inputs: int,
                       range: np.array,
                       order: int):
        """
        Create new LeastSquaresModel based on a polynomial.

        :param n_inputs: Number of inputs.
        :param range: Range of each input.
        :param order: Polynomial order.
        :return:
        """
        range = np.atleast_2d(range)
        assert range.shape == (n_inputs, 2), "There must be one row in range per input."

        # Function that calculates the terms of a multivariate power series with
        # all coefficients set to 1.

        # Example: n_input=2 and order=2 results in the following list. x and y are the inputs.
        # [1, y, y**2, x, x*y, x*y**2, x**2, x**2*y, x**2*y**2]
        # So if x=2 and y=3 the list will contain:
        # [1, 3, 9, 2, 6, 18, 4, 12, 36]
        def polynomial_func(inputs, n_inputs, order):
            return [np.prod([inp**exp for inp, exp in zip(inputs, exponentials)])
                    for exponentials in product(*(range(order+1) for i in range(n_inputs)))]

        return cls(func=partial(polynomial_func, n_inputs=n_inputs, order=order),
                   n_inputs=n_inputs,
                   range=range,
                   description=f"Least squares model fitting a `{order}` order polynomial.")

    def _repr_html_(self):
        return hyr(title="Least squares model", root_type=type(self), content={
            "description": self.description,
            "n_inputs": self.n_inputs,
            "n_output": self.n_outputs,
            "n_coefficients": self.coefficients.size if self.coefficients else self.coefficients,
            "range": self.range
        }).to_html()

    def evaluate(self, inputs: np.ndarray):
        if self.coefficients is None:
            raise RuntimeError("Least squares has not been run.")
        # Remove the extra dimension at the end.
        inputs = inputs[..., 0]
        # Calculate the input values for each term in the linear function.
        a_matrix = np.array([self.func(inp) for inp in inputs])
        # Multiple by coefficients, reshape and return
        return (a_matrix @ self.coefficients).reshape((-1, 1, 1))

    def train(self,
              inputs: np.ndarray,
              reference_outputs: np.ndarray,
              epochs: int = 1,
              method: str = "",
              goal: float = 0.,
              min_grad: float = 1e-10,
              train_log_freq: int = 1,
              evaluation_inputs: Optional[np.ndarray] = None,
              evaluation_reference_outputs: Optional[np.ndarray] = None,
              **_):
        """
        "Train" the polynomial coefficients using least squares. This functions and its parameters
        exist in order for the polynomial model to have the same interface as the neural network models.

        :param inputs: Training input data. Must have shape (n_samples, n_inputs, 1).
        :param reference_outputs: Training output data. Must have shape (n_samples, n_outputs, 1).
        :param epochs: Ignored.
        :param method: Ignored.
        :param goal: Ignored.
        :param min_grad: Ignored.
        :param train_log_freq: Ignored.
        :param evaluation_inputs: Optional evaluation input data. By default the training data is used.
        :param evaluation_reference_outputs: Optional evaluation input data. By default the training data is used.
        """

        assert (evaluation_inputs is None) == (evaluation_reference_outputs is None), \
            "Both the evaluation inputs and reference outputs must be given or omitted."

        evaluation_inputs = evaluation_inputs if evaluation_inputs is None else inputs
        evaluation_reference_outputs = evaluation_reference_outputs if evaluation_reference_outputs is None else reference_outputs

        # Remove the last dimension.
        inputs = inputs[..., 0]
        reference_outputs = reference_outputs[..., 0]

        a_matrix = np.array([self.func(inp) for inp in inputs])
        b = np.squeeze(reference_outputs)

        result = lsq_linear(a_matrix, b)

        self.coefficients = result.x

        self._training_log.clear()
        self.log(1, np.nan, self.evaluate_error(evaluation_inputs, evaluation_reference_outputs))

    def save_matlab(self):
        raise NotImplementedError()
