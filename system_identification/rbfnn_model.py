from typing import Union, Sequence, Optional, Tuple
from itertools import product

from tqdm.auto import trange
import numpy as np
import xarray as xr

from .base_model import BaseModel
from .utils.vdom import hyr
from .lsqr_model import LeastSquaresModel
from .utils.lloyds_algorithm import LloydsAlgorithm


class RadialBasisFunctionNeuralNetworkModel(BaseModel):
    name = "rbf"
    activation_functions = ("tansig", 'purelin')

    def __init__(self,
                 weights_a: np.ndarray,
                 weights_c: np.ndarray,
                 weights_w: np.ndarray,
                 input_range: np.ndarray,
                 description: str):
        """
        Dense neural network with radial basis function activation in the hidden layer.

        ..note: Do not use this constructor to create new RadialBasisFunctionNeuralNetwork instances. Use the
                RadialBasisFunctionNeuralNetwork.new_###(...) factory functions instead.

        :param weights_a: (n_hidden) vector. Amplitude of each RBF.
        :param weights_c: (n_hidden x n_input) matrix. Centers of each RBF.
        :param weights_w: (n_hidden x n_input) matrix. Width of each RBF.
        :param input_range: (n_input x 2) matrix with the range of each input in each row.
        """

        super().__init__(
            n_inputs=weights_c.shape[1],
            n_outputs=1,
            input_range=input_range,
            description=description,
        )

        assert weights_a.ndim == 1
        assert weights_w.shape[1] == self.n_inputs
        assert weights_w.shape[0] == weights_a.shape[0]

        self.weights_a = weights_a  #: RBF amplitudes
        self.weights_c = weights_c  #: RBF coordinates
        self.weights_w = weights_w  #: RBF widths

        self.n_hidden = weights_a.shape[0]  #: Number of RBF's
        self.range = input_range  # Range of inputs.

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
        return hyr(title="Radial Basis Function Neural Network", root_type=type(self), top_n_open=0, content={
            "description": self.description,
            "n_inputs": self.n_inputs,
            "n_hidden": self.n_hidden,
            "n_output": self.n_outputs,
            "training_parameters": self.training_parameters,
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
            input_range: Union[np.ndarray, Sequence[float]]):
        """
        Factory function used to create new RadialBasisFunctionNeuralNetwork instances with random RBF's.

        :param n_inputs: Number of inputs
        :param n_hidden: Number of RBF's in the hidden layer.
        :param input_range: Range of each input.
        :return: New RadialBasisFunctionNeuralNetwork instance.
        """

        input_range = np.atleast_2d(input_range)
        assert input_range.shape == (n_inputs, 2), "There must be one row in range per input."

        # TODO: Initialize weights_c to place the rbf's across the input range instead of just between [0, 1)

        return cls(
            weights_a=np.random.rand(n_hidden,),
            weights_c=np.random.rand(n_hidden, n_inputs),
            weights_w=np.random.rand(n_hidden, n_inputs),
            input_range=input_range,
            description="Feedforward neural network with random initial weights."
        )

    @classmethod
    def new_random_placement(cls,
                             n_inputs: int,
                             n_hidden: int,
                             width_range: Tuple[float, float],
                             amplitude_range: Tuple[float, float],
                             input_range: Union[np.ndarray, Sequence[float]]):
        """
        Factory function used to create new RadialBasisFunctionNeuralNetwork
        instances with randomly RBF's with the given amplitude and width.

        :param n_inputs: Number of inputs
        :param n_hidden: Number of RBF's in the hidden layer.
        :param width_range: RBF's width.
        :param amplitude_range: RBF's amplitude.
        :param input_range: (n_input x 2) matrix with the range of each input in each row.
        :return: New RadialBasisFunctionNeuralNetwork instance.
        """
        input_range = np.atleast_2d(input_range)

        # Generate vector of random amplitudes within range
        weights_a = np.random.uniform(*amplitude_range, (n_hidden,))

        # Generate random coordinates within the given inputs range.
        weights_c = np.hstack([np.random.uniform(*ir, (n_hidden, 1)) for ir in input_range])

        # Generate random RBF widths within the given range.
        weights_w = np.random.uniform(*width_range, (n_hidden, n_inputs))

        return cls(
            weights_a=weights_a,
            weights_c=weights_c,
            weights_w=weights_w,
            input_range=input_range,
            description="Feedforward neural network with random initial weights."
        )

    @classmethod
    def new_grid_placement(cls,
                           n_inputs: int,
                           grid_size: Sequence[int],
                           input_range: Union[np.ndarray, Sequence[float]],
                           width_range: Tuple[float, float],
                           amplitude_range: Tuple[float, float]):
        """
        Factory function used to create new RadialBasisFunctionNeuralNetwork
        instances with RBF's on a uniform grid with the given amplitude and width.

        :param n_inputs: Number of inputs
        :param grid_size: (n_input) vector with the number of RBF's per input on the grid.
        :param input_range: (n_input x 2) matrix with the range of each input in each row.
        :param rbf_width: RBF's width.
        :param rbf_amplitude: RBF's amplitude.
        :return:
        """

        # Make sure we have one min/max range for each input.
        input_range = np.atleast_2d(input_range)
        assert input_range.shape == (n_inputs, 2)
        assert len(grid_size) == n_inputs

        # List containing vectors with the grid coordinates.
        # Eg. Assume 2 inputs, a (2x3) grid size and inputs range [(1, 4), (-1, 1)]
        # Then `rbf_coordinates` will contain these two vectors.
        # [1, 4] and [-1, 0, 1]
        rbf_coordinates = [np.linspace(*input_range[i], grid_size[i]).tolist() for i in range(n_inputs)]

        # Loop through all combinations of coordinates to get the
        # coordinates at each node in the grid.
        # Put these coordinates in each row of the centers matrix.
        weights_c = np.array(list(product(*rbf_coordinates)))

        # The number of hidden cells, aka the number of rbf's, is now known.
        n_hidden = weights_c.shape[0]

        weights_a = np.random.uniform(*amplitude_range, (n_hidden,))
        weights_w = np.random.uniform(*width_range, (n_hidden, n_inputs))

        return cls(
            weights_a=weights_a,
            weights_c=weights_c,
            weights_w=weights_w,
            input_range=input_range,
            description="Radial basis function neural network with "
                        "the initial centers placed in a uniform grid."
        )

    @classmethod
    def new_centroidal_voronoi_tessellation_placement(cls,
                                           n_hidden: int,
                                           input_range: Union[np.ndarray, Sequence[float]],
                                           width_range: Tuple[float, float],
                                           amplitude_range: Tuple[float, float],
                                           iterations: int=15):
        """
        Factory function used to create new RadialBasisFunctionNeuralNetwork
        instances with RBF's placed on the centroids of the elements in an
        approximate centroidal Voronoi tessellation.

        note: Only 2 inputs supported.

        :param n: Number of RBF's
        :param bounding_box:
        :param iterations:
        :return:
        """
        n_inputs = 2
        assert len(input_range) == n_inputs, "Only 2 inputs supported."

        field = LloydsAlgorithm(n_hidden, (*input_range[0], *input_range[1]))
        field.relax_points(iterations)
        weights_c = field.points_as_array

        return cls(
            weights_a=np.random.rand(n_hidden,),
            weights_c=weights_c,
            weights_w=np.random.rand(n_hidden, n_inputs),
            input_range=input_range,
            description="Radial basis function neural network with "
                        "the initial centers placed a centroidal Voronoi tessellation."
        )

    @classmethod
    def new_from_other_at_epoch(cls, other: "RadialBasisFunctionNeuralNetworkModel", epoch=None) \
            -> "RadialBasisFunctionNeuralNetworkModel":
        """
        Create a new RadialBasisFunctionNeuralNetworkModel instance using the 
        weights from another RadialBasisFunctionNeuralNetworkModel at the specified epoch.
        """
        if epoch is None:
            training_log = other._training_log
        else:
            training_log = [log for log in other._training_log if log.epoch <= epoch]

        print(len(training_log))

        model = cls(
            weights_a=training_log[-1].weights['weights_a'],
            weights_c=training_log[-1].weights['weights_c'],
            weights_w=training_log[-1].weights['weights_w'],
            input_range=other.input_range,
            description="",
        )

        model._training_log = training_log

        return model

    @classmethod
    def new_from_training_log(cls, training_log: xr.Dataset, input_range=None, epoch: int=-1):
        """
        Create a new RadialBasisFunctionNeuralNetworkModel instance using the
        weights in a model training log at the specified epoch.

        :param epoch: Epoch from which to copy the model weights. By default it'll be
                      the last epoch.
        """
        weights_c = training_log.isel(epoch=epoch).weights_c

        if input_range is None:
            n_inputs = weights_c.shape[1]
            input_range = np.array([[-np.inf, np.inf] for _ in range(n_inputs)])

        return cls(
            weights_a=training_log.isel(epoch=epoch).weights_a,
            weights_c=training_log.isel(epoch=epoch).weights_c,
            weights_w=training_log.isel(epoch=epoch).weights_w,
            input_range=input_range,
            description="",
        )

    def clone_clean(self) -> 'RadialBasisFunctionNeuralNetworkModel':
        return RadialBasisFunctionNeuralNetworkModel(
            weights_a=self.weights_a,
            weights_c=self.weights_c,
            weights_w=self.weights_w,
            input_range=self.input_range,
            description=self.description,
        )

    # def __getstate__(self):
    #     pass
    #
    # def __setstate__(self, state):
    #     pass

    def evaluate(self, inputs: np.ndarray):
        """
        Evaluate model using the given inputs.

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
              validation_inputs: Optional[np.ndarray]=None,
              validation_outputs: Optional[np.ndarray]=None,
              verbose=True,
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
        :param validation_inputs: Optional evaluation input data. By default the training data is used.
        :param validation_outputs: Optional evaluation input data. By default the training data is used.
        :param kwargs: Extra parameters passed to the training algorithm functions.
                       trainlm` accepts `mu` and `alpha`. `alpha` may be set to `None` to make it non-adaptive.
        """

        assert (validation_inputs is None) == (validation_outputs is None), \
            "Both the evaluation inputs and reference outputs must be given or omitted."

        self._training_log.clear()

        self.training_parameters["training_algorithm"] = method
        self.training_parameters["goal"] = goal
        self.training_parameters["min_grad"] = min_grad
        self.training_parameters["epochs"] = epochs
        self.training_parameters.update(kwargs)

        if method == "trainlsqr":
            self._least_squares(inputs, reference_outputs, validation_inputs, validation_outputs)
            return

        elif method == "trainlm":
            if "mu" not in kwargs:
                raise ValueError("Parameter `mu` required for method `trainlm`.")
            train_function = self._levenberg_marquardt

        else:
            raise ValueError(f"Unknown training method {method}.")



        for i, grad in zip(trange(epochs - self.epochs, disable=not verbose, leave=False), train_function(inputs, reference_outputs, **kwargs)):
            if self.epochs >= epochs:
                print("Max number of epochs reached")
                break

            self.epochs += 1

            if grad <= self.training_parameters["min_grad"]:
                print("Minimum gradients reached")
                break

            if not (i % train_log_freq):
                training_data_errors, validation_data_errors = \
                    self.evaluate_errors(inputs, reference_outputs, validation_inputs, validation_outputs)

                error_mean = np.mean(abs(training_data_errors))
                if error_mean < self.training_parameters["goal"]:
                    print("Goal met")
                    break

                self.log(
                    self.epochs,
                    grad,
                    training_data_errors, validation_data_errors,
                    {
                        "weights_a": self.weights_a.copy(),
                        "weights_c": self.weights_c.copy(),
                        "weights_w": self.weights_w.copy(),
                    }
                )

    def _least_squares(self,
                       inputs: np.ndarray,
                       reference_outputs: np.ndarray,
                       evaluation_inputs: np.ndarray,
                       evaluation_reference_outputs: np.ndarray):
        def rbf_func(*args):
            inp = np.array(args)
            vj = np.einsum("ji,ji->j", self.weights_w**2, (inp - self.weights_c)**2)
            return np.exp(-vj)

        # Use the least squared model to solve this.
        lsq_model = LeastSquaresModel(
            func=rbf_func,
            n_inputs=self.n_inputs,
            input_range=self.range,
            description="",
        )

        lsq_model.train(
            inputs=inputs,
            reference_outputs=reference_outputs,
            validation_inputs=evaluation_inputs,
            validation_outputs=evaluation_reference_outputs,
        )

        # Sanity check
        assert lsq_model.coefficients.shape == self.weights_a.shape

        self.weights_a = lsq_model.coefficients
        self._training_log = lsq_model._training_log

    def _levenberg_marquardt(self,
                             inputs: np.ndarray,
                             reference_outputs: np.ndarray,
                             mu: float,
                             alpha: Optional[float],
                             **_):
        """
        Generator that performs one Levenberg-Marquardt step on each iteration.

        ..note: Do not call this function directly. Use train(...) instead.

        :param inputs: Must have shape (n_samples, n_input, 1).
        :param reference_outputs: Outputs for each input. Must have shape (n_samples, n_output, 1)
        :param mu: Damping parameter.
        :param alpha: Adaptive factor. If None, then the damping parameter will not be adapted.
        :yields: Absolute sum of the gradients on each iteration.
        """
        # Dimension sizes.
        # s: n_samples
        # i: n_input
        # j: n_hidden
        # k: n_output
        # m: 1

        # s1i
        inputs = inputs.transpose((0, 2, 1))

        last_error = None

        # Infinite loop. This is a generator that will indefinitely update the
        # weights on each iteration. The train(...) function handles the end
        # conditions.
        while True:
            # Evaluate network
            # ================
            # xi - cij
            # sji = s1i - ji
            vij = inputs - self.weights_c

            # sji
            yij = vij**2

            # sj
            zj = np.einsum("ji,sji->sj", self.weights_w ** 2, yij)

            # sj
            yj = np.exp(-zj)

            # s
            yk = np.einsum("j,sj->s", self.weights_a, yj)

            # Calculate derivatives
            # =====================
            # sj
            de_da = -yj

            # Should be `aj @ yij @ exp(-zj)`, but since `yj = np.exp(-zj)`
            # then we use `aj @ yij @ yj`
            # sji
            de_dwij = np.einsum("j,sji,sj->sji", self.weights_a, yij, yj)

            # `aj @ wij @ vij @ exp(-zj) = aj @ wij @ vij @ vj`
            # sji
            de_dcij = -2 * np.einsum("j,ji,sji,sj->sji", self.weights_a, self.weights_w, vij, yj)

            # s(i*j)
            de_dwij = de_dwij.reshape((de_dwij.shape[0], -1))
            de_dcij = de_dcij.reshape((de_dcij.shape[0], -1))

            j = np.hstack((de_da, de_dwij, de_dcij))

            # Calculate weights updates
            # ========================
            # s1
            errors = np.sum(reference_outputs - yk.reshape((-1, 1, 1)), axis=1)
            delta_weights = np.linalg.inv(j.T @ j + mu * np.eye(j.shape[1])) @ j.T @ errors

            # Split weight updates and apply
            delta_a = delta_weights[:self.n_hidden].squeeze()
            delta_w = delta_weights[self.n_hidden:(self.n_hidden + self.n_hidden * self.n_inputs)]\
                .reshape((self.n_hidden, self.n_inputs))
            delta_c = delta_weights[(self.n_hidden + self.n_hidden * self.n_inputs):] \
                .reshape((self.n_hidden, self.n_inputs))

            self.weights_a -= delta_a
            self.weights_w -= delta_w
            self.weights_c -= delta_c

            # Adapt mu
            error = np.sum(np.abs(errors))
            if last_error and alpha:
                if error < last_error:
                    mu *= alpha
                else:
                    mu /= alpha
            last_error = error

            # Yield gradient absolute sum to let the train(...) function log and check for end conditions.
            yield np.sum(np.abs(de_da)) + np.sum(np.abs(de_dwij)) + np.sum(np.abs(de_dcij))

    def save_matlab(self):
        """
        Saves the matlab structure given in the assignment.
        :return:
        """

        raise NotImplementedError()
